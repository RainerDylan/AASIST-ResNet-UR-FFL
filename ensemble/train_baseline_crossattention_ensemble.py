import sys
import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..")) if "ensemble" in CURRENT_DIR else CURRENT_DIR
sys.path.append(ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR  = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

from src.data.dataset        import ASVspoofDataset
from src.models.aasist       import AASIST
from src.models.resnet_simam import resnet18_simam

PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PREPROCESSED_DEV_DIR   = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
PROTOCOL_DEV   = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"
OUTPUT_WEIGHTS = os.path.join(MODELS_DIR, "crossattention_ensemble_baseline_best.pth")

TOTAL_EPOCHS  = 100
BATCH_SIZE    = 16
# Differential LR — base models need strong cold-start gradient signal;
# fusion head must generalise slowly to avoid memorising embedding patterns.
LR_BASE   = 5e-4   # AASIST and ResNet
LR_FUSER  = 5e-5   # CrossAttentionFuser  (10× lower than base models)
WEIGHT_DECAY  = 3e-4
WARMUP_EPOCHS = 5
PATIENCE      = 20

META_W = 0.50   # fusion head loss weight
AUX_W  = 0.50   # sum of base-model losses  (0.25 per model)


# ── Loss ─────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.10):
        super().__init__()
        self.gamma = gamma
        self.ls    = label_smoothing

    def forward(self, logits, targets):
        n = logits.shape[1]
        with torch.no_grad():
            s = torch.zeros_like(logits).fill_(self.ls / (n - 1))
            s.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        lp = F.log_softmax(logits, dim=1)
        w  = (1.0 - (lp.exp() * s).sum(1)).pow(self.gamma)
        return (w * -(s * lp).sum(1)).mean()


# ── Architecture ─────────────────────────────────────────────────────────────

class CrossAttentionFuser(nn.Module):
    # FIX (global): dropout raised to 0.50 — fusion head is the smallest
    # component and most prone to memorising embedding-level patterns.
    def __init__(self, dim_a=104, dim_r=512, embed_dim=256,
                 num_heads=8, num_classes=2, dropout=0.50):
        super().__init__()
        self.proj_a    = nn.Sequential(nn.Linear(dim_a, embed_dim), nn.LayerNorm(embed_dim))
        self.proj_r    = nn.Sequential(nn.Linear(dim_r, embed_dim), nn.LayerNorm(embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                         dim_feedforward=embed_dim*4, dropout=dropout,
                                         batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))

    def forward(self, ea, er):
        if ea.dtype != er.dtype: ea = ea.to(er.dtype)
        B   = ea.size(0)
        seq = torch.cat([self.cls_token.expand(B,-1,-1),
                         self.proj_a(ea).unsqueeze(1),
                         self.proj_r(er).unsqueeze(1)], dim=1)
        return self.head(self.transformer(seq)[:, 0, :])


class EndToEndEnsemble(nn.Module):
    def __init__(self, aasist, resnet, fuser):
        super().__init__()
        self.aasist = aasist; self.resnet = resnet; self.fusion_head = fuser
        self._ea = [None]; self._er = [None]
        self._ha = aasist.fc.register_forward_hook(lambda m,i,o: self._ea.__setitem__(0,i[0]))
        self._hr = resnet.fc.register_forward_hook(lambda m,i,o: self._er.__setitem__(0,i[0]))

    def forward(self, wav, mel, return_base=False):
        oa = self.aasist(wav)
        with torch.amp.autocast("cuda"):
            or_ = self.resnet(mel)
            om  = self.fusion_head(self._ea[0], self._er[0])
        return (om, oa, or_) if return_base else om

    def remove_hooks(self): self._ha.remove(); self._hr.remove()


# ── Cold-start init ───────────────────────────────────────────────────────────

def _init_aasist(m):
    attn = ("attn","gat","attention","query","key","value")
    for n, mod in m.named_modules():
        ia = any(k in n.lower() for k in attn)
        if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            (nn.init.xavier_uniform_ if ia else
             lambda w: nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu"))(mod.weight)
            if mod.bias is not None: nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Linear):
            (nn.init.xavier_uniform_ if ia else
             lambda w: nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu"))(mod.weight)
            if mod.bias is not None: nn.init.zeros_(mod.bias)
        elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(mod.weight); nn.init.zeros_(mod.bias)

def _init_resnet(m):
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
            if mod.bias is not None: nn.init.zeros_(mod.bias)
        elif isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(mod.weight); nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, 0., 0.01)
            if mod.bias is not None: nn.init.zeros_(mod.bias)

def _init_fuser(f):
    for sm in (f.proj_a, f.proj_r):
        for mod in sm.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_normal_(mod.weight, mode="fan_in", nonlinearity="relu")
                if mod.bias is not None: nn.init.zeros_(mod.bias)
    lins = [mod for mod in f.head.modules() if isinstance(mod, nn.Linear)]
    for i, lin in enumerate(lins):
        (nn.init.kaiming_normal_(lin.weight, mode="fan_in", nonlinearity="relu")
         if i < len(lins)-1 else nn.init.normal_(lin.weight, 0., 0.01))
        if lin.bias is not None: nn.init.zeros_(lin.bias)


# ── Utilities ─────────────────────────────────────────────────────────────────

def create_train_sampler(dataset):
    labels = dataset.labels
    counts = torch.bincount(torch.tensor(labels))
    w = len(labels) / counts.float()
    return WeightedRandomSampler([w[l] for l in labels], len(labels), replacement=True)


def create_balanced_val_indices(dataset):
    """
    FIX (global): The ASVspoof2019 LA dev set is ~8.8× spoof-heavy (2548 genuine,
    22296 spoof).  Computing EER on this imbalanced set is fundamentally unreliable:
    a model that predicts EVERYTHING as spoof achieves near-0% EER because the
    threshold where FPR≈FNR lands deep in the spoof-score distribution.  This
    creates a false checkpoint signal — the 'best' saved model is actually the most
    spoof-biased one, which then collapses (EER=40%+) on the balanced evaluation set.

    Fix: subsample the dev set to a perfectly balanced subset (n genuine + n spoof,
    where n = total genuine count ≈ 2548).  All validation metrics (EER, AUC, minDCF)
    are now computed on this balanced subset, matching the conditions of the external
    evaluation and making the checkpoint criterion trustworthy.
    """
    labels    = np.array(dataset.labels)
    gen_idx   = np.where(labels == 0)[0]
    spoof_idx = np.where(labels == 1)[0]
    n   = min(len(gen_idx), len(spoof_idx))
    rng = np.random.RandomState(42)
    idx = np.concatenate([rng.choice(gen_idx,   n, replace=False),
                          rng.choice(spoof_idx, n, replace=False)])
    rng.shuffle(idx)
    print(f"  Balanced val subset: {n} genuine + {n} spoof = {2*n} total "
          f"(from {len(gen_idx)} genuine, {len(spoof_idx)} spoof in dev set)")
    return idx.tolist()


def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1. - tpr
    return max(0., float(fpr[np.nanargmin(np.abs(fnr - fpr))]) * 100.)

def compute_min_dcf(y_true, y_scores, p=0.05, cm=1., cf=1.):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1. - tpr
    return float(np.min(cm*fnr*p + cf*fpr*(1.-p)) / min(cm*p, cf*(1.-p)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Baseline Cold-Start Cross-Attention Ensemble — {device}")
    print("=" * 70)

    aasist = AASIST(stft_window=698, stft_hop=398, freq_bins=116,
                    gat_layers=2, heads=5, head_dim=104, hidden_dim=455, dropout=0.33)
    resnet = resnet18_simam(num_classes=2, dropout_rate=0.22)
    fuser  = CrossAttentionFuser()

    _init_aasist(aasist); _init_resnet(resnet); _init_fuser(fuser)
    print("  AASIST : Kaiming(Conv/Linear) + Xavier(GAT/attn)")
    print("  ResNet : Kaiming(Conv) + small-normal(Linear)")
    print("  Fuser  : Kaiming(proj) + small-normal(head)")

    model = EndToEndEnsemble(aasist, resnet, fuser).to(device)
    print(f"  Params : {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 70)

    train_ds = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    val_ds   = ASVspoofDataset(PREPROCESSED_DEV_DIR,   PROTOCOL_DEV)

    train_sampler = create_train_sampler(train_ds)
    val_indices   = create_balanced_val_indices(val_ds)    # global fix
    val_sampler   = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                              num_workers=4, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, sampler=val_sampler,
                              num_workers=4, pin_memory=False)

    mel_t  = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80).to(device)
    a2db   = T.AmplitudeToDB(stype="power", top_db=80).to(device)

    # FIX (global): differential LR — base models 10× higher than fusion head.
    # Cold-start AASIST and ResNet need strong gradient to build useful embeddings.
    # The fusion head must learn slowly so it generalises rather than memorising
    # the embedding-space patterns of the training/dev utterances.
    optimizer = optim.AdamW([
        {'params': model.aasist.parameters(),      'lr': LR_BASE,  'weight_decay': WEIGHT_DECAY},
        {'params': model.resnet.parameters(),       'lr': LR_BASE,  'weight_decay': WEIGHT_DECAY},
        {'params': model.fusion_head.parameters(),  'lr': LR_FUSER, 'weight_decay': WEIGHT_DECAY},
    ])
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                         total_iters=WARMUP_EPOCHS)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=TOTAL_EPOCHS - WARMUP_EPOCHS,
                                                   eta_min=1e-7)
    sched     = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [WARMUP_EPOCHS])
    criterion = FocalLoss()
    scaler    = torch.amp.GradScaler("cuda")

    best_eer, no_imp = float("inf"), 0
    t0 = time.time()
    H  = {k: [] for k in ["tr_loss","vl_loss","l_meta","l_aux",
                            "tr_eer","vl_eer","vl_auc","vl_mdc"]}

    for ep in range(TOTAL_EPOCHS):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        st = sm = sa = 0.; tl = []; tp = []; nan_n = 0

        bar = tqdm(train_loader, desc=f"Ep {ep+1}/{TOTAL_EPOCHS} [Tr]")
        for wav, lbl in bar:
            wav = wav.squeeze(1).to(device); lbl = lbl.to(device)
            with torch.no_grad():
                mel = a2db(mel_t(wav)).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            om, oa, or_ = model(wav, mel, return_base=True)
            omf = om.float(); oaf = oa.float(); orf = or_.float()

            lm  = criterion(omf, lbl)
            la  = criterion(oaf, lbl) + criterion(orf, lbl)
            lt  = META_W * lm + AUX_W * la

            if torch.isnan(lt) or torch.isinf(lt):
                nan_n += 1; optimizer.zero_grad(set_to_none=True)
                if nan_n <= 3: print(f"\n  [NaN] ep{ep+1} — skipping batch")
                continue

            scaler.scale(lt).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            st += lt.item(); sm += lm.item(); sa += la.item()
            with torch.no_grad():
                tl.extend(lbl.cpu().numpy())
                tp.extend(torch.softmax(omf,1)[:,1].cpu().numpy())
            bar.set_postfix({"L": f"{lt.item():.4f}", "Lm": f"{lm.item():.4f}"})

        nb = max(1, len(train_loader) - nan_n)
        H["tr_loss"].append(st/nb); H["l_meta"].append(sm/nb); H["l_aux"].append(sa/nb)
        H["tr_eer"].append(compute_eer(tl, tp))

        # ── Validation (balanced subset) ──────────────────────────────────────
        model.eval(); sv = 0.; vl = []; vp = []
        with torch.no_grad():
            for wav, lbl in tqdm(val_loader, desc=f"Ep {ep+1} [Val]", leave=False):
                wav = wav.squeeze(1).to(device); lbl = lbl.to(device)
                mel = a2db(mel_t(wav)).unsqueeze(1)
                out = model(wav, mel)
                sv += criterion(out.float(), lbl).item()
                vl.extend(lbl.cpu().numpy())
                vp.extend(torch.softmax(out.float(),1)[:,1].cpu().numpy())

        veer = compute_eer(vl, vp)
        vauc = roc_auc_score(vl, vp)
        vmdc = compute_min_dcf(vl, vp)
        H["vl_loss"].append(sv/len(val_loader)); H["vl_eer"].append(veer)
        H["vl_auc"].append(vauc); H["vl_mdc"].append(vmdc)

        # Spoof-bias detector — prints mean P(spoof) per true class to surface collapse
        arr_lbl = np.array(vl); arr_prb = np.array(vp)
        mean_gen  = arr_prb[arr_lbl==0].mean() if (arr_lbl==0).any() else float('nan')
        mean_spf  = arr_prb[arr_lbl==1].mean() if (arr_lbl==1).any() else float('nan')

        sched.step()
        lr0 = optimizer.param_groups[0]["lr"]
        lr2 = optimizer.param_groups[2]["lr"]
        eta = str(datetime.timedelta(seconds=int(
            (time.time()-t0)/(ep+1)*(TOTAL_EPOCHS-ep-1))))

        print(f"Ep {ep+1:3d} | LR_base {lr0:.1e} LR_fuse {lr2:.1e} | "
              f"Tr {st/nb:.4f} | Val {sv/len(val_loader):.4f} | "
              f"EER {veer:.3f}% | AUC {vauc:.4f} | mDCF {vmdc:.4f} | ETA {eta}")
        print(f"       P(spoof)|genuine={mean_gen:.3f}  P(spoof)|spoof={mean_spf:.3f}  "
              f"[gap={mean_spf-mean_gen:.3f}]")

        if veer < best_eer:
            best_eer, no_imp = veer, 0
            torch.save({"epoch": ep+1, "model_state_dict": model.state_dict(),
                        "val_eer": veer, "val_auc": vauc, "val_min_dcf": vmdc},
                       OUTPUT_WEIGHTS)
            print(f"  -> EER {best_eer:.4f}% — saved ✓")
        else:
            no_imp += 1
            print(f"  -> No improvement ({no_imp}/{PATIENCE})")

        if no_imp >= PATIENCE:
            print("\nEarly stopping."); break

    model.remove_hooks()
    print(f"\nDone: {str(datetime.timedelta(seconds=int(time.time()-t0)))} | "
          f"Best EER: {best_eer:.4f}%")

    # ── Plots ─────────────────────────────────────────────────────────────────
    E = range(1, len(H["tr_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    ax = axes.flatten()

    ax[0].plot(E, H["tr_loss"], label="Train Loss",          color="blue")
    ax[0].plot(E, H["vl_loss"], label="Val Loss (balanced)", color="red",        ls="--")
    ax[0].plot(E, H["l_meta"],  label="L_meta (fusion)",     color="steelblue",  ls=":")
    ax[0].plot(E, H["l_aux"],   label="L_aux (base models)", color="darkorange", ls=":")
    ax[0].set_title("Loss Trajectory"); ax[0].set_xlabel("Epoch")
    ax[0].legend(fontsize=8); ax[0].grid(True, ls=":", alpha=0.6)

    ax[1].plot(E, H["tr_eer"], label="Train EER %",          color="teal")
    ax[1].plot(E, H["vl_eer"], label="Val EER % (balanced)", color="purple", ls="--")
    ax[1].set_title("Equal Error Rate — Balanced Val (↓ Better)")
    ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid(True, ls=":", alpha=0.6)

    ax[2].plot(E, H["vl_auc"], label="ROC-AUC", color="darkgreen")
    ax[2].axhline(0.5, ls=":", color="gray", alpha=0.5, label="Random")
    ax[2].set_ylim(0.4, 1.02)
    ax[2].set_title("Validation AUC — Balanced Val (↑ Better)")
    ax[2].set_xlabel("Epoch"); ax[2].legend(); ax[2].grid(True, ls=":", alpha=0.6)

    ax[3].plot(E, H["vl_mdc"], label="minDCF", color="darkred")
    ax[3].axhline(1., ls=":", color="gray", alpha=0.5, label="Chance")
    ax[3].set_title("Validation minDCF — Balanced Val (↓ Better)")
    ax[3].set_xlabel("Epoch"); ax[3].legend(); ax[3].grid(True, ls=":", alpha=0.6)

    gap = [abs(a-b) for a, b in zip(H["tr_loss"], H["vl_loss"])]
    ax[4].plot(E, gap, label="|Train−Val| loss", color="darkcyan")
    ax[4].set_title("Train–Val Loss Gap (Overfitting Indicator)")
    ax[4].set_xlabel("Epoch"); ax[4].legend(); ax[4].grid(True, ls=":", alpha=0.6)

    ax[5].axis("off")
    info = (
        "Configuration Summary\n"
        "─────────────────────\n"
        "Mode     : Cold-start, end-to-end\n"
        "Augment  : None (pure baseline)\n"
        f"meta_w   : {META_W}  aux_w : {AUX_W}\n"
        f"LR_base  : {LR_BASE}  LR_fuser : {LR_FUSER}\n"
        f"FocalLoss: γ=2.0, ls=0.10\n"
        f"Dropout  : 0.50 (fuser)\n"
        f"WD       : {WEIGHT_DECAY}\n"
        f"Warmup   : {WARMUP_EPOCHS} ep linear + cosine\n"
        f"Patience : {PATIENCE}\n"
        "Val set  : BALANCED subset of dev\n"
        f"Best EER : {best_eer:.4f}%"
    )
    ax[5].text(0.05, 0.95, info, transform=ax[5].transAxes, fontsize=10,
               va="top", family="monospace",
               bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    fig.suptitle("Baseline Cross-Attention Ensemble — Cold-Start\n"
                 "(Balanced Val Subset · Differential LR · No Augmentation)", fontsize=13)
    fig.tight_layout()
    gp = os.path.join(RESULTS_DIR, "crossattention_ensemble_baseline_metrics.png")
    fig.savefig(gp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plots → {gp}")


if __name__ == "__main__":
    main()
