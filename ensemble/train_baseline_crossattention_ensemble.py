
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = (os.path.abspath(os.path.join(CURRENT_DIR, ".."))
            if "ensemble" in CURRENT_DIR else CURRENT_DIR)
sys.path.append(ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR  = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

from src.data.dataset        import ASVspoofDataset
from src.models.aasist       import AASIST
from src.models.resnet_simam import resnet18_simam

# ── Paths ─────────────────────────────────────────────────────────────────────
PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PREPROCESSED_DEV_DIR   = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
PROTOCOL_DEV   = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"
OUTPUT_WEIGHTS = os.path.join(MODELS_DIR, "crossattention_ensemble_baseline_best.pth")

# ── Hyperparameters ────────────────────────────────────────────────────────────
TOTAL_EPOCHS  = 100
BATCH_SIZE    = 16
LR            = 1e-4
WEIGHT_DECAY  = 3e-4    # FIX C: increased from 1e-4
WARMUP_EPOCHS = 5
PATIENCE      = 25

# FIX A — Phased loss weights
# Phase 1: base models dominate → learn general representations first
# Phase 2: fusion head focuses → combines now-meaningful embeddings
PHASE1_END  = 20      # epoch boundary (1-indexed comparison: epoch < PHASE1_END)
PHASE1_META = 0.25    # CrossAttentionFuser loss weight in phase 1
PHASE1_AUX  = 0.75    # base-model loss weight in phase 1  (0.375 per model)
PHASE2_META = 0.65    # CrossAttentionFuser loss weight in phase 2
PHASE2_AUX  = 0.35    # base-model loss weight in phase 2  (0.175 per model)

# FIX B — SpecAugment parameters
FREQ_MASK_PARAM = 15  # max mel-frequency bins to mask (out of n_mels=80)
TIME_MASK_PARAM = 25  # max time frames to mask


def get_phase_weights(epoch: int):
    """Return (meta_w, aux_w) for the current epoch (0-indexed)."""
    if epoch < PHASE1_END:
        return PHASE1_META, PHASE1_AUX
    return PHASE2_META, PHASE2_AUX


# ══════════════════════════════════════════════════════════════════════════════
# Loss Function
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss with label smoothing.  FIX C: label_smoothing raised to 0.10.

    The increased smoothing prevents overconfident probability assignments to
    2019 LA-specific patterns. When the model assigns P(spoof)=0.99 to a 2019
    LA sample, it is simultaneously assigning P(spoof)≈0.01 to any unseen
    deepfake with similar surface features. Label smoothing forces P(spoof)
    to stay in [0.10, 0.90] range, improving calibration on out-of-domain data.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.10):
        super().__init__()
        self.gamma = gamma
        self.ls    = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.shape[1]
        with torch.no_grad():
            smooth = torch.zeros_like(logits).fill_(self.ls / (n_cls - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        log_p  = F.log_softmax(logits, dim=1)
        pt     = (log_p.exp() * smooth).sum(dim=1)
        weight = (1.0 - pt).pow(self.gamma)
        ce     = -(smooth * log_p).sum(dim=1)
        return (weight * ce).mean()


# ══════════════════════════════════════════════════════════════════════════════
# CrossAttentionFuser
# ══════════════════════════════════════════════════════════════════════════════

class CrossAttentionFuser(nn.Module):
    """
    Transformer-based fusion of AASIST (104-dim) and ResNet-SimAM (512-dim).

    Sequence: [CLS | proj_a(emb_a) | proj_r(emb_r)]  — length 3, d_model=256.
    CLS token aggregates cross-modal context via self-attention.
    Pre-LN (norm_first=True) for stable cold-start gradient flow.
    FIX C: dropout raised from 0.30 → 0.35.
    """

    def __init__(
        self,
        dim_a:       int   = 104,
        dim_r:       int   = 512,
        embed_dim:   int   = 256,
        num_heads:   int   = 8,
        num_classes: int   = 2,
        dropout:     float = 0.35,   # FIX C
    ):
        super().__init__()
        self.proj_a = nn.Sequential(
            nn.Linear(dim_a, embed_dim), nn.LayerNorm(embed_dim)
        )
        self.proj_r = nn.Sequential(
            nn.Linear(dim_r, embed_dim), nn.LayerNorm(embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, emb_a: torch.Tensor, emb_r: torch.Tensor) -> torch.Tensor:
        if emb_a.dtype != emb_r.dtype:
            emb_a = emb_a.to(emb_r.dtype)
        B       = emb_a.size(0)
        token_a = self.proj_a(emb_a).unsqueeze(1)
        token_r = self.proj_r(emb_r).unsqueeze(1)
        cls     = self.cls_token.expand(B, -1, -1)
        seq     = torch.cat((cls, token_a, token_r), dim=1)
        seq     = self.transformer(seq)
        return self.head(seq[:, 0, :])


# ══════════════════════════════════════════════════════════════════════════════
# End-to-End Ensemble Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class EndToEndEnsemble(nn.Module):
    """
    Wraps AASIST, ResNet-SimAM, and CrossAttentionFuser.

    Mixed-Precision Strategy
    ─────────────────────────
    AASIST → FP32 (SincNet sinc-function bank: division by near-zero values
              at low-frequency filter edges → NaN in FP16).
    ResNet + Fuser → FP16 inside torch.amp.autocast (stable CNN/transformer ops).

    The forward hook on aasist.fc and resnet.fc captures inp[0] — the
    pre-classifier embedding that encodes the model's learned representation
    without the final linear projection's training-distribution bias.
    """

    def __init__(self, aasist: nn.Module, resnet: nn.Module, fusion_head: nn.Module):
        super().__init__()
        self.aasist      = aasist
        self.resnet      = resnet
        self.fusion_head = fusion_head
        self._emb_a = [None]
        self._emb_r = [None]

        def _ha(m, i, o): self._emb_a[0] = i[0]
        def _hr(m, i, o): self._emb_r[0] = i[0]

        self._h_a = self.aasist.fc.register_forward_hook(_ha)
        self._h_r = self.resnet.fc.register_forward_hook(_hr)

    def forward(self, waveform: torch.Tensor, mel_db: torch.Tensor,
                return_base_outs: bool = False):
        out_a = self.aasist(waveform)                             # FP32
        with torch.amp.autocast("cuda"):
            out_r    = self.resnet(mel_db)                        # FP16
            out_meta = self.fusion_head(self._emb_a[0], self._emb_r[0])
        if return_base_outs:
            return out_meta, out_a, out_r
        return out_meta

    def remove_hooks(self):
        self._h_a.remove()
        self._h_r.remove()


# ══════════════════════════════════════════════════════════════════════════════
# Cold-Start Weight Initialisers
# ══════════════════════════════════════════════════════════════════════════════

def init_aasist_cold_start(model: nn.Module) -> None:
    """
    AASIST cold start:
    · Conv/Linear with ReLU activations:    Kaiming He (fan_out / fan_in)
    · GAT / attention layers:               Xavier Uniform (no ReLU gate)
    · BatchNorm:                            ones / zeros
    """
    attn_kw = ("attn", "gat", "attention", "query", "key", "value")
    for name, module in model.named_modules():
        is_attn = any(kw in name.lower() for kw in attn_kw)
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            if is_attn:
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            if is_attn:
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.kaiming_normal_(module.weight, mode="fan_in",
                                        nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    print("  AASIST: Kaiming (Conv/Linear) + Xavier (GAT/attn) cold start.")


def init_resnet_cold_start(model: nn.Module) -> None:
    """
    ResNet-SimAM cold start:
    · Conv2d:  Kaiming He (fan_out, relu)  — standard ResNet practice
    · BN:      ones / zeros
    · Linear:  small normal (std=0.01)  — prevents overconfident FC logits
               before convolutional features are meaningful
    """
    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                    nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    print("  ResNet: Kaiming (Conv) + small-normal (Linear) cold start.")


def init_fuser_cold_start(fuser: CrossAttentionFuser) -> None:
    """
    CrossAttentionFuser cold start:
    · proj_a / proj_r Linear:  Kaiming (fan_in, relu) — feeds into LayerNorm + transformer
    · head intermediate Linear: Kaiming (fan_in, relu)
    · head final output Linear: small normal (std=0.01) — no overconfident fusion decisions
    """
    for seq_mod in (fuser.proj_a, fuser.proj_r):
        for m in seq_mod.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    head_lins = [m for m in fuser.head.modules() if isinstance(m, nn.Linear)]
    for i, lin in enumerate(head_lins):
        if i < len(head_lins) - 1:
            nn.init.kaiming_normal_(lin.weight, mode="fan_in", nonlinearity="relu")
        else:
            nn.init.normal_(lin.weight, mean=0.0, std=0.01)
        if lin.bias is not None:
            nn.init.zeros_(lin.bias)
    print("  CrossAttentionFuser: Kaiming (proj) + small-normal (head) cold start.")


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def create_weighted_sampler(dataset: ASVspoofDataset) -> WeightedRandomSampler:
    labels        = dataset.labels
    class_counts  = torch.bincount(torch.tensor(labels))
    total_samples = len(labels)
    class_weights = total_samples / class_counts.float()
    sample_weights = [class_weights[lbl] for lbl in labels]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=total_samples, replacement=True
    )


def compute_eer(y_true, y_scores) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    return max(0.0, float(fpr[np.nanargmin(np.abs(fnr - fpr))]) * 100.0)


def compute_min_dcf(y_true, y_scores,
                    p_target: float = 0.05,
                    c_miss:   float = 1.0,
                    c_fa:     float = 1.0) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr         = 1.0 - tpr
    dcf         = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    default_dcf = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return float(np.min(dcf) / default_dcf)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Baseline Cold-Start Cross-Attention Ensemble v3 — device: {device}")
    print("=" * 70)

    # ── Models (cold start) ───────────────────────────────────────────────────
    aasist_model = AASIST(
        stft_window=698, stft_hop=398, freq_bins=116,
        gat_layers=2, heads=5, head_dim=104, hidden_dim=455, dropout=0.33,
    )
    resnet_model = resnet18_simam(num_classes=2, dropout_rate=0.22)
    fusion_head  = CrossAttentionFuser(
        dim_a=104, dim_r=512, embed_dim=256,
        num_heads=8, num_classes=2, dropout=0.35,
    )

    print("Initialising all components from scratch (cold start):")
    init_aasist_cold_start(aasist_model)
    init_resnet_cold_start(resnet_model)
    init_fuser_cold_start(fusion_head)

    wrapper_model = EndToEndEnsemble(aasist_model, resnet_model, fusion_head).to(device)
    n_params = sum(p.numel() for p in wrapper_model.parameters())
    print(f"  Total trainable parameters: {n_params:,}")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    val_ds   = ASVspoofDataset(PREPROCESSED_DEV_DIR,   PROTOCOL_DEV)
    sampler  = create_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=False)
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # ── Audio feature extractors ───────────────────────────────────────────────
    mel_transform = T.MelSpectrogram(
        sample_rate=16000, n_fft=512, hop_length=160, n_mels=80
    ).to(device)
    amp_to_db = T.AmplitudeToDB(stype="power", top_db=80).to(device)

    # FIX B — SpecAugment (training-only spectral regularisation)
    # iid_masks=True: each sample in the batch gets an independent random mask.
    # This is more effective than a shared mask because it prevents the model
    # from compensating by attending to the unmasked region of every sample.
    freq_masking = T.FrequencyMasking(
        freq_mask_param=FREQ_MASK_PARAM, iid_masks=True
    ).to(device)
    time_masking = T.TimeMasking(
        time_mask_param=TIME_MASK_PARAM, iid_masks=True
    ).to(device)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(
        wrapper_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS, eta_min=1e-7
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[WARMUP_EPOCHS],
    )
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.10)
    scaler    = torch.amp.GradScaler("cuda")

    # ── Training state ─────────────────────────────────────────────────────────
    best_eer          = float("inf")
    epochs_no_improve = 0
    start_time        = time.time()
    history = dict(
        train_loss=[], val_loss=[],
        loss_meta=[], loss_aux=[],
        train_eer=[], val_eer=[],
        val_auc=[], val_min_dcf=[],
        meta_w=[], aux_w=[],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Training Loop
    # ══════════════════════════════════════════════════════════════════════════
    for epoch in range(TOTAL_EPOCHS):
        meta_w, aux_w = get_phase_weights(epoch)
        phase_tag = "P1-BaseFirst" if epoch < PHASE1_END else "P2-FusionFocus"

        # ── Training ──────────────────────────────────────────────────────────
        wrapper_model.train()
        sum_total = sum_meta = sum_aux = 0.0
        train_labels = []
        train_probs  = []
        nan_batches  = 0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [{phase_tag}]")
        for waveforms, labels in pbar:
            waveforms = waveforms.squeeze(1).to(device)
            labels    = labels.to(device)

            # Mel spectrogram (transform is not a learned parameter)
            with torch.no_grad():
                mel    = mel_transform(waveforms)
                mel_db = amp_to_db(mel).unsqueeze(1)   # (B, 1, n_mels, T)

            # FIX B: SpecAugment — training only, independent per sample
            mel_db = freq_masking(mel_db)
            mel_db = time_masking(mel_db)

            optimizer.zero_grad(set_to_none=True)

            out_meta, out_a, out_r = wrapper_model(
                waveforms, mel_db, return_base_outs=True
            )

            # Cast all logits to FP32 before loss (mixed-precision safety)
            out_meta_f = out_meta.float()
            out_a_f    = out_a.float()
            out_r_f    = out_r.float()

            # FIX A: phased loss composition
            loss_meta  = criterion(out_meta_f, labels)
            loss_aux   = criterion(out_a_f, labels) + criterion(out_r_f, labels)
            loss_total = meta_w * loss_meta + aux_w * loss_aux

            if torch.isnan(loss_total) or torch.isinf(loss_total):
                nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                if nan_batches <= 3:
                    print(f"\n  [Warning] NaN/Inf at epoch {epoch+1}. Skipping batch.")
                continue

            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(wrapper_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            sum_total += loss_total.item()
            sum_meta  += loss_meta.item()
            sum_aux   += loss_aux.item()

            with torch.no_grad():
                train_labels.extend(labels.cpu().numpy())
                train_probs.extend(
                    torch.softmax(out_meta_f, dim=1)[:, 1].cpu().numpy()
                )

            pbar.set_postfix({
                "loss":   f"{loss_total.item():.4f}",
                "L_meta": f"{loss_meta.item():.4f}",
                "L_aux":  f"{loss_aux.item():.4f}",
                "mw":     f"{meta_w:.2f}",
            })

        n_b       = max(1, len(train_loader) - nan_batches)
        avg_train = sum_total / n_b
        avg_lmeta = sum_meta  / n_b
        avg_laux  = sum_aux   / n_b
        train_eer = compute_eer(train_labels, train_probs)

        # ── Validation ────────────────────────────────────────────────────────
        wrapper_model.eval()
        sum_val  = 0.0
        val_lbls = []
        val_prbs = []

        with torch.no_grad():
            for wv, lv in tqdm(val_loader,
                               desc=f"Epoch {epoch+1} [Valid]", leave=False):
                wv = wv.squeeze(1).to(device)
                lv = lv.to(device)
                mel    = mel_transform(wv)
                mel_db = amp_to_db(mel).unsqueeze(1)
                # No SpecAugment at validation — clean spectrogram
                out_v    = wrapper_model(wv, mel_db)
                sum_val += criterion(out_v.float(), lv).item()
                val_lbls.extend(lv.cpu().numpy())
                val_prbs.extend(
                    torch.softmax(out_v.float(), dim=1)[:, 1].cpu().numpy()
                )

        avg_val     = sum_val / len(val_loader)
        val_eer     = compute_eer(val_lbls, val_prbs)
        val_auc     = roc_auc_score(val_lbls, val_prbs)
        val_min_dcf = compute_min_dcf(val_lbls, val_prbs)

        scheduler.step()

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["loss_meta"].append(avg_lmeta)
        history["loss_aux"].append(avg_laux)
        history["train_eer"].append(train_eer)
        history["val_eer"].append(val_eer)
        history["val_auc"].append(val_auc)
        history["val_min_dcf"].append(val_min_dcf)
        history["meta_w"].append(meta_w)
        history["aux_w"].append(aux_w)

        elapsed  = time.time() - start_time
        eta_secs = int((elapsed / (epoch + 1)) * (TOTAL_EPOCHS - epoch - 1))
        eta_str  = str(datetime.timedelta(seconds=eta_secs))
        cur_lr   = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:3d} [{phase_tag}] | LR {cur_lr:.2e} | "
            f"Train {avg_train:.4f} | Val {avg_val:.4f} | "
            f"TrainEER {train_eer:.3f}% | ValEER {val_eer:.3f}%"
        )
        print(
            f"           L_meta {avg_lmeta:.4f} (w={meta_w:.2f}) | "
            f"L_aux {avg_laux:.4f} (w={aux_w:.2f}) | "
            f"AUC {val_auc:.4f} | minDCF {val_min_dcf:.4f} | ETA {eta_str}"
        )

        if val_eer < best_eer:
            best_eer          = val_eer
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": wrapper_model.state_dict(),
                    "val_eer": val_eer, "val_auc": val_auc,
                    "val_min_dcf": val_min_dcf,
                },
                OUTPUT_WEIGHTS,
            )
            print(f"  -> Val EER improved to {best_eer:.4f}%. Saved ✓")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement ({epochs_no_improve}/{PATIENCE})")

        if epochs_no_improve >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    wrapper_model.remove_hooks()
    total_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"\nTotal time: {total_str} | Best Val EER: {best_eer:.4f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # Diagnostic Plots
    # ══════════════════════════════════════════════════════════════════════════
    E   = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    axes = axes.flatten()

    # Loss trajectory
    axes[0].plot(E, history["train_loss"],  label="Train Loss (Total)", color="blue")
    axes[0].plot(E, history["val_loss"],    label="Val Loss",           color="red",        ls="--")
    axes[0].plot(E, history["loss_meta"],   label="Train L_meta",       color="steelblue",  ls=":")
    axes[0].plot(E, history["loss_aux"],    label="Train L_aux (bases)",color="darkorange", ls=":")
    if PHASE1_END < len(history["train_loss"]):
        axes[0].axvline(x=PHASE1_END, color="gray", ls="--", alpha=0.6,
                        label=f"Phase switch (ep {PHASE1_END})")
    axes[0].set_title("Loss Trajectory (Phased Weights)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8); axes[0].grid(True, ls=":", alpha=0.6)

    # EER
    axes[1].plot(E, history["train_eer"], label="Train EER %", color="teal")
    axes[1].plot(E, history["val_eer"],   label="Val EER %",   color="purple", ls="--")
    if PHASE1_END < len(history["train_eer"]):
        axes[1].axvline(x=PHASE1_END, color="gray", ls="--", alpha=0.6)
    axes[1].set_title("Equal Error Rate (↓ Better)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("EER (%)")
    axes[1].legend(); axes[1].grid(True, ls=":", alpha=0.6)

    # AUC
    axes[2].plot(E, history["val_auc"], label="ROC-AUC", color="darkgreen")
    axes[2].axhline(y=0.5, ls=":", color="gray", alpha=0.5, label="Random baseline")
    axes[2].set_title("Validation AUC (↑ Better)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("AUC")
    axes[2].set_ylim(0.4, 1.02)
    axes[2].legend(); axes[2].grid(True, ls=":", alpha=0.6)

    # minDCF
    axes[3].plot(E, history["val_min_dcf"], label="minDCF", color="darkred")
    axes[3].axhline(y=1.0, ls=":", color="gray", alpha=0.5, label="Chance baseline")
    axes[3].set_title("Validation minDCF (↓ Better)")
    axes[3].set_xlabel("Epoch"); axes[3].set_ylabel("Normalised minDCF")
    axes[3].legend(); axes[3].grid(True, ls=":", alpha=0.6)

    # Phase weights
    axes[4].plot(E, history["meta_w"], label="meta_w (fusion)", color="navy")
    axes[4].plot(E, history["aux_w"],  label="aux_w  (bases)",  color="orange")
    if PHASE1_END < len(history["meta_w"]):
        axes[4].axvline(x=PHASE1_END, color="gray", ls="--", alpha=0.6,
                        label=f"Phase switch (ep {PHASE1_END})")
    axes[4].set_title("Phased Loss Weights")
    axes[4].set_xlabel("Epoch"); axes[4].set_ylabel("Weight")
    axes[4].set_ylim(0.0, 1.0); axes[4].legend()
    axes[4].grid(True, ls=":", alpha=0.6)

    # Config summary
    axes[5].axis("off")
    info = (
        "Configuration Summary\n"
        "─────────────────────\n"
        "Mode: Cold-start, end-to-end\n"
        f"SpecAugment: freq={FREQ_MASK_PARAM}, time={TIME_MASK_PARAM}\n"
        f"Phase 1 (ep 1-{PHASE1_END}): meta={PHASE1_META}, aux={PHASE1_AUX}\n"
        f"Phase 2 (ep {PHASE1_END+1}+): meta={PHASE2_META}, aux={PHASE2_AUX}\n"
        f"FocalLoss: γ=2.0, ls=0.10\n"
        f"Dropout (fuser): 0.35\n"
        f"Weight decay: {WEIGHT_DECAY}\n"
        f"LR: {LR} w/ {WARMUP_EPOCHS}-ep warmup + cosine\n"
        f"Best Val EER: {best_eer:.4f}%"
    )
    axes[5].text(0.05, 0.95, info, transform=axes[5].transAxes,
                 fontsize=10, va="top", family="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(
        "Baseline Cross-Attention Ensemble — Cold-Start v3\n"
        "(Phased Training + SpecAugment + Increased Regularisation)",
        fontsize=13,
    )
    fig.tight_layout()
    gp = os.path.join(RESULTS_DIR, "crossattention_ensemble_baseline_metrics.png")
    fig.savefig(gp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plots saved to {gp}")


if __name__ == "__main__":
    main()