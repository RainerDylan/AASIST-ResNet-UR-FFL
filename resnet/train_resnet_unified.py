"""
train_resnet_unified.py — Unified UR-FFL Training for ResNet-SimAM.
Exactly mirrors the AASIST v-final codebase for 1:1 comparison.
"""

import sys, os, time, datetime, math
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(ROOT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR  = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torchaudio.transforms as T
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.data.dataset     import ASVspoofDataset
from src.models.resnet_simam import resnet18_simam
from src.ur_ffl.sensor    import UncertaintySensor
from src.ur_ffl.controller import PDController
from src.ur_ffl.selector  import DegradationSelector
from src.ur_ffl.actuator  import DegradationActuator

# ── config ────────────────────────────────────────────────────────────────────
PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PREPROCESSED_DEV_DIR   = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
PROTOCOL_DEV   = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"
OUTPUT_WEIGHTS = os.path.join(MODELS_DIR, "resnet_unified_best.pth")

# Hyperparameters (From Optuna Tuning)
TOTAL_EPOCHS   = 60
BATCH_SIZE     = 32
LR             = 0.0008678047296390247
WEIGHT_DECAY   = 0.00016671229135810644
DROPOUT_RATE   = 0.22489397436884667

# Loss weights 
CLEAN_WEIGHT   = 0.40    
DEG_WEIGHT     = 0.40    
CONS_WEIGHT    = 0.10    
ARC_WEIGHT     = 0.10    

# Checkpointing weights
CKPT_CLEAN_W   = 0.30
CKPT_AUG_W     = 0.70

EER_C_SOFT_WARN = 5.0    

SWA_START_FRAC = 0.80
SWA_LR         = 1e-6
PATIENCE       = 20
COLD_START_TOL = 0.05


# ── ArcFace head (Deng et al. CVPR 2019) ─────────────────────────────────────
class ArcFaceHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 2,
                 s: float = 32.0, m: float = 0.35):
        super().__init__()
        self.s       = s
        self.m       = m
        self.cos_m   = math.cos(m)
        self.sin_m   = math.sin(m)
        self.th      = math.cos(math.pi - m)
        self.mm      = math.sin(math.pi - m) * m   

        self.weight  = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor,
                labels: torch.Tensor | None = None) -> torch.Tensor:
        feat_norm   = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine      = F.linear(feat_norm, weight_norm)          
        cosine      = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None:
            return cosine * self.s

        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi  = cosine * self.cos_m - sine * self.sin_m
        phi  = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output  = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s


# ── Focal Loss (Lin et al. ICCV 2017) ────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.ls    = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.shape[1]
        with torch.no_grad():
            smooth = torch.zeros_like(logits).fill_(self.ls / (n_cls - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)

        log_p  = F.log_softmax(logits, dim=1)
        p      = log_p.exp()
        pt     = (p * smooth).sum(dim=1)
        weight = (1.0 - pt).pow(self.gamma)
        ce     = -(smooth * log_p).sum(dim=1)
        return (weight * ce).mean()


# ── Weight initialisation ─────────────────────────────────────────────────────
def init_weights_cold_start(model: nn.Module) -> None:
    for full_name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            # Ultra-tight variance prevents random logit drift from strictly positive ReLUs
            nn.init.normal_(module.weight, mean=0.0, std=1e-5) 
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if hasattr(module, 'weight') and module.weight is not None: nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None: nn.init.zeros_(module.bias)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Cold start: Kaiming Normal (Conv) + Normal 1e-5 (Linear). All {n_params:,} parameters re-initialised.")

def verify_cold_start(model: nn.Module, device: torch.device, n_samples: int = 256, tol: float = COLD_START_TOL) -> None:
    model.eval()
    was_training = model.training
    
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80).to(device)
    amp_to_db = T.AmplitudeToDB(stype='power', top_db=80).to(device)
    
    with torch.no_grad():
        dummy_wav = torch.randn(n_samples, 64600, device=device) 
        dummy_mel = mel_transform(dummy_wav)
        dummy_feat = amp_to_db(dummy_mel).unsqueeze(1)
        
        with torch.amp.autocast('cuda'):
            out = model(dummy_feat)
            
        probs = torch.softmax(out, dim=1)
        mean_spoof = probs[:, 0].mean().item()
        mean_bonafide = probs[:, 1].mean().item()

    print(f"  [ColdStart verify] Pre-training mean class probs: spoof={mean_spoof:.3f}, bonafide={mean_bonafide:.3f}")
    if abs(mean_spoof - 0.5) > tol or abs(mean_bonafide - 0.5) > tol:
        raise RuntimeError("Cold-start verification FAILED. Check for pre-trained weight leakage.")
    if was_training: model.train()


# ── Helpers ───────────────────────────────────────────────────────────────────
def create_weighted_sampler(dataset: ASVspoofDataset) -> WeightedRandomSampler:
    labels = dataset.labels
    cc     = torch.bincount(torch.tensor(labels))
    cw     = len(labels) / cc.float()
    return WeightedRandomSampler([cw[l] for l in labels], len(labels), replacement=True)

def compute_eer(y_true, y_scores) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr         = 1.0 - tpr
    return fpr[np.nanargmin(np.abs(fnr - fpr))] * 100.0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Unified UR-FFL Training (ResNet-SimAM) — device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = resnet18_simam(num_classes=2, dropout_rate=DROPOUT_RATE).to(device)

    init_weights_cold_start(model)
    verify_cold_start(model, device)

    # ── ArcFace head + feature hook ───────────────────────────────────────────
    # ResNet18 outputs a 512-dim embedding before the FC layer
    arc_head = ArcFaceHead(in_features=512, num_classes=2, s=32.0, m=0.35).to(device)
    print(f"  ArcFaceHead: in=512, classes=2, s=32.0, m=0.35 (Deng et al. CVPR 2019)")

    _arc_buf = [None]
    def _fc_hook(module, inp, out):
        _arc_buf[0] = inp[0]
    _hook_handle = model.fc.register_forward_hook(_fc_hook)

    # ── SWA ───────────────────────────────────────────────────────────────────
    swa_model  = AveragedModel(model)
    swa_start  = int(TOTAL_EPOCHS * SWA_START_FRAC)
    swa_active = False
    print(f"  SWA starts at epoch {swa_start + 1}/{TOTAL_EPOCHS}")

    # ── Data ───────────────────────────────────────────────────────────────────
    train_ds     = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    val_ds       = ASVspoofDataset(PREPROCESSED_DEV_DIR,   PROTOCOL_DEV)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=create_weighted_sampler(train_ds),
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Transforms (1D -> 2D) ──────────────────────────────────────────────────
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80).to(device)
    amp_to_db     = T.AmplitudeToDB(stype='power', top_db=80).to(device)

    # ── UR-FFL components ─────────────────────────────────────────────────────
    sensor     = UncertaintySensor(mc_passes=5)
    controller = PDController()
    selector   = DegradationSelector()
    actuator   = DegradationActuator(device)

    # ── Optimiser ──────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        list(model.parameters()) + list(arc_head.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler     = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=swa_start, eta_min=1e-7)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    criterion     = FocalLoss(gamma=2.0, label_smoothing=0.05)
    scaler        = torch.amp.GradScaler('cuda')

    # ── Training state ─────────────────────────────────────────────────────────
    best_composite    = float("inf")
    epochs_no_improve = 0

    history = dict(
        train_loss=[], val_loss=[], eer_clean=[], eer_aug=[],
        alpha=[], composite=[],
    )
    total_time = 0.0

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(TOTAL_EPOCHS):
        t0 = time.time()
        model.train()
        arc_head.train()

        train_loss = 0.0
        epoch_gaps = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [Train]")

        for waveforms, labels in pbar:
            waveforms = waveforms.squeeze(1).to(device)
            labels    = labels.to(device)

            # ── Step 1: measure uncertainty on clean batch ────────────────────
            with torch.no_grad():
                # Sensor requires 2D input for ResNet
                features_clean = amp_to_db(mel_transform(waveforms)).unsqueeze(1)
                z_u, _ = sensor.measure(model, features_clean)
                
            selections    = selector.select(z_u)
            alpha         = controller.alpha
            # Actuator applies augmentations to 1D waveform
            aug_waveforms = actuator.apply(waveforms, labels, selections, alpha)

            # ── Step 2: forward pass (clean + aug together) ───────────────────
            model.train()
            arc_head.train()

            optimizer.zero_grad()
            _arc_buf[0] = None

            combined      = torch.cat([waveforms, aug_waveforms], dim=0)
            combined_lbl  = torch.cat([labels,    labels],         dim=0)
            
            # Transform to 2D
            features_combined = amp_to_db(mel_transform(combined)).unsqueeze(1)

            with torch.amp.autocast('cuda'):
                out_combined  = model(features_combined)         
                feats_combined = _arc_buf[0]                     

                B          = waveforms.size(0)
                out_clean  = out_combined[:B]
                out_deg    = out_combined[B:]

                # ── Step 3: compute losses ────────────────────────────────────────
                if feats_combined is not None:
                    arc_out   = arc_head(feats_combined, combined_lbl)  
                    arc_clean = arc_out[:B]
                    arc_deg   = arc_out[B:]

                    loss_clean = criterion(arc_clean, labels)          
                    loss_deg   = criterion(arc_deg,   labels)          
                else:
                    loss_clean = criterion(out_clean, labels)
                    loss_deg   = criterion(out_deg,   labels)

                loss_cons = F.mse_loss(F.softmax(out_clean, dim=1),
                                       F.softmax(out_deg,   dim=1))

                loss_aux = criterion(out_clean, labels) * ARC_WEIGHT

                loss_total = (CLEAN_WEIGHT * loss_clean +
                              DEG_WEIGHT   * loss_deg   +
                              CONS_WEIGHT  * loss_cons  +
                              loss_aux)

            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(arc_head.parameters()),
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                acc_c = (out_clean.argmax(1) == labels).float().mean().item() * 100
                acc_a = (out_deg.argmax(1)   == labels).float().mean().item() * 100
            epoch_gaps.append(acc_c - acc_a)
            train_loss += loss_total.item()

            pbar.set_postfix({
                "loss": f"{loss_total.item():.4f}",
                "α":    f"{alpha:.3f}",
                "gap":  f"{acc_c - acc_a:.1f}pp",
            })

        mean_gap  = float(np.mean(epoch_gaps))
        new_alpha = controller.update(mean_gap)
        avg_train = train_loss / len(train_loader)

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            if not swa_active:
                swa_active = True
                print(f"  [SWA] Activated at epoch {epoch + 1}")
        else:
            scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        eval_model = swa_model if swa_active else model
        if swa_active:
            update_bn(train_loader, swa_model, device=device)

        eval_model.eval()
        arc_head.eval()
        val_loss = 0.0
        lc, pc, la, pa = [], [], [], []

        with torch.no_grad():
            for wv, lv in tqdm(val_loader, desc=f"Epoch {epoch+1} [Valid]", leave=False):
                wv = wv.squeeze(1).to(device)
                lv = lv.to(device)

                # Clean validation
                feat_c = amp_to_db(mel_transform(wv)).unsqueeze(1)
                with torch.amp.autocast('cuda'):
                    out_v = eval_model(feat_c)
                    val_loss += criterion(out_v, lv).item()
                lc.extend(lv.cpu().numpy())
                pc.extend(torch.softmax(out_v, dim=1)[:, 1].cpu().numpy())

                # Augmented validation
                aug_v = actuator._ssi(wv, alpha=max(0.3, controller.alpha))
                feat_a = amp_to_db(mel_transform(aug_v)).unsqueeze(1)
                with torch.amp.autocast('cuda'):
                    out_a = eval_model(feat_a)
                la.extend(lv.cpu().numpy())
                pa.extend(torch.softmax(out_a, dim=1)[:, 1].cpu().numpy())

        avg_val   = val_loss / len(val_loader)
        eer_clean = compute_eer(lc, pc)
        eer_aug   = compute_eer(la, pa)
        composite = CKPT_CLEAN_W * eer_clean + CKPT_AUG_W * eer_aug

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["eer_clean"].append(eer_clean)
        history["eer_aug"].append(eer_aug)
        history["alpha"].append(new_alpha)
        history["composite"].append(composite)

        total_time += time.time() - t0
        eta_s = int((total_time / (epoch + 1)) * (TOTAL_EPOCHS - epoch - 1))
        print(
            f"Epoch {epoch+1:3d} | LR {optimizer.param_groups[0]['lr']:.2e} | "
            f"α {new_alpha:.3f} | gap {mean_gap:.1f}pp | "
            f"Train {avg_train:.4f} | Val {avg_val:.4f} | "
            f"EER_c {eer_clean:.4f}% | EER_a {eer_aug:.4f}% | "
            f"Score {composite:.3f}% | SWA {'ON' if swa_active else 'off'} | "
            f"ETA {str(datetime.timedelta(seconds=eta_s))}"
        )

        if eer_clean > EER_C_SOFT_WARN:
            print(f"  [Info] EER_c={eer_clean:.2f}% > soft limit {EER_C_SOFT_WARN}% "
                  f"(cold-start expected; no action taken)")

        if composite < best_composite:
            best_composite    = composite
            epochs_no_improve = 0
            save_state = (swa_model.module.state_dict()
                          if swa_active else model.state_dict())
            torch.save(save_state, OUTPUT_WEIGHTS)
            print(f"  -> Best composite {best_composite:.3f}% "
                  f"(EER_c={eer_clean:.2f}%, EER_a={eer_aug:.2f}%) — saved ✓")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement ({epochs_no_improve}/{PATIENCE})")

        if epochs_no_improve >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    _hook_handle.remove()

    # ── Learning-curve plots ───────────────────────────────────────────────────
    E   = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    axes[0].plot(E, history["train_loss"], label="Train",      color="blue")
    axes[0].plot(E, history["val_loss"],   label="Val (clean)", color="red", ls="--")
    axes[0].set_title("Loss (Focal + ArcFace)")
    axes[0].legend(); axes[0].grid(True, ls=":", alpha=0.6)

    axes[1].plot(E, history["eer_clean"], label="EER clean %",  color="green")
    axes[1].plot(E, history["eer_aug"],   label="EER aug %",    color="purple", ls="--")
    axes[1].axhline(y=EER_C_SOFT_WARN, ls=":", color="orange", alpha=0.7,
                    label=f"EER_c soft warn={EER_C_SOFT_WARN}%")
    axes[1].set_title("EER (↓ better)")
    axes[1].legend(); axes[1].grid(True, ls=":", alpha=0.6)

    axes[2].plot(E, history["composite"], label="Composite score", color="navy")
    axes[2].set_title("Checkpoint Metric (↓ better)")
    axes[2].legend(); axes[2].grid(True, ls=":", alpha=0.6)

    axes[3].plot(E, history["alpha"], label="α (aug intensity)", color="orange")
    if hasattr(controller, 'setpoint'):
        axes[3].axhline(y=controller.setpoint / 10.0, ls=":", color="gray", alpha=0.6,
                        label=f"Setpoint ref={controller.setpoint:.0f}pp")
    axes[3].set_title("UR-FFL Alpha (PD Controller)")
    axes[3].legend(); axes[3].grid(True, ls=":", alpha=0.6)

    if swa_active:
        for ax in axes:
            ax.axvline(x=swa_start + 1, color="brown", ls="--", alpha=0.4,
                       label="SWA start")

    fig.tight_layout()
    gp = os.path.join(RESULTS_DIR, "resnet_unified_metrics.png")
    fig.savefig(gp, dpi=300)
    plt.close(fig)
    print(f"\nGraphs saved to {gp}")
    print(f"Best composite = {best_composite:.3f}%")
    print(f"Weights: {OUTPUT_WEIGHTS}")

if __name__ == "__main__":
    main()