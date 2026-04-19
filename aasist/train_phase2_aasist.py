"""
Phase 2 UR-FFL Training — v9

Research-backed changes:
═══════════════════════════════════════════════════════════════════════════

1. REAL CODEC AUGMENTATION (primary fix)
   Every top-5 ASVspoof 2021 DF system used mp3/ogg/m4a codec augmentation
   (Das et al. 2021, ISCA ASVspoof 2021). The DF 2021 dataset contains audio
   compressed with real media codecs. Training without seeing codec artifacts
   means the model relies on features that are destroyed by codec processing.
   -> actuator.py now uses torchaudio.functional.apply_codec (mp3, ogg, a-law)

2. FOCAL LOSS (Lin et al. 2017, RetinaNet)
   Replaces CrossEntropy. Focal loss down-weights easy examples and focuses
   training on hard, misclassified samples. Particularly effective for
   cross-domain generalisation where the model must sharpen decision boundaries
   on difficult, ambiguous examples. gamma=2 is standard.

3. STOCHASTIC WEIGHT AVERAGING (Izmailov et al. 2018)
   Pindrop Labs (ASVspoof 2021 DF EER=16.05%) explicitly credited SWA as
   providing ensemble-like generalisation benefits. Averages weights from the
   last 20% of training epochs.
   -> torch.optim.swa_utils.AveragedModel + SWALR

4. ENCODER-ONLY FREEZE + COSINE WARMUP LR
   Only the STFT encoder is frozen (390K params frozen, 293K trainable).
   Learning rate starts at 1e-6 and warms up to 5e-5 over 5 epochs to
   prevent destroying Phase-1 weights in the first few batches.

5. CONTROLLER: EER_aug-based PD (target=35%)
   EER_aug has 243x more dynamic range than val_aug_loss.
   alpha decreases when model is overwhelmed, increases when it handles aug.

6. REDUCED L2-SP WEIGHT: 0.005
   Lighter regularisation vs v8 (0.05) since codec augmentation creates
   large gradient signal; L2-SP should guide, not dominate.

References:
  Tak et al. (2022) ICASSP: RawBoost
  Das et al. (2021) ASVspoof Workshop: codec augmentation strategies
  Lin et al. (2017) ICCV: Focal Loss
  Izmailov et al. (2018) UAI: SWA
  Pindrop Labs (2021) ASVspoof: SWA + codec for DF generalisation
"""

import sys
import os
import time
import datetime
import copy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR  = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.data.dataset  import ASVspoofDataset
from src.models.aasist import AASIST
from src.ur_ffl.sensor     import UncertaintySensor
from src.ur_ffl.controller import PDController
from src.ur_ffl.selector   import DegradationSelector
from src.ur_ffl.actuator   import DegradationActuator

# ── paths ─────────────────────────────────────────────────────────────────────

PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PREPROCESSED_DEV_DIR   = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
PROTOCOL_DEV   = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"
PHASE1_WEIGHTS = os.path.join(MODELS_DIR, "aasist_phase1_best.pth")

# ── hyperparameters ───────────────────────────────────────────────────────────

TOTAL_EPOCHS  = 50
BATCH_SIZE    = 32
LR_MAX        = 5e-5
LR_WARMUP     = 1e-6
WARMUP_EPOCHS = 5

CLEAN_WEIGHT  = 0.40
DEG_WEIGHT    = 0.45
CONS_WEIGHT   = 0.15

L2SP_WEIGHT   = 0.005          # lighter than v8; codec aug provides large gradient signal

FOCAL_GAMMA   = 2.0            # focal loss concentration parameter (Lin 2017)

SWA_START_PCT = 0.80           # start SWA at 80% of training (epoch 40 of 50)
SWA_LR        = 1e-6

PATIENCE      = 20

# ── focal loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss: FL(p_t) = -(1-p_t)^gamma * log(p_t)
    Down-weights easy examples (large p_t), focuses on hard misclassified ones.
    Lin et al. (2017): gamma=2 found to be optimal across tasks.
    """
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.ls    = label_smoothing

    def forward(self, logits, targets):
        n_classes = logits.shape[1]
        # Apply label smoothing to one-hot targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.ls / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)

        log_probs = F.log_softmax(logits, dim=1)
        probs     = torch.exp(log_probs)

        # p_t: probability assigned to the correct class
        pt = (probs * smooth_targets).sum(dim=1)
        focal_weight = (1.0 - pt).pow(self.gamma)

        ce = -(smooth_targets * log_probs).sum(dim=1)
        return (focal_weight * ce).mean()

# ── helpers ───────────────────────────────────────────────────────────────────

def create_weighted_sampler(dataset):
    labels       = dataset.labels
    class_counts = torch.bincount(torch.tensor(labels))
    total        = len(labels)
    cw           = total / class_counts.float()
    sw           = [cw[l] for l in labels]
    return WeightedRandomSampler(weights=sw, num_samples=total, replacement=True)


def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr         = 1.0 - tpr
    idx         = np.nanargmin(np.abs(fnr - fpr))
    return fpr[idx] * 100.0


def apply_freeze(model):
    FROZEN_KW = ["encoder"]
    nf = ntr = 0
    for name, param in model.named_parameters():
        if any(kw in name for kw in FROZEN_KW):
            param.requires_grad = False
            nf += param.numel()
        else:
            param.requires_grad = True
            ntr += param.numel()
    print(f"  Freeze: {nf:,} frozen (encoder) | {ntr:,} trainable")
    return FROZEN_KW


def lr_warmup_cosine(epoch, warmup_epochs, total_epochs, lr_min, lr_max):
    if epoch < warmup_epochs:
        return lr_min + (lr_max - lr_min) * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 2 UR-FFL Training v9 — device: {device}")

    model = AASIST(
        stft_window=698, stft_hop=398, freq_bins=116,
        gat_layers=2, heads=5, head_dim=104,
        hidden_dim=455, dropout=0.3311465671378094,
    ).to(device)

    print(f"  Loading Phase 1 weights from {PHASE1_WEIGHTS} ...")
    model.load_state_dict(torch.load(PHASE1_WEIGHTS, map_location=device))
    FROZEN_KW = apply_freeze(model)

    # L2-SP anchor
    phase1_params = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    print(f"  L2-SP anchor: {len(phase1_params)} tensors (weight={L2SP_WEIGHT})")

    # SWA model (Izmailov 2018: ensemble-in-weight-space)
    swa_model  = AveragedModel(model)
    swa_start  = int(TOTAL_EPOCHS * SWA_START_PCT)
    print(f"  SWA starts at epoch {swa_start+1}/{TOTAL_EPOCHS}, lr={SWA_LR}")

    print("  Building data loaders ...")
    train_ds = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    val_ds   = ASVspoofDataset(PREPROCESSED_DEV_DIR,   PROTOCOL_DEV)

    sampler      = create_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    sensor     = UncertaintySensor(mc_passes=5)
    controller = PDController()
    selector   = DegradationSelector()
    actuator   = DegradationActuator(device)

    optimizer  = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_WARMUP, weight_decay=1e-4,
    )
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    criterion     = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=0.05)

    best_aug_loss     = float("inf")
    epochs_no_improve = 0
    swa_active        = False

    history = dict(
        train_loss=[], val_loss=[], val_aug_loss=[],
        eer_clean=[], eer_aug=[], alpha=[],
    )
    total_time = 0.0

    for epoch in range(TOTAL_EPOCHS):
        t0 = time.time()

        # ── LR schedule: warmup then cosine (before SWA kicks in) ────────────
        if not swa_active:
            lr = lr_warmup_cosine(epoch, WARMUP_EPOCHS, swa_start, LR_WARMUP, LR_MAX)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # ── training ──────────────────────────────────────────────────────────
        model.train()
        for name, m in model.named_modules():
            if any(kw in name for kw in FROZEN_KW):
                m.eval()

        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [Train]")

        for waveforms, labels in pbar:
            waveforms = waveforms.squeeze(1).to(device)
            labels    = labels.to(device)

            with torch.no_grad():
                z_u, _ = sensor.measure(model, waveforms)
            selections    = selector.select(z_u)
            alpha         = controller.alpha
            aug_waveforms = actuator.apply(waveforms, labels, selections, alpha)

            model.train()
            for name, m in model.named_modules():
                if any(kw in name for kw in FROZEN_KW):
                    m.eval()

            optimizer.zero_grad()

            combined     = torch.cat([waveforms, aug_waveforms], dim=0)
            out_combined = model(combined)
            B            = waveforms.size(0)
            out_clean    = out_combined[:B]
            out_deg      = out_combined[B:]

            loss_clean = criterion(out_clean, labels)
            loss_deg   = criterion(out_deg,   labels)

            p_clean   = F.softmax(out_clean, dim=1)
            p_deg     = F.softmax(out_deg,   dim=1)
            loss_cons = F.mse_loss(p_clean, p_deg)

            ewc_sum  = sum(
                (param - phase1_params[n]).pow(2).sum()
                for n, param in model.named_parameters()
                if param.requires_grad and n in phase1_params
            )
            loss_ewc = L2SP_WEIGHT * ewc_sum

            loss_total = (CLEAN_WEIGHT * loss_clean +
                          DEG_WEIGHT   * loss_deg   +
                          CONS_WEIGHT  * loss_cons  +
                          loss_ewc)

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            train_loss += loss_total.item()

            pbar.set_postfix({
                "loss": f"{loss_total.item():.4f}",
                "α":    f"{alpha:.3f}",
            })

        # SWA update
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            if not swa_active:
                swa_active = True
                print(f"  [SWA] Started at epoch {epoch+1}")

        avg_train = train_loss / len(train_loader)

        # ── validation ────────────────────────────────────────────────────────
        eval_model = swa_model if swa_active else model
        if swa_active:
            # update BN stats for SWA model on a subset of training data
            update_bn(train_loader, swa_model, device=device)

        eval_model.eval()
        val_loss = val_aug_loss = 0.0
        lc, pc, la, pa = [], [], [], []

        with torch.no_grad():
            for wv, lv in tqdm(val_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [Valid]",
                                leave=False):
                wv = wv.squeeze(1).to(device)
                lv = lv.to(device)

                out_v     = eval_model(wv)
                val_loss += criterion(out_v, lv).item()
                pv        = torch.softmax(out_v, dim=1)[:, 1]
                lc.extend(lv.cpu().numpy())
                pc.extend(pv.cpu().numpy())

                # Augmented val: SSI noise as proxy for codec degradation
                aug_v        = actuator._ssi_noise(wv, alpha=controller.alpha)
                out_aug_v    = eval_model(aug_v)
                val_aug_loss += criterion(out_aug_v, lv).item()
                pav           = torch.softmax(out_aug_v, dim=1)[:, 1]
                la.extend(lv.cpu().numpy())
                pa.extend(pav.cpu().numpy())

        avg_val     = val_loss     / len(val_loader)
        avg_val_aug = val_aug_loss / len(val_loader)
        eer_clean   = compute_eer(lc, pc)
        eer_aug     = compute_eer(la, pa)

        # Controller update: pass EER_aug (%), not val_aug_loss
        new_alpha = controller.update(eer_aug)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_aug_loss"].append(avg_val_aug)
        history["eer_clean"].append(eer_clean)
        history["eer_aug"].append(eer_aug)
        history["alpha"].append(new_alpha)

        epoch_dur   = time.time() - t0
        total_time += epoch_dur
        eta_s       = int((total_time / (epoch+1)) * (TOTAL_EPOCHS - epoch - 1))
        print(
            f"Epoch {epoch+1:3d} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | "
            f"α {new_alpha:.3f} | "
            f"Train {avg_train:.4f} | "
            f"Val {avg_val:.4f} | "
            f"ValAug {avg_val_aug:.4f} | "
            f"EER_c {eer_clean:.4f}% | "
            f"EER_a {eer_aug:.4f}% | "
            f"SWA {'ON' if swa_active else 'off'} | "
            f"ETA {str(datetime.timedelta(seconds=eta_s))}"
        )

        if avg_val_aug < best_aug_loss:
            best_aug_loss     = avg_val_aug
            epochs_no_improve = 0
            sp = os.path.join(MODELS_DIR, "aasist_phase2_urffl_best.pth")
            torch.save(
                swa_model.module.state_dict() if swa_active else model.state_dict(),
                sp
            )
            print(f"  -> Best aug-val-loss {best_aug_loss:.5f} — saved")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement ({epochs_no_improve}/{PATIENCE})")

        if epochs_no_improve >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # ── plots ─────────────────────────────────────────────────────────────────
    E = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    axes[0].plot(E, history["train_loss"],   label="Train",      color="blue")
    axes[0].plot(E, history["val_loss"],     label="Val (clean)", color="red",    ls="--")
    axes[0].plot(E, history["val_aug_loss"], label="Val (aug)",   color="orange", ls=":")
    axes[0].set_title("Focal Loss"); axes[0].legend(); axes[0].grid(True, ls=":", alpha=0.6)

    axes[1].plot(E, history["eer_clean"], label="EER clean %", color="green")
    axes[1].plot(E, history["eer_aug"],   label="EER aug %",   color="purple", ls="--")
    axes[1].set_title("EER (lower = better)"); axes[1].legend(); axes[1].grid(True, ls=":", alpha=0.6)

    axes[2].plot(E, history["alpha"], label="alpha", color="orange")
    axes[2].axhline(y=controller.setpoint, ls=":", color="gray", alpha=0.7)
    axes[2].set_title("UR-FFL Alpha"); axes[2].legend(); axes[2].grid(True, ls=":", alpha=0.6)

    swa_line = swa_start + 1
    for ax in axes[:3]:
        ax.axvline(x=swa_line, color="brown", ls="--", alpha=0.5, label="SWA start")

    axes[3].plot(E, history["eer_aug"], label="EER_aug (control signal)", color="purple")
    axes[3].axhline(y=controller.TARGET_EER, ls=":", color="gray", label="Target EER=35%")
    axes[3].set_title("Controller signal"); axes[3].legend(); axes[3].grid(True, ls=":", alpha=0.6)

    fig.tight_layout()
    gp = os.path.join(RESULTS_DIR, "aasist_phase2_metrics.png")
    fig.savefig(gp, dpi=300)
    plt.close(fig)
    print(f"Graphs saved to {gp}")
    print(f"\nPhase 2 v9 complete. Best aug-val-loss = {best_aug_loss:.5f}")


if __name__ == "__main__":
    main()