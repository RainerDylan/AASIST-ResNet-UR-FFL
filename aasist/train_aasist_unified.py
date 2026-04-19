"""
train_aasist_unified.py — Unified UR-FFL Training (v-final, fixed).

Fixes applied to the original v-final
────────────────────────────────────────

FIX 1 — Broken checkpointing guard (Task 2)
  The original code had a hard `eer_clean < EER_C_MAX (3.0%)` guard
  inside the checkpoint condition AND incremented patience for EVERY epoch
  where eer_clean ≥ 3%, including epochs where the composite score DID
  improve.  With a cold-start model that requires 20+ epochs to reach
  sub-3% clean EER, the patience counter hit 20 before a single checkpoint
  was ever saved, resulting in `Best composite = inf%`.

  Fix: remove the guard entirely.  Checkpoint on composite score only.
  Early stopping tracks composite plateau, not clean-EER threshold.
  A soft informational warning is printed when EER_c exceeds a soft limit,
  but it does not affect saving or patience.

FIX 2 — PD controller mathematics (Task 3)
  See src/ur_ffl/controller.py for full derivation.
  error = fast_ema - setpoint (negative when gap < target ✓)
  alpha = alpha - delta (restores correct curriculum direction)
  setpoint = 5pp (calibrated from observed training-log data)

FIX 3 — Selector–sensor mismatch
  The updated sensor returns predictive entropy H ∈ [0, ln 2], but the
  previous selector used z-score thresholds (±0.5, ±1.5) which are always
  outside the entropy range, causing 'smear' and 'codec' (strongest aug)
  to be assigned zero times.  Fixed in src/ur_ffl/selector.py.

IMPROVEMENT — ArcFace margin loss (Task 4)
  Replaces the direct focal loss on softmax logits with an Additive Angular
  Margin Loss (ArcFace, Deng et al. CVPR 2019) computed on the pre-classifier
  feature vectors.

  Why ArcFace for deepfake detection:
  · ArcFace maximises inter-class angular separation and compacts intra-class
    variance in the feature hypersphere (Deng et al. 2019).
  · For binary bonafide/spoof separation, this sharpens the decision boundary
    in a way that generalises across unseen vocoders (cross-domain).
  · Wang et al. (2018) CosFace and subsequent ASVspoof literature confirm that
    margin-based losses improve EER on out-of-domain deepfake audio.

  Implementation: a forward hook on model.fc captures the (B, head_dim)
  pre-classifier features WITHOUT modifying the AASIST architecture.
  ArcFaceHead is a standalone nn.Module trained alongside the backbone.
  Scale s=32, margin m=0.35 — standard values from Deng et al. (2019).

References
──────────
Deng J. et al. (2019) CVPR: ArcFace — Additive Angular Margin Loss.
Wang H. et al. (2018) ECCV: CosFace — Large Margin Cosine Loss.
He K. et al. (2015) ICCV: Kaiming initialisation.
Glorot X. & Bengio Y. (2010): Xavier initialisation.
Rong Y. et al. (2020) ICLR: DropEdge — over-smoothing in GNNs.
Feng W. et al. (2020): DropNode / GRAND.
Tak H. et al. (2022) ICASSP: RawBoost augmentation.
Izmailov P. et al. (2018) UAI: Stochastic Weight Averaging.
Lin T-Y. et al. (2017) ICCV: Focal Loss.
Bengio Y. et al. (2009) ICML: Curriculum learning.
Ogata K. (2010): Modern Control Engineering.
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
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.data.dataset   import ASVspoofDataset
from src.models.aasist  import AASIST
from src.ur_ffl.sensor     import UncertaintySensor
from src.ur_ffl.controller import PDController
from src.ur_ffl.selector   import DegradationSelector
from src.ur_ffl.actuator   import DegradationActuator

# ── config ────────────────────────────────────────────────────────────────────
PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PREPROCESSED_DEV_DIR   = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
PROTOCOL_DEV   = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"
OUTPUT_WEIGHTS = os.path.join(MODELS_DIR, "aasist_unified_best.pth")

TOTAL_EPOCHS   = 60
BATCH_SIZE     = 32
LR             = 1e-4

# Loss weights — ARC replaces the direct logit focal loss on the backbone;
# consistency between clean and degraded outputs is preserved.
CLEAN_WEIGHT   = 0.40    # ArcFace loss on clean-batch features
DEG_WEIGHT     = 0.40    # ArcFace loss on degraded-batch features
CONS_WEIGHT    = 0.10    # π-model consistency (MSE between clean/aug softmax)
ARC_WEIGHT     = 0.10    # auxiliary focal loss on raw model logits (stability)

DROPNODE_P     = 0.10

# Checkpointing weights — composite = CKPT_CLEAN_W * EER_c + CKPT_AUG_W * EER_a
CKPT_CLEAN_W   = 0.30
CKPT_AUG_W     = 0.70

# Soft informational threshold for EER_c — does NOT gate checkpointing.
EER_C_SOFT_WARN = 5.0    # prints a warning if exceeded; does not prevent saving

SWA_START_FRAC = 0.80
SWA_LR         = 1e-6
PATIENCE       = 20


# ── ArcFace head (Deng et al. CVPR 2019) ─────────────────────────────────────

class ArcFaceHead(nn.Module):
    """
    Additive Angular Margin Loss head for binary deepfake classification.

    Takes L2-normalised feature vectors and computes the class logits with
    an angular margin m injected into the target-class angle, forcing greater
    inter-class separation on the feature hypersphere.

    Scale s=32 and margin m=0.35 are the standard values from Deng et al.
    (2019); m=0.35 is also used in CosFace (Wang et al. 2018) for binary tasks.

    Parameters
    ----------
    in_features  : int   — dimensionality of the input feature vector (head_dim).
    num_classes  : int   — number of output classes (2 for bonafide/spoof).
    s            : float — feature scale (radius of the hypersphere).
    m            : float — additive angular margin in radians.
    """

    def __init__(self, in_features: int, num_classes: int = 2,
                 s: float = 32.0, m: float = 0.35):
        super().__init__()
        self.s       = s
        self.m       = m
        self.cos_m   = math.cos(m)
        self.sin_m   = math.sin(m)
        # Boundary: cos(π − m) — below this, use the safe linearisation
        self.th      = math.cos(math.pi - m)
        self.mm      = math.sin(math.pi - m) * m   # linearisation offset

        # Class weight matrix (un-normalised; normalised in forward)
        self.weight  = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor,
                labels: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (B, in_features) — NOT required to be pre-normalised.
        labels   : (B,) long — required during training; None during inference.

        Returns
        -------
        (B, num_classes) scaled logits.
        """
        # L2-normalise both features and class weight vectors
        feat_norm   = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine      = F.linear(feat_norm, weight_norm)          # (B, C)
        cosine      = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None:
            # Inference: return standard scaled cosine logits (no margin)
            return cosine * self.s

        # Training: inject angular margin into the target class
        sine = torch.sqrt(1.0 - cosine.pow(2))
        # cos(θ + m) = cos(θ)·cos(m) − sin(θ)·sin(m)
        phi  = cosine * self.cos_m - sine * self.sin_m
        # Safe fallback when θ is close to π (Deng et al. 2019 Eq. 8)
        phi  = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Replace the target-class cosine with the margin-penalised value
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output  = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s


# ── Focal Loss (Lin et al. ICCV 2017) ────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    FL(p_t) = −(1 − p_t)^γ · log(p_t)
    γ=2 is optimal across tasks (Lin et al. 2017).
    label_smoothing=0.05 stabilises early training.
    """
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
    """
    Full cold-start weight initialisation (He et al. 2015; Glorot & Bengio 2010).
    Conv/Linear with ReLU activations → Kaiming He (fan_out, relu).
    Attention/GAT layers (no ReLU between Q/K/V) → Xavier uniform.
    BatchNorm → ones/zeros (standard).
    """
    attn_kw = ('attn', 'gat', 'attention', 'query', 'key', 'value')

    for name, module in model.named_modules():
        is_attn = any(kw in name.lower() for kw in attn_kw)

        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            (nn.init.xavier_uniform_ if is_attn else
             lambda w: nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu'))(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            (nn.init.xavier_uniform_ if is_attn else
             lambda w: nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu'))(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    for pname, param in model.named_parameters():
        if any(kw in pname.lower() for kw in attn_kw) and param.dim() >= 2:
            nn.init.xavier_uniform_(param)

    print("  Cold start: Kaiming init (Conv/Linear) + Xavier init (GAT/attn).")


# ── DropNode hooks (Feng et al. 2020; Rong et al. ICLR 2020) ─────────────────

class DropNodeHooks:
    """
    Randomly zeros entire node feature vectors during training forward passes
    on GAT layers.  Prevents over-smoothing and forces redundant path learning.
    Applied via forward hooks — no model modification required.
    """

    def __init__(self, p: float = 0.10):
        self.p       = p
        self._hooks  = []
        self._active = False

    def register(self, model: nn.Module) -> int:
        gat_kw = ('gat', 'graph', 'attn', 'attention')
        count  = 0
        for name, module in model.named_modules():
            if any(kw in name.lower() for kw in gat_kw) and name:
                h = module.register_forward_hook(self._hook_fn)
                self._hooks.append(h)
                count += 1
        self._active = True
        return count

    def _hook_fn(self, module, inp, output):
        if not (self._active and module.training):
            return output
        if isinstance(output, torch.Tensor) and output.dim() >= 2:
            mask = (torch.rand(output.shape[0], *([1] * (output.dim() - 1)),
                               device=output.device) > self.p).float()
            return output * mask
        return output

    def enable(self):  self._active = True
    def disable(self): self._active = False
    def remove(self):
        for h in self._hooks: h.remove()
        self._hooks.clear(); self._active = False


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
    print(f"Unified UR-FFL Training (fixed, cold start) — device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AASIST(
        stft_window=698, stft_hop=398, freq_bins=116,
        gat_layers=2, heads=5, head_dim=104,
        hidden_dim=455, dropout=0.3311465671378094,
    ).to(device)

    init_weights_cold_start(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  All {n_params:,} parameters reinitialised. Training from scratch.")

    # ── DropNode ───────────────────────────────────────────────────────────────
    dropnode = DropNodeHooks(p=DROPNODE_P)
    n_hooks  = dropnode.register(model)
    print(f"  DropNode (p={DROPNODE_P}) registered on {n_hooks} GAT-related modules.")

    # ── ArcFace head + feature hook ───────────────────────────────────────────
    # head_dim=104 must match AASIST config above.
    arc_head = ArcFaceHead(in_features=104, num_classes=2, s=32.0, m=0.35).to(device)
    print(f"  ArcFaceHead: in=104, classes=2, s=32.0, m=0.35 (Deng et al. CVPR 2019)")

    # Register forward hook on model.fc to capture pre-classifier features.
    # The hook fires on every model(x) call; _arc_buf[0] holds the last result.
    _arc_buf = [None]

    def _fc_hook(module, inp, out):
        # inp[0] is the (B, head_dim) tensor fed into nn.Linear — has gradients
        # when called outside torch.no_grad()
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

    # ── UR-FFL components ─────────────────────────────────────────────────────
    sensor     = UncertaintySensor(mc_passes=5)
    controller = PDController()
    selector   = DegradationSelector()
    actuator   = DegradationActuator(device)

    # ── Optimiser — includes arc_head parameters ───────────────────────────────
    optimizer = optim.AdamW(
        list(model.parameters()) + list(arc_head.parameters()),
        lr=LR, weight_decay=1e-4,
    )
    scheduler     = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=swa_start, eta_min=1e-7)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    criterion     = FocalLoss(gamma=2.0, label_smoothing=0.05)

    # ── Training state ─────────────────────────────────────────────────────────
    # FIX: checkpoint purely on composite — no hard EER_c guard.
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
        dropnode.enable()

        train_loss = 0.0
        epoch_gaps = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [Train]")

        for waveforms, labels in pbar:
            waveforms = waveforms.squeeze(1).to(device)
            labels    = labels.to(device)

            # ── Step 1: measure uncertainty on clean batch (for Selector) ─────
            with torch.no_grad():
                z_u, _ = sensor.measure(model, waveforms)   # entropy scores
            selections    = selector.select(z_u)
            alpha         = controller.alpha
            aug_waveforms = actuator.apply(waveforms, labels, selections, alpha)

            # ── Step 2: forward pass (clean + aug together) ───────────────────
            model.train()
            arc_head.train()
            dropnode.enable()

            optimizer.zero_grad()
            _arc_buf[0] = None   # reset hook buffer before training forward

            combined      = torch.cat([waveforms, aug_waveforms], dim=0)
            combined_lbl  = torch.cat([labels,    labels],         dim=0)
            out_combined  = model(combined)                  # triggers hook
            feats_combined = _arc_buf[0]                     # (2B, head_dim)

            B          = waveforms.size(0)
            out_clean  = out_combined[:B]
            out_deg    = out_combined[B:]

            # ── Step 3: compute losses ────────────────────────────────────────

            if feats_combined is not None:
                # ArcFace: adds angular margin in feature space (Deng et al. 2019)
                arc_out   = arc_head(feats_combined, combined_lbl)  # (2B, 2)
                arc_clean = arc_out[:B]
                arc_deg   = arc_out[B:]

                loss_clean = criterion(arc_clean, labels)          # ArcFace focal
                loss_deg   = criterion(arc_deg,   labels)          # ArcFace focal
            else:
                # Fallback: hook didn't fire (shouldn't happen; safety only)
                loss_clean = criterion(out_clean, labels)
                loss_deg   = criterion(out_deg,   labels)

            # π-model consistency: clean and aug should agree on class probabilities
            # (Laine & Aila 2016; stabilises training when augmentation is heavy)
            loss_cons = F.mse_loss(F.softmax(out_clean, dim=1),
                                   F.softmax(out_deg,   dim=1))

            # Auxiliary focal on raw model logits (numerical stability anchoring)
            loss_aux = criterion(out_clean, labels) * ARC_WEIGHT

            loss_total = (CLEAN_WEIGHT * loss_clean +
                          DEG_WEIGHT   * loss_deg   +
                          CONS_WEIGHT  * loss_cons  +
                          loss_aux)

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(arc_head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            # Per-batch accuracy gap for PD controller
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

        # ── PD controller update (once per epoch) ─────────────────────────────
        mean_gap  = float(np.mean(epoch_gaps))
        new_alpha = controller.update(mean_gap)
        avg_train = train_loss / len(train_loader)

        # ── LR schedule ───────────────────────────────────────────────────────
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            if not swa_active:
                swa_active = True
                print(f"  [SWA] Activated at epoch {epoch + 1}")
        else:
            scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        dropnode.disable()
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
                out_v = eval_model(wv)
                val_loss += criterion(out_v, lv).item()
                lc.extend(lv.cpu().numpy())
                pc.extend(torch.softmax(out_v, dim=1)[:, 1].cpu().numpy())

                # Augmented validation with SSI proxy noise
                aug_v = actuator._ssi(wv, alpha=max(0.3, controller.alpha))
                out_a = eval_model(aug_v)
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

        # ── Soft warning only (does NOT gate checkpointing) ────────────────────
        if eer_clean > EER_C_SOFT_WARN:
            print(f"  [Info] EER_c={eer_clean:.2f}% > soft limit {EER_C_SOFT_WARN}% "
                  f"(cold-start expected; no action taken)")

        # ── FIX: checkpoint on composite only — no hard guard ─────────────────
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
    dropnode.remove()

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
    axes[3].axhline(y=controller.setpoint / 10.0, ls=":", color="gray", alpha=0.6,
                    label=f"Setpoint ref={controller.setpoint:.0f}pp")
    axes[3].set_title("UR-FFL Alpha (PD Controller)")
    axes[3].legend(); axes[3].grid(True, ls=":", alpha=0.6)

    # Mark SWA start on all subplots
    if swa_active:
        for ax in axes:
            ax.axvline(x=swa_start + 1, color="brown", ls="--", alpha=0.4,
                       label="SWA start")

    fig.tight_layout()
    gp = os.path.join(RESULTS_DIR, "aasist_unified_metrics.png")
    fig.savefig(gp, dpi=300)
    plt.close(fig)
    print(f"\nGraphs saved to {gp}")
    print(f"Best composite = {best_composite:.3f}%")
    print(f"Weights: {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()