"""
train_urffl_crossattention_ensemble.py
Cold-Start End-to-End UR-FFL Cross-Attention Ensemble — v3

═══════════════════════════════════════════════════════════════════════════════
RELATIONSHIP TO BASELINE v3
═══════════════════════════════════════════════════════════════════════════════
This file is mathematically identical to train_baseline_crossattention_ensemble.py
in every shared component:
  · Architecture:   identical AASIST / ResNet-SimAM / CrossAttentionFuser
  · Initialisers:   identical cold-start (Kaiming / Xavier / small-normal)
  · Loss function:  identical FocalLoss (γ=2.0, ls=0.10)
  · AMP strategy:   identical (AASIST FP32, ResNet+Fuser FP16)
  · Phased weights: identical PHASE1_END=20 boundary
  · SpecAugment:    identical parameters (freq=15, time=25), applied to
                    the clean spectrogram branch only (see Note A below)
  · Regularisation: identical weight_decay=3e-4, dropout=0.35
  · Scheduler:      identical 5-ep warmup + cosine annealing
  · Patience/epochs: identical PATIENCE=25, TOTAL_EPOCHS=100

UR-FFL additions layered on top of the baseline:
  · UncertaintySensor / DegradationSelector / DegradationActuator
  · Revised inlined PDController (fixes PD cold-start trap — see Fix 3)
  · Phased PD activation (frozen in Phase 1, active in Phase 2 — Fix 3)
  · UR-FFL training objective with phased weights (Fix 4)
  · π-model consistency loss (Laine & Aila 2017)
  · Composite checkpoint metric (0.30·EER_c + 0.70·EER_a)
  · Augmented validation pass for EER_aug measurement

═══════════════════════════════════════════════════════════════════════════════
FIXES APPLIED TO BRING UR-FFL IN LINE WITH BASELINE v3
═══════════════════════════════════════════════════════════════════════════════

FIX 1 — label_smoothing 0.05 → 0.10  (was line 150 in uploaded file)
  Baseline v3 uses 0.10 for better calibration on out-of-domain data.
  The uploaded UR-FFL kept the old 0.05 value.

FIX 2 — weight_decay 1e-4 → 3e-4  (was line 436, not even a named constant)
  The uploaded UR-FFL had weight_decay=1e-4 hard-coded in AdamW.
  Baseline v3 uses WEIGHT_DECAY=3e-4. Added named constant for clarity.

FIX 3 — CrossAttentionFuser dropout 0.30 → 0.35
  Uploaded file instantiated fuser with dropout=0.30.
  Baseline v3 uses 0.35. Unified.

FIX 4 — SpecAugment missing from UR-FFL
  The uploaded UR-FFL applied no spectral augmentation.
  Note A: because the combined batch is [clean | aug_waveforms], SpecAugment
  is applied to mel_db[:B] (clean branch) only. The aug branch already
  carries waveform-level degradation from the UR-FFL actuator (pitch shift,
  codec noise, temporal smearing). Adding SpecAugment on top of actuator
  degradation would create double-augmentation on the aug branch.
  Clean branch gets identical SpecAugment to the baseline.

FIX 5 — No phased loss weights in UR-FFL
  The uploaded UR-FFL used fixed CLEAN_W=0.35, DEG_W=0.35, AUX_W=0.20,
  CONS_W=0.10 throughout training. This causes the same failure mode as
  the unphased baseline: fusion head takes over before base models have
  built meaningful representations.

  Phased UR-FFL loss weights — proportionally matched to baseline:
    Phase 1 (ep 1-20):  P1_CLEAN_W=0.10, P1_DEG_W=0.10,
                        P1_AUX_W=0.70,   P1_CONS_W=0.10
      → "meta" total = 0.20, "aux" = 0.70 (0.35 per base model)
      → mirrors baseline Phase 1: meta=0.25, aux=0.75

    Phase 2 (ep 21+):   P2_CLEAN_W=0.30, P2_DEG_W=0.30,
                        P2_AUX_W=0.30,   P2_CONS_W=0.10
      → "meta" total = 0.60, "aux" = 0.30 (0.15 per base model)
      → mirrors baseline Phase 2: meta=0.65, aux=0.35

  CONS_W is kept constant across phases: consistency between clean and
  aug predictions is always a relevant objective regardless of phase.

FIX 6 — PD controller cold-start trap (α stuck at alpha_min=0.05)
  Root cause: During cold-start, both clean and aug confidence ≈ 50%,
  so gap ≈ 0pp at every early epoch.
    error = setpoint - gap = 10 - 0 = +10
    delta = Kp·10 + Kd·10 = 0.15 + 0.05 = 0.20
    new_alpha = 0.10 - 0.20 = -0.10 → clamped to 0.05 (alpha_min)
  Once clamped, d_error ≈ 0, so delta remains positive and alpha stays
  at alpha_min indefinitely. UR-FFL augmentation is effectively disabled
  for the entire training run.

  Fix: Freeze the PD controller during Phase 1. A fixed PHASE1_ALPHA=0.20
  provides mild, consistent augmentation while base models develop their
  representations. This also exposes AASIST and ResNet to degraded inputs
  early in training, improving their robustness before the fusion phase.

  At the Phase 1→2 boundary (epoch PHASE1_END):
    · controller.alpha is reset to PHASE1_ALPHA (=0.20)
    · controller._prev_error is reset to 0.0
    · PD updates resume with a meaningful gap signal (base models now
      produce discriminative embeddings, so gap is informative)

References
──────────
He K. et al. (2015) ICCV: Kaiming init.
Lin T-Y. et al. (2017) ICCV: Focal Loss.
Laine S. & Aila T. (2017) ICLR: π-model consistency.
Park D.S. et al. (2019) Interspeech: SpecAugment.
Vaswani A. et al. (2017) NeurIPS: Transformer.
Devlin J. et al. (2019) NAACL: BERT CLS-token.
Ogata K. (2010): Modern Control Engineering (PD anti-windup).
"""

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
from src.ur_ffl.sensor       import UncertaintySensor
from src.ur_ffl.selector     import DegradationSelector
from src.ur_ffl.actuator     import DegradationActuator

# ── Paths ─────────────────────────────────────────────────────────────────────
PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PREPROCESSED_DEV_DIR   = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
PROTOCOL_DEV   = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"
OUTPUT_WEIGHTS = os.path.join(MODELS_DIR, "crossattention_ensemble_urffl_best.pth")

# ── Shared hyperparameters (identical to baseline v3) ─────────────────────────
TOTAL_EPOCHS    = 100
BATCH_SIZE      = 16
LR              = 1e-4
WEIGHT_DECAY    = 3e-4    # FIX 2: was 1e-4 hard-coded in AdamW
WARMUP_EPOCHS   = 5
PATIENCE        = 25

# ── SpecAugment config (identical to baseline v3) ─────────────────────────────
FREQ_MASK_PARAM = 15
TIME_MASK_PARAM = 25

# ── Phased training boundary (identical to baseline v3) ───────────────────────
PHASE1_END   = 20         # epochs 0..(PHASE1_END-1) are Phase 1

# ── UR-FFL: phased loss weights (FIX 5) ───────────────────────────────────────
# Phase 1 — base models first (AUX dominates):
P1_CLEAN_W = 0.10   # fusion loss on clean batch
P1_DEG_W   = 0.10   # fusion loss on aug batch
P1_AUX_W   = 0.70   # base-model auxiliary loss  (0.35 per base model)
P1_CONS_W  = 0.10   # π-model consistency
# Phase 2 — fusion focus (meta dominates):
P2_CLEAN_W = 0.30
P2_DEG_W   = 0.30
P2_AUX_W   = 0.30   # base-model auxiliary loss  (0.15 per base model)
P2_CONS_W  = 0.10

# ── UR-FFL: PD controller phase behaviour (FIX 6) ─────────────────────────────
PHASE1_ALPHA = 0.20   # fixed augmentation intensity during Phase 1

# ── Composite checkpoint metric weights ───────────────────────────────────────
CKPT_CLEAN_W = 0.30
CKPT_AUG_W   = 0.70


def get_phase_weights(epoch: int):
    """Return (clean_w, deg_w, aux_w, cons_w) for the current epoch."""
    if epoch < PHASE1_END:
        return P1_CLEAN_W, P1_DEG_W, P1_AUX_W, P1_CONS_W
    return P2_CLEAN_W, P2_DEG_W, P2_AUX_W, P2_CONS_W


# ══════════════════════════════════════════════════════════════════════════════
# Revised PD Controller (inlined — FIX 6)
# ══════════════════════════════════════════════════════════════════════════════

class PDController:
    """
    Proportional-Derivative controller for UR-FFL augmentation intensity α.

    Control Law (discrete, anti-windup)
    ─────────────────────────────────────
        error[k]   = setpoint − gap[k]
        d_error[k] = error[k] − error[k−1]
        delta      = Kp·error[k] + Kd·d_error[k]
        α[k]       = clamp(α[k−1] − delta, α_min, α_max)

    Sign Convention
    ───────────────
        gap > setpoint → error < 0 → delta < 0 → α increases
            (clean model more confident than aug → harder augmentation)
        gap < setpoint → error > 0 → delta > 0 → α decreases
            (augmentation already challenging → ease off)

    Phase Behaviour (FIX 6)
    ───────────────────────
    During Phase 1 (epochs 0..PHASE1_END-1), the PD controller is NOT
    called. Instead, alpha is held at PHASE1_ALPHA=0.20. At the phase
    boundary, alpha and _prev_error are reset so Phase 2 starts cleanly.

    Anti-Windup
    ───────────
    α is clamped at every step. No external clamping needed.
    """

    def __init__(
        self,
        setpoint:   float = 10.0,
        Kp:         float = 0.015,
        Kd:         float = 0.005,
        alpha_min:  float = 0.10,
        alpha_max:  float = 0.90,
        alpha_init: float = 0.20,
    ):
        self.setpoint    = setpoint
        self.Kp          = Kp
        self.Kd          = Kd
        self.alpha_min   = alpha_min
        self.alpha_max   = alpha_max
        self.alpha       = alpha_init
        self._prev_error = 0.0

    def reset(self, alpha: float) -> None:
        """Reset controller state at the Phase 1→2 boundary."""
        self.alpha       = alpha
        self._prev_error = 0.0

    def update(self, gap: float) -> float:
        """
        Update α given the current epoch-mean accuracy gap (pp).
        Call only during Phase 2.
        """
        error   = self.setpoint - gap
        d_error = error - self._prev_error
        delta   = self.Kp * error + self.Kd * d_error
        new_alpha = self.alpha - delta
        new_alpha = float(np.clip(new_alpha, self.alpha_min, self.alpha_max))
        self._prev_error = error
        self.alpha       = new_alpha
        return new_alpha


# ══════════════════════════════════════════════════════════════════════════════
# Loss Function (FIX 1: label_smoothing=0.10, identical to baseline v3)
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. ICCV 2017) with label smoothing.
    FIX 1: label_smoothing raised from 0.05 to 0.10 to match baseline v3.
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
# CrossAttentionFuser (FIX 3: dropout=0.35, identical to baseline v3)
# ══════════════════════════════════════════════════════════════════════════════

class CrossAttentionFuser(nn.Module):
    """
    Transformer-based fusion of AASIST (104-dim) and ResNet-SimAM (512-dim).
    Sequence: [CLS | proj_a(emb_a) | proj_r(emb_r)] — length 3, d_model=256.
    Pre-LN (norm_first=True) for stable cold-start gradient flow.
    FIX 3: dropout raised from 0.30 to 0.35 to match baseline v3.
    Mathematically identical to baseline v3 CrossAttentionFuser.
    """

    def __init__(
        self,
        dim_a:       int   = 104,
        dim_r:       int   = 512,
        embed_dim:   int   = 256,
        num_heads:   int   = 8,
        num_classes: int   = 2,
        dropout:     float = 0.35,   # FIX 3
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
# End-to-End Ensemble Wrapper (identical to baseline v3)
# ══════════════════════════════════════════════════════════════════════════════

class EndToEndEnsemble(nn.Module):
    """
    AASIST (FP32) + ResNet-SimAM (FP16) + CrossAttentionFuser (FP16).
    AASIST in FP32: SincNet sinc-function NaN safety.
    Hook captures inp[0] of each model's final FC — pre-classifier embedding.
    Identical to baseline v3 wrapper.
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
        out_a = self.aasist(waveform)                              # FP32
        with torch.amp.autocast("cuda"):
            out_r    = self.resnet(mel_db)                         # FP16
            out_meta = self.fusion_head(self._emb_a[0], self._emb_r[0])
        if return_base_outs:
            return out_meta, out_a, out_r
        return out_meta

    def remove_hooks(self):
        self._h_a.remove()
        self._h_r.remove()


# ══════════════════════════════════════════════════════════════════════════════
# Cold-Start Initialisers (identical to baseline v3)
# ══════════════════════════════════════════════════════════════════════════════

def init_aasist_cold_start(model: nn.Module) -> None:
    """Kaiming (Conv/Linear with relu) + Xavier (GAT/attention)."""
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
    """Kaiming (Conv2d) + small-normal (Linear FC)."""
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
    """Kaiming (proj linear layers) + small-normal (output head)."""
    for seq_mod in (fuser.proj_a, fuser.proj_r):
        for m in seq_mod.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    head_lins = [m for m in fuser.head.modules() if isinstance(m, nn.Linear)]
    for i, lin in enumerate(head_lins):
        if i < len(head_lins) - 1:
            nn.init.kaiming_normal_(lin.weight, mode="fan_in",
                                    nonlinearity="relu")
        else:
            nn.init.normal_(lin.weight, mean=0.0, std=0.01)
        if lin.bias is not None:
            nn.init.zeros_(lin.bias)
    print("  CrossAttentionFuser: Kaiming (proj) + small-normal (head) cold start.")


# ══════════════════════════════════════════════════════════════════════════════
# Utilities (identical to baseline v3)
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
    return float(np.min(dcf) / min(c_miss * p_target, c_fa * (1.0 - p_target)))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"UR-FFL Cold-Start Cross-Attention Ensemble v3 — device: {device}")
    print("=" * 70)

    # ── Models (cold start, identical to baseline v3) ─────────────────────────
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

    # ── Audio feature extractors (identical to baseline v3) ───────────────────
    mel_transform = T.MelSpectrogram(
        sample_rate=16000, n_fft=512, hop_length=160, n_mels=80
    ).to(device)
    amp_to_db = T.AmplitudeToDB(stype="power", top_db=80).to(device)

    # FIX 4 — SpecAugment (identical params to baseline v3, clean branch only)
    # Applied to mel_db[:B] (clean spectrogram) only. The aug branch [B:]
    # already carries waveform-level degradation from the UR-FFL actuator.
    freq_masking = T.FrequencyMasking(
        freq_mask_param=FREQ_MASK_PARAM, iid_masks=True
    ).to(device)
    time_masking = T.TimeMasking(
        time_mask_param=TIME_MASK_PARAM, iid_masks=True
    ).to(device)

    # ── UR-FFL components ─────────────────────────────────────────────────────
    sensor     = UncertaintySensor(mc_passes=5)
    controller = PDController(
        setpoint=10.0, Kp=0.015, Kd=0.005,
        alpha_min=0.10, alpha_max=0.90, alpha_init=PHASE1_ALPHA,
    )
    selector = DegradationSelector()
    actuator = DegradationActuator(device)

    print(
        f"  UR-FFL: setpoint={controller.setpoint}pp | "
        f"Kp={controller.Kp} | Kd={controller.Kd} | "
        f"α ∈ [{controller.alpha_min}, {controller.alpha_max}] | "
        f"Phase 1 α fixed={PHASE1_ALPHA}"
    )

    # ── Optimiser + scheduler (FIX 2: weight_decay=3e-4, identical to baseline)
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
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.10)   # FIX 1
    scaler    = torch.amp.GradScaler("cuda")

    # ── Training state ─────────────────────────────────────────────────────────
    best_composite    = float("inf")
    epochs_no_improve = 0
    start_time        = time.time()
    pd_activated      = False   # tracks whether Phase 2 has started

    history = dict(
        train_loss=[], val_loss=[],
        loss_clean=[], loss_deg=[], loss_aux=[], loss_cons=[],
        train_eer=[], val_eer_clean=[], val_eer_aug=[],
        val_auc=[], val_min_dcf=[], composite=[],
        alpha=[], clean_w=[], deg_w=[], aux_w=[],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Training Loop
    # ══════════════════════════════════════════════════════════════════════════
    for epoch in range(TOTAL_EPOCHS):
        clean_w, deg_w, aux_w, cons_w = get_phase_weights(epoch)
        in_phase1  = epoch < PHASE1_END
        phase_tag  = "P1-BaseFirst" if in_phase1 else "P2-FusionFocus"

        # ── Phase 1→2 transition (FIX 6) ──────────────────────────────────────
        if not in_phase1 and not pd_activated:
            controller.reset(PHASE1_ALPHA)
            pd_activated = True
            print(f"\n  [Phase 2] PD controller activated at epoch {epoch+1}. "
                  f"α reset to {PHASE1_ALPHA:.2f}.")

        # Current alpha: fixed in Phase 1, PD-controlled in Phase 2
        current_alpha = PHASE1_ALPHA if in_phase1 else controller.alpha

        # ── Training phase ────────────────────────────────────────────────────
        wrapper_model.train()
        sum_total = sum_clean = sum_deg = sum_aux = sum_cons = 0.0
        train_labels = []
        train_probs  = []
        epoch_gaps   = []
        nan_batches  = 0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [{phase_tag}]")
        for waveforms, labels in pbar:
            waveforms = waveforms.squeeze(1).to(device)
            labels    = labels.to(device)

            # ── UR-FFL: measure uncertainty on clean batch ─────────────────────
            with torch.no_grad():
                z_u, _ = sensor.measure(wrapper_model.aasist, waveforms)

            # ── UR-FFL: select degradation and augment ─────────────────────────
            selections    = selector.select(z_u)
            alpha         = current_alpha
            aug_waveforms = actuator.apply(waveforms, labels, selections, alpha)

            # ── Mel spectrograms for combined batch ────────────────────────────
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                # Compute mel for clean and aug separately
                mel_clean_raw = amp_to_db(mel_transform(waveforms)).unsqueeze(1)
                mel_aug_raw   = amp_to_db(mel_transform(aug_waveforms)).unsqueeze(1)

            # FIX 4: SpecAugment on clean branch only (aug has waveform-level aug)
            mel_clean_aug = freq_masking(mel_clean_raw)
            mel_clean_aug = time_masking(mel_clean_aug)
            # Recombine: [SpecAugmented-clean | raw-aug]
            mel_db_combined = torch.cat([mel_clean_aug, mel_aug_raw], dim=0)

            # Combined waveform batch
            combined     = torch.cat([waveforms, aug_waveforms], dim=0)
            combined_lbl = torch.cat([labels,    labels],        dim=0)

            # ── Forward pass ───────────────────────────────────────────────────
            out_meta, out_a, out_r = wrapper_model(
                combined, mel_db_combined, return_base_outs=True
            )

            B = waveforms.size(0)

            # Split and cast to FP32
            out_meta_clean = out_meta[:B].float()
            out_meta_deg   = out_meta[B:].float()
            out_a_clean    = out_a[:B].float()
            out_r_clean    = out_r[:B].float()

            # ── FIX 5: phased loss computation ─────────────────────────────────
            loss_clean = criterion(out_meta_clean, labels)
            loss_deg   = criterion(out_meta_deg,   labels)

            # π-model consistency (Laine & Aila 2017)
            loss_cons  = F.mse_loss(
                F.softmax(out_meta_clean, dim=1),
                F.softmax(out_meta_deg,   dim=1),
            )

            # Base-model auxiliary loss (clean branch only)
            loss_aux = criterion(out_a_clean, labels) + criterion(out_r_clean, labels)

            loss_total = (
                clean_w * loss_clean +
                deg_w   * loss_deg   +
                aux_w   * loss_aux   +
                cons_w  * loss_cons
            )

            if torch.isnan(loss_total) or torch.isinf(loss_total):
                nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                if nan_batches <= 3:
                    print(f"\n  [Warning] NaN/Inf at epoch {epoch+1}. Skipping.")
                continue

            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(wrapper_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # ── PD gap signal accumulation ─────────────────────────────────────
            with torch.no_grad():
                prob_c = torch.softmax(out_meta_clean, dim=1)
                prob_a = torch.softmax(out_meta_deg,   dim=1)
                conf_c = prob_c.gather(1, labels.view(-1, 1)).mean().item() * 100
                conf_a = prob_a.gather(1, labels.view(-1, 1)).mean().item() * 100
                epoch_gaps.append(conf_c - conf_a)
                train_labels.extend(labels.cpu().numpy())
                train_probs.extend(prob_c[:, 1].cpu().numpy())

            sum_total += loss_total.item()
            sum_clean += loss_clean.item()
            sum_deg   += loss_deg.item()
            sum_aux   += loss_aux.item()
            sum_cons  += loss_cons.item()

            pbar.set_postfix({
                "loss":  f"{loss_total.item():.4f}",
                "α":     f"{alpha:.3f}",
                "gap":   f"{(conf_c - conf_a):.1f}pp",
                "mw":    f"{clean_w+deg_w:.2f}",
            })

        # ── PD controller update (Phase 2 only) ───────────────────────────────
        mean_gap  = float(np.mean(epoch_gaps)) if epoch_gaps else 0.0
        if not in_phase1:
            new_alpha = controller.update(mean_gap)
        else:
            new_alpha = PHASE1_ALPHA   # fixed during Phase 1, no update

        n_b       = max(1, len(train_loader) - nan_batches)
        avg_train = sum_total / n_b
        avg_clean = sum_clean / n_b
        avg_deg   = sum_deg   / n_b
        avg_aux   = sum_aux   / n_b
        avg_cons  = sum_cons  / n_b
        train_eer = compute_eer(train_labels, train_probs)

        # ── Validation phase ──────────────────────────────────────────────────
        wrapper_model.eval()
        sum_val  = 0.0
        val_lc, val_pc = [], []
        val_la, val_pa = [], []

        with torch.no_grad():
            for wv, lv in tqdm(val_loader,
                               desc=f"Epoch {epoch+1} [Valid]", leave=False):
                wv = wv.squeeze(1).to(device)
                lv = lv.to(device)

                # Clean validation: no SpecAugment (identical to baseline v3)
                mel    = mel_transform(wv)
                mel_db = amp_to_db(mel).unsqueeze(1)
                out_c  = wrapper_model(wv, mel_db)
                sum_val += criterion(out_c.float(), lv).item()
                val_lc.extend(lv.cpu().numpy())
                val_pc.extend(
                    torch.softmax(out_c.float(), dim=1)[:, 1].cpu().numpy()
                )

                # Augmented validation: SSI at max(0.3, current_alpha)
                aug_v      = actuator._ssi(wv, alpha=max(0.3, current_alpha))
                mel_aug    = mel_transform(aug_v)
                mel_db_aug = amp_to_db(mel_aug).unsqueeze(1)
                out_av     = wrapper_model(aug_v, mel_db_aug)
                val_la.extend(lv.cpu().numpy())
                val_pa.extend(
                    torch.softmax(out_av.float(), dim=1)[:, 1].cpu().numpy()
                )

        avg_val     = sum_val / len(val_loader)
        eer_clean   = compute_eer(val_lc, val_pc)
        eer_aug     = compute_eer(val_la, val_pa)
        composite   = CKPT_CLEAN_W * eer_clean + CKPT_AUG_W * eer_aug
        val_auc     = roc_auc_score(val_lc, val_pc)
        val_min_dcf = compute_min_dcf(val_lc, val_pc)

        scheduler.step()

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["loss_clean"].append(avg_clean)
        history["loss_deg"].append(avg_deg)
        history["loss_aux"].append(avg_aux)
        history["loss_cons"].append(avg_cons)
        history["train_eer"].append(train_eer)
        history["val_eer_clean"].append(eer_clean)
        history["val_eer_aug"].append(eer_aug)
        history["val_auc"].append(val_auc)
        history["val_min_dcf"].append(val_min_dcf)
        history["composite"].append(composite)
        history["alpha"].append(new_alpha)
        history["clean_w"].append(clean_w)
        history["deg_w"].append(deg_w)
        history["aux_w"].append(aux_w)

        elapsed  = time.time() - start_time
        eta_secs = int((elapsed / (epoch + 1)) * (TOTAL_EPOCHS - epoch - 1))
        eta_str  = str(datetime.timedelta(seconds=eta_secs))
        cur_lr   = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:3d} [{phase_tag}] | LR {cur_lr:.2e} | "
            f"α {new_alpha:.3f} | gap {mean_gap:.1f}pp | "
            f"Train {avg_train:.4f} | Val {avg_val:.4f}"
        )
        print(
            f"           EER_c {eer_clean:.4f}% | EER_a {eer_aug:.4f}% | "
            f"Score {composite:.4f}% | AUC {val_auc:.4f} | "
            f"minDCF {val_min_dcf:.4f} | ETA {eta_str}"
        )

        if composite < best_composite:
            best_composite    = composite
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch":            epoch + 1,
                    "model_state_dict": wrapper_model.state_dict(),
                    "eer_clean":        eer_clean,
                    "eer_aug":          eer_aug,
                    "composite":        composite,
                    "val_auc":          val_auc,
                    "val_min_dcf":      val_min_dcf,
                    "alpha":            new_alpha,
                },
                OUTPUT_WEIGHTS,
            )
            print(
                f"  -> Composite {best_composite:.4f}% "
                f"(EER_c={eer_clean:.4f}%, EER_a={eer_aug:.4f}%) — saved ✓"
            )
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement ({epochs_no_improve}/{PATIENCE})")

        if epochs_no_improve >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    wrapper_model.remove_hooks()
    total_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"\nTotal time: {total_str} | Best composite: {best_composite:.4f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # Diagnostic Plots
    # ══════════════════════════════════════════════════════════════════════════
    E   = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    axes = axes.flatten()

    # ── Plot 1: Loss trajectory ───────────────────────────────────────────────
    axes[0].plot(E, history["train_loss"], label="Train Loss (Total)", color="blue")
    axes[0].plot(E, history["val_loss"],   label="Val Loss (clean)",   color="red",        ls="--")
    axes[0].plot(E, history["loss_clean"], label="L_clean",            color="steelblue",  ls=":")
    axes[0].plot(E, history["loss_deg"],   label="L_deg",              color="darkorange", ls=":")
    axes[0].plot(E, history["loss_aux"],   label="L_aux (bases)",      color="green",      ls=":")
    axes[0].plot(E, history["loss_cons"],  label="L_cons",             color="purple",     ls=":")
    if PHASE1_END < len(history["train_loss"]):
        axes[0].axvline(x=PHASE1_END, color="gray", ls="--", alpha=0.6,
                        label=f"Phase switch (ep {PHASE1_END})")
    axes[0].set_title("Loss Trajectory (Phased Weights)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7); axes[0].grid(True, ls=":", alpha=0.6)

    # ── Plot 2: EER ───────────────────────────────────────────────────────────
    axes[1].plot(E, history["val_eer_clean"], label="EER_clean %", color="green")
    axes[1].plot(E, history["val_eer_aug"],   label="EER_aug %",   color="purple", ls="--")
    axes[1].plot(E, history["train_eer"],     label="Train EER %", color="teal",   ls=":")
    if PHASE1_END < len(history["val_eer_clean"]):
        axes[1].axvline(x=PHASE1_END, color="gray", ls="--", alpha=0.6)
    axes[1].set_title("Equal Error Rate (↓ Better)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("EER (%)")
    axes[1].legend(); axes[1].grid(True, ls=":", alpha=0.6)

    # ── Plot 3: Composite checkpoint metric ───────────────────────────────────
    axes[2].plot(E, history["composite"], label="Composite score", color="navy")
    if PHASE1_END < len(history["composite"]):
        axes[2].axvline(x=PHASE1_END, color="gray", ls="--", alpha=0.6)
    axes[2].set_title(
        f"Checkpoint Metric (↓ Better)\n= {CKPT_CLEAN_W}·EER_c + {CKPT_AUG_W}·EER_a"
    )
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Composite EER (%)")
    axes[2].legend(); axes[2].grid(True, ls=":", alpha=0.6)

    # ── Plot 4: AUC ───────────────────────────────────────────────────────────
    axes[3].plot(E, history["val_auc"], label="ROC-AUC", color="darkgreen")
    axes[3].axhline(y=0.5, ls=":", color="gray", alpha=0.5, label="Random baseline")
    axes[3].set_title("Validation AUC (↑ Better)")
    axes[3].set_xlabel("Epoch"); axes[3].set_ylabel("AUC")
    axes[3].set_ylim(0.4, 1.02)
    axes[3].legend(); axes[3].grid(True, ls=":", alpha=0.6)

    # ── Plot 5: minDCF ────────────────────────────────────────────────────────
    axes[4].plot(E, history["val_min_dcf"], label="minDCF", color="darkred")
    axes[4].axhline(y=1.0, ls=":", color="gray", alpha=0.5, label="Chance baseline")
    axes[4].set_title("Validation minDCF (↓ Better)")
    axes[4].set_xlabel("Epoch"); axes[4].set_ylabel("Normalised minDCF")
    axes[4].legend(); axes[4].grid(True, ls=":", alpha=0.6)

    # ── Plot 6: Alpha + phase weights ─────────────────────────────────────────
    ax6  = axes[5]
    ax6b = ax6.twinx()
    ax6.plot(E, history["alpha"],   label="α (aug intensity)", color="orange",  lw=2)
    ax6b.plot(E, history["clean_w"], label="clean_w+deg_w",    color="navy",    ls=":", lw=1.5,
              alpha=0.7)
    ax6b.plot(E, history["aux_w"],   label="aux_w",            color="darkcyan",ls=":", lw=1.5,
              alpha=0.7)
    ax6.axhline(y=controller.alpha_min, ls=":", color="gray",  alpha=0.5,
                label=f"α_min={controller.alpha_min}")
    ax6.axhline(y=controller.alpha_max, ls=":", color="red",   alpha=0.5,
                label=f"α_max={controller.alpha_max}")
    if PHASE1_END < len(history["alpha"]):
        ax6.axvline(x=PHASE1_END, color="gray", ls="--", alpha=0.6,
                    label=f"Phase switch (ep {PHASE1_END})")
    ax6.set_title("UR-FFL Alpha + Phased Loss Weights")
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("α", color="orange")
    ax6b.set_ylabel("Loss weight", color="navy")
    ax6.set_ylim(0.0, 1.0); ax6b.set_ylim(0.0, 1.0)
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6b.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
    ax6.grid(True, ls=":", alpha=0.6)

    fig.suptitle(
        "UR-FFL Cross-Attention Ensemble — Cold-Start v3\n"
        "(Phased Training + SpecAugment on Clean Branch + PD Phase Activation)",
        fontsize=13,
    )
    fig.tight_layout()
    gp = os.path.join(RESULTS_DIR, "crossattention_ensemble_urffl_metrics.png")
    fig.savefig(gp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plots saved to {gp}")


if __name__ == "__main__":
    main()