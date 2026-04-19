"""
train_aasist_unified.py — Unified UR-FFL Training (v-final).

Four mandatory fixes:

1. COLD START (Kaiming/Xavier weight init)
   ─────────────────────────────────────────
   Warm-start from Phase-1 begins with a model fully converged on clean LA-2019
   audio.  Its encoder has memorised spectral shortcuts (silence padding, codec
   fingerprints) that do not exist in DF-2021.  RawBoost gradients cannot overwrite
   these shortcuts when the Phase-1 loss surface is so flat.
   Fix: reinitialise all weights from scratch before epoch 1.
   - Conv1d / Conv2d:  Kaiming He et al. (2015) — fan_out, relu nonlinearity
   - Linear:           Kaiming He et al. (2015) — fan_in
   - Graph Attention:  Xavier Glorot & Bengio (2010) — preserves gradient variance
     through the attention dot-product, which is quadratic in feature dimension
   - BatchNorm:        ones/zeros (standard)

   Justification: He et al. (2015) showed that Kaiming init is necessary for
   layers with ReLU activations to prevent vanishing/exploding gradients from the
   start of training.  Xavier is preferred for attention layers (no ReLU) because
   it preserves variance through the softmax attention mapping.

2. DROPNODE ON GRAPH ATTENTION LAYERS (Feng et al. 2020 / Rong et al. ICLR 2020)
   ──────────────────────────────────────────────────────────────────────────────
   AASIST uses 2 heterogeneous GAT layers.  With only clean LA-2019 data,
   nodes (spectral/temporal graph vertices) develop highly correlated
   representations — over-smoothing (Li et al. 2018).  DropNode (randomly
   zero entire node feature vectors with probability p=0.10) disrupts this
   correlation by forcing redundant path learning (Feng et al. 2020).
   Applied via forward hooks — no AASIST model modification required.

3. RAWBOOST AUGMENTATION (Tak et al. ICASSP 2022)
   ─────────────────────────────────────────────────
   All previous STFT simulations produced acc_gap ≈10-12pp independent of alpha,
   making the PD controller blind.  RawBoost (LnL + ISD + SSI) creates:
       alpha=0.10 → gap≈1pp  (trivially easy)
       alpha=0.50 → gap≈14pp (at setpoint; model challenged)
       alpha=0.70 → gap≈48pp (model overwhelmed → alpha falls)
   This monotone alpha↔gap relationship is the prerequisite for any PD controller.

4. COMPOSITE CHECKPOINT + EER_C GUARD
   ─────────────────────────────────────
   Save on: composite = 0.3*EER_c + 0.7*EER_a  AND  EER_c < 3.0%
   Prevents saving the epoch-1 Phase-1-like model (EER_c=0.67%, EER_a=37.5%)
   which scores 26.2% composite — worse than epoch 11 (EER_c=2.3%, EER_a=32.3%,
   score=23.3%).

References:
    He et al. (2015) ICCV: Kaiming initialisation
    Glorot & Bengio (2010): Xavier initialisation
    Rong et al. (2020) ICLR: DropEdge — over-smoothing in GNNs
    Feng et al. (2020): DropNode / GRAND — random node feature dropping
    Tak et al. (2022) ICASSP: RawBoost augmentation
    Bengio et al. (2009) ICML: curriculum learning
"""

import sys, os, time, datetime, contextlib
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
LR             = 1e-4    # higher than warm-start since we're training from scratch

CLEAN_WEIGHT   = 0.45
DEG_WEIGHT     = 0.45
CONS_WEIGHT    = 0.10

DROPNODE_P     = 0.10    # probability of zeroing an entire graph node's features

CKPT_CLEAN_W   = 0.30
CKPT_AUG_W     = 0.70
EER_C_MAX      = 3.0     # guard: never save if model catastrophically forgets clean

SWA_START_FRAC = 0.80
SWA_LR         = 1e-6
PATIENCE       = 20

# ── weight initialisation ─────────────────────────────────────────────────────

def init_weights_cold_start(model: nn.Module) -> None:
    """
    Full cold-start weight initialisation.

    Conv / Linear (ReLU activations):
        Kaiming He et al. (2015): Var(W) = 2/fan_in
        Prevents gradient vanishing for deep ReLU networks.

    Graph Attention layers (dot-product attention, no nonlinearity between
    query and key):
        Xavier Glorot & Bengio (2010): Var(W) = 2/(fan_in + fan_out)
        Preserves signal variance through the attention softmax.
        Identified by layer name containing 'attn', 'gat', or 'attention'.

    BatchNorm: weight=1, bias=0 (standard).
    """
    attn_kw = ('attn', 'gat', 'attention', 'query', 'key', 'value')

    for name, module in model.named_modules():
        is_attn = any(kw in name.lower() for kw in attn_kw)

        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            if is_attn:
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            if is_attn:
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # Explicitly Xavier-initialise any remaining GAT parameters by name
    for pname, param in model.named_parameters():
        if any(kw in pname.lower() for kw in attn_kw) and param.dim() >= 2:
            nn.init.xavier_uniform_(param)

    print("  Cold start: Kaiming init (Conv/Linear) + Xavier init (GAT/attn layers).")


# ── DropNode via forward hooks ────────────────────────────────────────────────

class DropNodeHooks:
    """
    DropNode (Feng et al. 2020 / Rong et al. ICLR 2020) via PyTorch forward hooks.

    Randomly zeros entire node feature vectors during the forward pass.
    Applied ONLY to graph attention (GAT) layers, identified by name.
    Effect:
      - Prevents over-smoothing: nodes cannot always rely on the same neighbours
      - Acts as data augmentation on the graph structure (Rong et al. 2020)
      - Forces redundant pathway learning: model cannot memorise specific node roles

    No modification of AASIST model source code required.
    """

    def __init__(self, p: float = 0.10):
        self.p       = p
        self._hooks  = []
        self._active = False

    def register(self, model: nn.Module) -> int:
        """Register hooks on all GAT-related modules. Returns number of hooks."""
        gat_kw = ('gat', 'graph', 'attn', 'attention')
        count  = 0
        for name, module in model.named_modules():
            if any(kw in name.lower() for kw in gat_kw) and name != '':
                h = module.register_forward_hook(self._hook_fn)
                self._hooks.append(h)
                count += 1
        self._active = True
        return count

    def _hook_fn(self, module, inp, output):
        if not (self._active and module.training):
            return output
        if isinstance(output, torch.Tensor) and output.dim() >= 2:
            # Zero complete node vectors along dim 0 (nodes) or batch dim
            mask = (torch.rand(output.shape[0], *([1] * (output.dim() - 1)),
                               device=output.device) > self.p).float()
            return output * mask
        return output

    def enable(self):
        self._active = True

    def disable(self):
        self._active = False

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._active = False


# ── helpers ───────────────────────────────────────────────────────────────────

def create_weighted_sampler(dataset):
    labels = dataset.labels
    cc     = torch.bincount(torch.tensor(labels))
    cw     = len(labels) / cc.float()
    return WeightedRandomSampler([cw[l] for l in labels], len(labels), replacement=True)


def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr         = 1.0 - tpr
    return fpr[np.nanargmin(np.abs(fnr - fpr))] * 100.0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Unified UR-FFL Training (cold start) — device: {device}")

    model = AASIST(
        stft_window=698, stft_hop=398, freq_bins=116,
        gat_layers=2, heads=5, head_dim=104,
        hidden_dim=455, dropout=0.3311465671378094,
    ).to(device)

    # Cold start: full weight reinitialisation
    init_weights_cold_start(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  All {n_params:,} parameters reinitialised. Training from scratch.")

    # DropNode hooks on GAT layers
    dropnode = DropNodeHooks(p=DROPNODE_P)
    n_hooks  = dropnode.register(model)
    print(f"  DropNode (p={DROPNODE_P}) registered on {n_hooks} GAT-related modules.")

    swa_model  = AveragedModel(model)
    swa_start  = int(TOTAL_EPOCHS * SWA_START_FRAC)
    swa_active = False
    print(f"  SWA starts at epoch {swa_start+1}/{TOTAL_EPOCHS}")

    train_ds = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    val_ds   = ASVspoofDataset(PREPROCESSED_DEV_DIR,   PROTOCOL_DEV)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=create_weighted_sampler(train_ds),
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    sensor     = UncertaintySensor(mc_passes=5)
    controller = PDController()
    selector   = DegradationSelector()
    actuator   = DegradationActuator(device)

    optimizer     = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler     = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=swa_start, eta_min=1e-7)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    criterion     = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_composite    = float("inf")
    epochs_no_improve = 0
    history = dict(train_loss=[], val_loss=[], eer_clean=[], eer_aug=[], alpha=[], composite=[])
    total_time = 0.0

    for epoch in range(TOTAL_EPOCHS):
        t0 = time.time()
        model.train()
        dropnode.enable()  # DropNode active during training only

        train_loss = 0.0
        epoch_gaps = []
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
            dropnode.enable()
            optimizer.zero_grad()

            # Fused forward pass (single batch through model)
            combined  = torch.cat([waveforms, aug_waveforms], dim=0)
            out_comb  = model(combined)
            B         = waveforms.size(0)
            out_clean = out_comb[:B]
            out_deg   = out_comb[B:]

            loss_clean = criterion(out_clean, labels)
            loss_deg   = criterion(out_deg,   labels)
            # π-model student-student consistency (Laine & Aila 2016)
            loss_cons  = F.mse_loss(F.softmax(out_clean, dim=1),
                                    F.softmax(out_deg,   dim=1))

            loss = CLEAN_WEIGHT*loss_clean + DEG_WEIGHT*loss_deg + CONS_WEIGHT*loss_cons

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                acc_c = (out_clean.argmax(1) == labels).float().mean().item() * 100
                acc_a = (out_deg.argmax(1)   == labels).float().mean().item() * 100
            epoch_gaps.append(acc_c - acc_a)
            train_loss += loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "α":    f"{alpha:.3f}",
                "gap":  f"{acc_c-acc_a:.1f}pp",
            })

        # PD controller update: one call per epoch, after training loop
        mean_gap  = float(np.mean(epoch_gaps))
        new_alpha = controller.update(mean_gap)
        avg_train = train_loss / len(train_loader)

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            if not swa_active:
                swa_active = True
                print(f"  [SWA] activated at epoch {epoch+1}")
        else:
            scheduler.step()

        # Validation — DropNode disabled; eval on both clean and SSI-augmented audio
        dropnode.disable()
        eval_model = swa_model if swa_active else model
        if swa_active:
            update_bn(train_loader, swa_model, device=device)

        eval_model.eval()
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

                # SSI-augmented validation: same proxy used by controller
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
        eta_s = int((total_time / (epoch+1)) * (TOTAL_EPOCHS - epoch - 1))
        print(
            f"Epoch {epoch+1:3d} | LR {optimizer.param_groups[0]['lr']:.2e} | "
            f"α {new_alpha:.3f} | gap {mean_gap:.1f}pp | "
            f"Train {avg_train:.4f} | Val {avg_val:.4f} | "
            f"EER_c {eer_clean:.4f}% | EER_a {eer_aug:.4f}% | "
            f"Score {composite:.3f}% | SWA {'ON' if swa_active else 'off'} | "
            f"ETA {str(datetime.timedelta(seconds=eta_s))}"
        )

        # Checkpoint on composite score WITH clean EER guard
        if composite < best_composite and eer_clean < EER_C_MAX:
            best_composite    = composite
            epochs_no_improve = 0
            torch.save(
                swa_model.module.state_dict() if swa_active else model.state_dict(),
                OUTPUT_WEIGHTS
            )
            print(f"  -> Best composite {best_composite:.3f}% "
                  f"(EER_c={eer_clean:.2f}%, EER_a={eer_aug:.2f}%) — saved")
        elif eer_clean >= EER_C_MAX:
            epochs_no_improve += 1
            print(f"  -> Guard: EER_c={eer_clean:.2f}% >= {EER_C_MAX}%  "
                  f"({epochs_no_improve}/{PATIENCE})")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement ({epochs_no_improve}/{PATIENCE})")

        if epochs_no_improve >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # ── plots ─────────────────────────────────────────────────────────────────
    dropnode.remove()
    E = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    axes[0].plot(E, history["train_loss"], label="Train", color="blue")
    axes[0].plot(E, history["val_loss"],   label="Val",   color="red", ls="--")
    axes[0].set_title("CE Loss"); axes[0].legend(); axes[0].grid(True, ls=":", alpha=0.6)

    axes[1].plot(E, history["eer_clean"], label="EER clean %",  color="green")
    axes[1].plot(E, history["eer_aug"],   label="EER aug %",    color="purple", ls="--")
    axes[1].axhline(y=EER_C_MAX, ls=":", color="red", alpha=0.7, label=f"EER_c guard={EER_C_MAX}%")
    axes[1].set_title("EER (↓ better)"); axes[1].legend(); axes[1].grid(True, ls=":", alpha=0.6)

    axes[2].plot(E, history["composite"], label="Composite score", color="navy")
    axes[2].set_title("Checkpoint Metric (↓ better)"); axes[2].legend()
    axes[2].grid(True, ls=":", alpha=0.6)

    axes[3].plot(E, history["alpha"], label="α (RawBoost severity)", color="orange")
    axes[3].axhline(y=controller.setpoint/40, ls=":", color="gray", alpha=0.5)
    axes[3].set_title("UR-FFL Alpha"); axes[3].legend(); axes[3].grid(True, ls=":", alpha=0.6)

    fig.tight_layout()
    gp = os.path.join(RESULTS_DIR, "aasist_unified_metrics.png")
    fig.savefig(gp, dpi=300); plt.close(fig)
    print(f"Graphs saved to {gp}")
    print(f"\nBest composite = {best_composite:.3f}%")
    print(f"Weights: {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()