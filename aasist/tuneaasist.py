import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import optuna
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as vis

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from src.data.dataset import ASVspoofDataset
from src.models.aasist import AASIST

PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"

def get_balanced_subsets(dataset, train_size=1000, val_size=400):
    bonafide_idx = [i for i, label in enumerate(dataset.labels) if label == 1]
    spoof_idx = [i for i, label in enumerate(dataset.labels) if label == 0]
    
    random.shuffle(bonafide_idx)
    random.shuffle(spoof_idx)
    
    half_train = train_size // 2
    train_indices = bonafide_idx[:half_train] + spoof_idx[:half_train]
    
    half_val = val_size // 2
    val_indices = bonafide_idx[half_train:half_train+half_val] + spoof_idx[half_train:half_train+half_val]
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    return float(fpr[np.nanargmin(np.abs(fnr - fpr))]) * 100.0

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, targets):
        n_cls = logits.shape[1]
        with torch.no_grad():
            smooth = torch.zeros_like(logits).fill_(self.ls / (n_cls - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        pt = (p * smooth).sum(dim=1)
        weight = (1.0 - pt).pow(self.gamma)
        ce = -(smooth * log_p).sum(dim=1)
        return (weight * ce).mean()

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ── Structural Search Space (from official baseline bounds) ─────────
    stft_window = trial.suggest_int("stft_window", 256, 1024)
    stft_hop    = trial.suggest_int("stft_hop", 64, 512)
    freq_bins   = trial.suggest_int("freq_bins", 64, 256)
    gat_layers  = trial.suggest_int("gat_layers", 2, 4)
    heads       = trial.suggest_int("heads", 2, 8)
    head_dim    = trial.suggest_int("head_dim", 32, 128)
    hidden_dim  = trial.suggest_int("hidden_dim", 64, 512)
    
    # ── Regularization & Optimization Search Space (Single Phase) ───────
    dropout     = trial.suggest_float("dropout", 0.1, 0.5)
    lr          = trial.suggest_float("lr", 1e-6, 5e-3, log=True)
    batch_size  = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay= trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    print(f"\n[Trial {trial.number}] Structural: GAT={gat_layers}, Heads={heads}, H_Dim={hidden_dim} | LR: {lr:.2e} | Drop: {dropout:.2f}")
    
    model = AASIST(
        stft_window=stft_window, stft_hop=stft_hop, freq_bins=freq_bins,
        gat_layers=gat_layers, heads=heads, head_dim=head_dim, 
        hidden_dim=hidden_dim, dropout=dropout
    ).to(device)
    
    dataset = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    train_subset, val_subset = get_balanced_subsets(dataset, train_size=1000, val_size=400)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    
    epochs = 15
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for waveforms, labels in train_loader:
            waveforms = waveforms.squeeze(1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for wv, lv in val_loader:
                wv = wv.squeeze(1).to(device)
                lv = lv.to(device)
                outputs = model(wv)
                val_loss += criterion(outputs, lv).item()
                val_labels.extend(lv.cpu().numpy())
                val_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        val_eer = compute_eer(val_labels, val_probs)
        
        print(f"  Epoch {epoch+1:2d}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val EER: {val_eer:.2f}%")
        
        trial.report(val_eer, epoch)
        if trial.should_prune():
            print(f"  [Pruned] Trial {trial.number} underperformed and was halted early by Hyperband.")
            raise optuna.exceptions.TrialPruned()
            
    return val_eer

if __name__ == "__main__":
    print("Initiating TPE + Hyperband Structural Tuning for AASIST (Single Phase)...")
    pruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=15, reduction_factor=3)
    sampler = optuna.samplers.TPESampler(seed=42)
    
    study = optuna.create_study(
        study_name="aasist_structural_optimization",
        storage="sqlite:///aasist_tuning.db",
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
        sampler=sampler
    )
    
    study.optimize(objective, n_trials=50, timeout=14400)
    
    print("\n=================================================")
    print("Optimization Completed!")
    best_trial = study.best_trial
    print(f"Best Validation EER: {best_trial.value:.4f}%")
    
    txt_path = os.path.join(RESULTS_DIR, "aasist_best_params.txt")
    with open(txt_path, "w") as f:
        f.write("=========================================\n")
        f.write("AASIST OPTIMAL HYPERPARAMETERS\n")
        f.write("=========================================\n")
        f.write(f"Best Validation EER: {best_trial.value:.4f}%\n\n")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
            f.write(f"{key}: {value}\n")
    print(f"\nSaved optimal parameters to {txt_path}")

    print("\nGenerating Diagnostic Optuna Graphs...")

    fig_hist = vis.plot_optimization_history(study)
    plt.title("AASIST Tuning Optimization History")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "aasist_opt_history.png"), dpi=300)
    plt.close()

    try:
        fig_imp = vis.plot_param_importances(study)
        plt.title("AASIST Hyperparameter Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "aasist_param_importance.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Skipping Parameter Importance graph. Not enough completed trials yet.")

    fig_slice = vis.plot_slice(study)
    plt.title("AASIST Parameter Distribution Slice Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "aasist_slice_plot.png"), dpi=300)
    plt.close()

    print(f"Graphs successfully saved to {RESULTS_DIR}")