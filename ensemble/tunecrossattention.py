import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Subset
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
from src.models.resnet_simam import resnet18_simam

PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")

# NOTE: The base models must use the exact dimensions they were trained with. 
# Do not modify these.
AASIST_WEIGHTS = os.path.join(MODELS_DIR, "aasist_unified_best.pth")
RESNET_WEIGHTS = os.path.join(MODELS_DIR, "resnet_unified_best.pth")

class CrossAttentionFuser(nn.Module):
    # UPDATED: dim_a changed to 51 to match the new AASIST head_dim
    def __init__(self, dim_a=51, dim_r=512, embed_dim=256, num_heads=8, num_classes=2, dropout=0.3):
        super().__init__()
        self.proj_a = nn.Sequential(nn.Linear(dim_a, embed_dim), nn.LayerNorm(embed_dim))
        self.proj_r = nn.Sequential(nn.Linear(dim_r, embed_dim), nn.LayerNorm(embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, emb_a, emb_r):
        if emb_a.dtype != emb_r.dtype: emb_a = emb_a.to(emb_r.dtype)
        B = emb_a.size(0)
        feat_a = self.proj_a(emb_a).unsqueeze(1) 
        feat_r = self.proj_r(emb_r).unsqueeze(1) 
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, feat_a, feat_r), dim=1) 
        x = self.transformer(x)
        return self.head(x[:, 0, :])

class EndToEndEnsemble(nn.Module):
    def __init__(self, aasist, resnet, fusion_head):
        super().__init__()
        self.aasist = aasist
        self.resnet = resnet
        self.fusion_head = fusion_head
        self._emb_a = [None]
        self._emb_r = [None]
        def _ha(m, i, o): self._emb_a[0] = i[0]
        def _hr(m, i, o): self._emb_r[0] = i[0]
        self._h_a = self.aasist.fc.register_forward_hook(_ha)
        self._h_r = self.resnet.fc.register_forward_hook(_hr)

    def forward(self, waveform, mel_db):
        _ = self.aasist(waveform)
        with torch.amp.autocast("cuda"):
            _ = self.resnet(mel_db)
            out_meta = self.fusion_head(self._emb_a[0], self._emb_r[0])
        return out_meta

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

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    embed_dim = trial.suggest_categorical("embed_dim", [128, 256, 512])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    print(f"\n[Trial {trial.number}] LR: {lr:.2e} | Embed: {embed_dim} | Heads: {num_heads} | Drop: {dropout:.2f} | WD: {weight_decay:.2e}")

    # UPDATED: Using the exact optimal AASIST parameters
    aasist_model = AASIST(
        stft_window=978, stft_hop=465, freq_bins=179, 
        gat_layers=4, heads=2, head_dim=51, 
        hidden_dim=84, dropout=0.23013213230530574
    )
    # UPDATED: Using the exact optimal ResNet parameters
    resnet_model = resnet18_simam(num_classes=2, dropout_rate=0.1798695128633439)
    
    ckpt_a = torch.load(AASIST_WEIGHTS, map_location=device)
    aasist_model.load_state_dict(ckpt_a['model_state_dict'] if 'model_state_dict' in ckpt_a else ckpt_a)
    ckpt_r = torch.load(RESNET_WEIGHTS, map_location=device)
    resnet_model.load_state_dict(ckpt_r['model_state_dict'] if 'model_state_dict' in ckpt_r else ckpt_r)
    
    # UPDATED: dim_a matches new AASIST head_dim=51
    fusion_head = CrossAttentionFuser(dim_a=51, dim_r=512, embed_dim=embed_dim, num_heads=num_heads, num_classes=2, dropout=dropout)
    wrapper_model = EndToEndEnsemble(aasist_model, resnet_model, fusion_head).to(device)
    
    # Freeze Base Models for Fusion Tuning
    for param in wrapper_model.aasist.parameters():
        param.requires_grad = False
    for param in wrapper_model.resnet.parameters():
        param.requires_grad = False

    dataset = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    train_subset, val_subset = get_balanced_subsets(dataset, train_size=1000, val_size=400)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    optimizer = optim.AdamW(wrapper_model.fusion_head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # UPDATED: Using the exact optimal ResNet frontend parameters
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=128, n_mels=128).to(device)
    amp_to_db = T.AmplitudeToDB(stype="power", top_db=80).to(device)
    scaler = torch.amp.GradScaler("cuda")
    
    epochs = 15
    for epoch in range(epochs):
        wrapper_model.train()
        wrapper_model.aasist.eval()
        wrapper_model.resnet.eval()
        train_loss = 0.0
        
        for waveforms, labels in train_loader:
            waveforms = waveforms.squeeze(1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            with torch.no_grad():
                mel = mel_transform(waveforms)
                mel_db = amp_to_db(mel).unsqueeze(1)
            
            outputs = wrapper_model(waveforms, mel_db)
            loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        wrapper_model.eval()
        val_loss = 0.0
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for wv, lv in val_loader:
                wv = wv.squeeze(1).to(device)
                lv = lv.to(device)
                mel = mel_transform(wv)
                mel_db = amp_to_db(mel).unsqueeze(1)
                
                outputs = wrapper_model(wv, mel_db)
                val_loss += criterion(outputs, lv).item()
                val_labels.extend(lv.cpu().numpy())
                val_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        val_eer = compute_eer(val_labels, val_probs)
        
        print(f"  Epoch {epoch+1:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val EER: {val_eer:.2f}%")
        
        trial.report(val_eer, epoch)
        if trial.should_prune():
            print(f"  [Pruned] Trial {trial.number} halted by Hyperband.")
            raise optuna.exceptions.TrialPruned()
            
    return val_eer

if __name__ == "__main__":
    print("Initiating Frozen Base TPE + Hyperband Tuning for Cross-Attention...")
    pruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=15, reduction_factor=3)
    sampler = optuna.samplers.TPESampler(seed=42)
    
    study = optuna.create_study(
        study_name="crossattention_optimization",
        storage="sqlite:///crossattention_tuning.db",
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
        sampler=sampler
    )
    
    study.optimize(objective, n_trials=30, timeout=14400)
    
    print("\n=================================================")
    print("Optimization Completed!")
    best_trial = study.best_trial
    print(f"Best Validation EER: {best_trial.value:.4f}%")
    
    txt_path = os.path.join(RESULTS_DIR, "crossattention_best_params.txt")
    with open(txt_path, "w") as f:
        f.write("=========================================\n")
        f.write("CROSS-ATTENTION OPTIMAL HYPERPARAMETERS\n")
        f.write("=========================================\n")
        f.write(f"Best Validation EER: {best_trial.value:.4f}%\n\n")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
            f.write(f"{key}: {value}\n")
    print(f"\nSaved optimal parameters to {txt_path}")

    print("\nGenerating Diagnostic Optuna Graphs...")
    fig_hist = vis.plot_optimization_history(study)
    plt.title("Cross-Attention Optimization History")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "crossattention_opt_history.png"), dpi=300)
    plt.close()

    try:
        fig_imp = vis.plot_param_importances(study)
        plt.title("Cross-Attention Hyperparameter Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "crossattention_param_importance.png"), dpi=300)
        plt.close()
    except:
        pass

    fig_slice = vis.plot_slice(study)
    plt.title("Cross-Attention Slice Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "crossattention_slice_plot.png"), dpi=300)
    plt.close()
    print(f"Graphs successfully saved to {RESULTS_DIR}")