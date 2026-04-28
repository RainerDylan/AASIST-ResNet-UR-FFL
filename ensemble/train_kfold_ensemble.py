import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, ConcatDataset
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..')) if 'ensemble' in CURRENT_DIR else CURRENT_DIR
sys.path.append(ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

from src.data.dataset import ASVspoofDataset 
from src.models.aasist import AASIST
from src.models.resnet_simam import resnet18_simam
from src.ur_ffl.sensor import UncertaintySensor
from src.ur_ffl.controller import PDController
from src.ur_ffl.selector import DegradationSelector
from src.ur_ffl.actuator import DegradationActuator

PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PREPROCESSED_DEV_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
PROTOCOL_DEV = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"

NUM_FOLDS = 5
EPOCHS_PER_FOLD = 40
BATCH_SIZE = 16
LR = 1e-4

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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

class MetaLearner(nn.Module):
    def __init__(self, input_dim=616, hidden_dim=256, num_classes=2, dropout=0.3):
        super(MetaLearner, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, emb_aasist, emb_resnet):
        if emb_aasist.dtype != emb_resnet.dtype:
            emb_aasist = emb_aasist.to(emb_resnet.dtype)
        x = torch.cat([emb_aasist, emb_resnet], dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

class EndToEndEnsemble(nn.Module):
    def __init__(self, aasist, resnet, meta_learner):
        super(EndToEndEnsemble, self).__init__()
        self.aasist = aasist
        self.resnet = resnet
        self.meta_learner = meta_learner

        self.emb_a = [None]
        self.emb_r = [None]

        def hook_a(module, inp, out):
            self.emb_a[0] = inp[0]
            
        def hook_r(module, inp, out):
            self.emb_r[0] = inp[0]

        self.h_a = self.aasist.fc.register_forward_hook(hook_a)
        self.h_r = self.resnet.fc.register_forward_hook(hook_r)

    def forward(self, waveform, mel_db, return_base_outs=False):
        out_a = self.aasist(waveform)
        
        with torch.amp.autocast('cuda'):
            out_r = self.resnet(mel_db)
            out_meta = self.meta_learner(self.emb_a[0], self.emb_r[0])
            
        if return_base_outs:
            return out_meta, out_a, out_r
        return out_meta

def init_aasist_cold_start(model):
    attn_kw = ('attn', 'gat', 'attention', 'query', 'key', 'value')
    for name, module in model.named_modules():
        is_attn = any(kw in name.lower() for kw in attn_kw)
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            if is_attn: nn.init.xavier_uniform_(module.weight)
            else: nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            if is_attn: nn.init.xavier_uniform_(module.weight)
            else: nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None: nn.init.zeros_(module.bias)

def init_resnet_cold_start(model):
    for full_name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=1e-5) 
            if module.bias is not None: nn.init.zeros_(module.bias)

def create_fold_sampler(fold_labels):
    class_counts = torch.bincount(torch.tensor(fold_labels))
    total_samples = len(fold_labels)
    class_weights = total_samples / class_counts.float()
    sample_weights = [class_weights[label] for label in fold_labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=total_samples, replacement=True)

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[idx] * 100

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initiating {NUM_FOLDS}-Fold End-to-End Ensemble Validation on {device}")

    print("Loading and Merging Train and Dev Datasets to memory")
    train_dataset = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    val_dataset = ASVspoofDataset(PREPROCESSED_DEV_DIR, PROTOCOL_DEV)
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    combined_labels = np.concatenate([train_dataset.labels, val_dataset.labels])
    print(f"Successfully merged datasets: {len(combined_labels)} total samples.")

    kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_results_eer = []

    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80).to(device)
    amp_to_db = T.AmplitudeToDB(stype='power', top_db=80).to(device)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(combined_labels)), combined_labels)):
        print(f"\n========== FOLD {fold + 1}/{NUM_FOLDS} ==========")
        
        train_subset = Subset(combined_dataset, train_idx)
        val_subset = Subset(combined_dataset, val_idx)
        
        fold_train_labels = combined_labels[train_idx]
        sampler = create_fold_sampler(fold_train_labels)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=False)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)

        aasist_model = AASIST(stft_window=698, stft_hop=398, freq_bins=116, gat_layers=2, heads=5, head_dim=104, hidden_dim=455, dropout=0.33)
        init_aasist_cold_start(aasist_model)
        
        resnet_model = resnet18_simam(num_classes=2, dropout_rate=0.22)
        init_resnet_cold_start(resnet_model)
        
        meta_learner = MetaLearner(input_dim=616)
        wrapper_model = EndToEndEnsemble(aasist_model, resnet_model, meta_learner).to(device)

        sensor = UncertaintySensor(mc_passes=5)
        controller = PDController()
        selector = DegradationSelector()
        actuator = DegradationActuator(device)

        criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
        optimizer = optim.AdamW(wrapper_model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_FOLD, eta_min=1e-7)
        scaler = torch.amp.GradScaler('cuda')
        
        best_fold_composite = float('inf')
        fold_weight_path = os.path.join(MODELS_DIR, f"meta_ensemble_urffl_fold_{fold + 1}.pth")
        final_fold_eer = 100.0

        for epoch in range(EPOCHS_PER_FOLD):
            wrapper_model.train()
            train_loss = 0.0
            epoch_gaps = []
            
            pbar = tqdm(train_loader, desc=f"Fold {fold+1} | Epoch {epoch+1}/{EPOCHS_PER_FOLD} [Train]", leave=False)
            for waveforms, batch_labels in pbar:
                waveforms = waveforms.squeeze(1).to(device)
                batch_labels = batch_labels.to(device)

                with torch.no_grad():
                    z_u, _ = sensor.measure(wrapper_model.aasist, waveforms)
                    
                selections = selector.select(z_u)
                alpha = controller.alpha
                aug_waveforms = actuator.apply(waveforms, batch_labels, selections, alpha)

                optimizer.zero_grad()
                combined = torch.cat([waveforms, aug_waveforms], dim=0)
                combined_lbl = torch.cat([batch_labels, batch_labels], dim=0)

                mel_combined = mel_transform(combined)
                mel_db_combined = amp_to_db(mel_combined).unsqueeze(1)

                out_meta, out_a, out_r = wrapper_model(combined, mel_db_combined, return_base_outs=True)
                    
                B = waveforms.size(0)
                out_meta_clean = out_meta[:B]
                out_meta_deg = out_meta[B:]
                
                loss_clean = criterion(out_meta_clean, batch_labels)
                loss_deg = criterion(out_meta_deg, batch_labels)
                loss_cons = F.mse_loss(F.softmax(out_meta_clean, dim=1), F.softmax(out_meta_deg, dim=1))
                
                loss_aux = criterion(out_a[:B], batch_labels) + criterion(out_r[:B], batch_labels)
                loss_total = (0.40 * loss_clean) + (0.40 * loss_deg) + (0.10 * loss_cons) + (0.10 * loss_aux)

                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(wrapper_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                with torch.no_grad():
                    prob_c = torch.softmax(out_meta_clean, dim=1)
                    prob_a = torch.softmax(out_meta_deg, dim=1)
                    conf_c = prob_c.gather(1, batch_labels.view(-1, 1)).mean().item() * 100
                    conf_a = prob_a.gather(1, batch_labels.view(-1, 1)).mean().item() * 100
                
                epoch_gaps.append(conf_c - conf_a)
                train_loss += loss_total.item()
                
            mean_gap = float(np.mean(epoch_gaps))
            _ = controller.update(mean_gap)
            avg_train_loss = train_loss / len(train_loader)
            
            wrapper_model.eval()
            val_loss = 0.0
            lc, pc, la, pa = [], [], [], []
            
            with torch.no_grad():
                for wv, lv in val_loader:
                    wv = wv.squeeze(1).to(device)
                    lv = lv.to(device)
                    
                    mel_wv = mel_transform(wv)
                    mel_db_wv = amp_to_db(mel_wv).unsqueeze(1)
                    out_c = wrapper_model(wv, mel_db_wv)
                    val_loss += criterion(out_c, lv).item()
                    
                    lc.extend(lv.cpu().numpy())
                    pc.extend(torch.softmax(out_c, dim=1)[:, 1].cpu().numpy())

                    aug_v = actuator._ssi(wv, alpha=max(0.3, controller.alpha))
                    mel_aug = mel_transform(aug_v)
                    mel_db_aug = amp_to_db(mel_aug).unsqueeze(1)
                    out_a = wrapper_model(aug_v, mel_db_aug)
                    
                    la.extend(lv.cpu().numpy())
                    pa.extend(torch.softmax(out_a, dim=1)[:, 1].cpu().numpy())
                    
            avg_val_loss = val_loss / len(val_loader)
            eer_clean = compute_eer(lc, pc)
            eer_aug = compute_eer(la, pa)
            composite = (0.30 * eer_clean) + (0.70 * eer_aug)
            scheduler.step()
            
            if composite < best_fold_composite:
                best_fold_composite = composite
                final_fold_eer = eer_clean
                torch.save(wrapper_model.state_dict(), fold_weight_path)
                
            print(f"Fold {fold + 1} | Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val EER: {eer_clean:.4f}% | Score: {composite:.4f}%")

        fold_results_eer.append(final_fold_eer)
        print(f"Best EER for Fold {fold + 1}: {final_fold_eer:.4f}% saved to {fold_weight_path}")

    print("\n========================================")
    print("K-FOLD CROSS VALIDATION RESULTS")
    print("========================================")
    for i, eer in enumerate(fold_results_eer):
        print(f"Fold {i + 1} EER: {eer:.4f}%")
        
    mean_eer = np.mean(fold_results_eer)
    std_eer = np.std(fold_results_eer)
    print(f"\nFinal Meta-Learner Validation EER: {mean_eer:.4f}% ± {std_eer:.4f}%")
    print("========================================")

if __name__ == "__main__":
    main()