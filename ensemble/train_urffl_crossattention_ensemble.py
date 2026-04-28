import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

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

AASIST_WEIGHTS = os.path.join(MODELS_DIR, "aasist_unified_best.pth")
RESNET_WEIGHTS = os.path.join(MODELS_DIR, "resnet_unified_best.pth")
OUTPUT_WEIGHTS = os.path.join(MODELS_DIR, "crossattention_ensemble_urffl_best.pth")

TOTAL_EPOCHS = 60
BATCH_SIZE = 16

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

class CrossAttentionFuser(nn.Module):
    def __init__(self, dim_a=104, dim_r=512, embed_dim=256, num_heads=8, num_classes=2, dropout=0.3):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, embed_dim)
        self.proj_r = nn.Linear(dim_r, embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, emb_a, emb_r):
        if emb_a.dtype != emb_r.dtype:
            emb_a = emb_a.to(emb_r.dtype)
            
        B = emb_a.size(0)
        feat_a = self.proj_a(emb_a).unsqueeze(1) 
        feat_r = self.proj_r(emb_r).unsqueeze(1) 
        cls_tokens = self.cls_token.expand(B, -1, -1) 

        x = torch.cat((cls_tokens, feat_a, feat_r), dim=1) 
        x = self.transformer(x)
        cls_out = x[:, 0, :]
        return self.fc(cls_out)

class EndToEndEnsemble(nn.Module):
    def __init__(self, aasist, resnet, fusion_head):
        super(EndToEndEnsemble, self).__init__()
        self.aasist = aasist
        self.resnet = resnet
        self.fusion_head = fusion_head
        self.emb_a = [None]
        self.emb_r = [None]

        def hook_a(module, inp, out): self.emb_a[0] = inp[0]
        def hook_r(module, inp, out): self.emb_r[0] = inp[0]

        self.h_a = self.aasist.fc.register_forward_hook(hook_a)
        self.h_r = self.resnet.fc.register_forward_hook(hook_r)

    def forward(self, waveform, mel_db, return_base_outs=False):
        out_a = self.aasist(waveform)
        with torch.amp.autocast('cuda'):
            out_r = self.resnet(mel_db)
            out_meta = self.fusion_head(self.emb_a[0], self.emb_r[0])
            
        if return_base_outs:
            return out_meta, out_a, out_r
        return out_meta

def create_weighted_sampler(dataset):
    labels = dataset.labels
    class_counts = torch.bincount(torch.tensor(labels))
    total_samples = len(labels)
    class_weights = total_samples / class_counts.float()
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=total_samples, replacement=True)

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[idx] * 100

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initiating Warm-Start UR-FFL Cross-Attention Ensemble Training on {device}")

    aasist_model = AASIST(stft_window=698, stft_hop=398, freq_bins=116, gat_layers=2, heads=5, head_dim=104, hidden_dim=455, dropout=0.33)
    checkpoint_a = torch.load(AASIST_WEIGHTS, map_location=device)
    if 'model_state_dict' in checkpoint_a: aasist_model.load_state_dict(checkpoint_a['model_state_dict'])
    else: aasist_model.load_state_dict(checkpoint_a)
    
    resnet_model = resnet18_simam(num_classes=2, dropout_rate=0.22)
    checkpoint_r = torch.load(RESNET_WEIGHTS, map_location=device)
    if 'model_state_dict' in checkpoint_r: resnet_model.load_state_dict(checkpoint_r['model_state_dict'])
    else: resnet_model.load_state_dict(checkpoint_r)
    
    fusion_head = CrossAttentionFuser()
    wrapper_model = EndToEndEnsemble(aasist_model, resnet_model, fusion_head).to(device)

    train_dataset = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    val_dataset = ASVspoofDataset(PREPROCESSED_DEV_DIR, PROTOCOL_DEV)
    sampler = create_weighted_sampler(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)

    sensor = UncertaintySensor(mc_passes=5)
    controller = PDController()
    selector = DegradationSelector()
    actuator = DegradationActuator(device)

    optimizer = optim.AdamW([
        {'params': wrapper_model.aasist.parameters(), 'lr': 1e-5},
        {'params': wrapper_model.resnet.parameters(), 'lr': 1e-5},
        {'params': wrapper_model.fusion_head.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    scaler = torch.amp.GradScaler('cuda')

    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80).to(device)
    amp_to_db = T.AmplitudeToDB(stype='power', top_db=80).to(device)

    best_composite = float('inf')
    epochs_no_improve = 0

    for epoch in range(TOTAL_EPOCHS):
        wrapper_model.train()
        train_loss = 0.0
        epoch_gaps = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [Train]")
        for waveforms, labels in pbar:
            waveforms = waveforms.squeeze(1).to(device)
            labels = labels.to(device)

            with torch.no_grad():
                z_u, _ = sensor.measure(wrapper_model.aasist, waveforms)
                
            selections = selector.select(z_u)
            alpha = controller.alpha
            aug_waveforms = actuator.apply(waveforms, labels, selections, alpha)

            optimizer.zero_grad()
            combined = torch.cat([waveforms, aug_waveforms], dim=0)
            combined_lbl = torch.cat([labels, labels], dim=0)

            mel_combined = mel_transform(combined)
            mel_db_combined = amp_to_db(mel_combined).unsqueeze(1)

            out_meta, out_a, out_r = wrapper_model(combined, mel_db_combined, return_base_outs=True)
            
            B = waveforms.size(0)
            out_meta_clean = out_meta[:B]
            out_meta_deg = out_meta[B:]
            
            loss_clean = criterion(out_meta_clean, combined_lbl[:B])
            loss_deg = criterion(out_meta_deg, combined_lbl[:B])
            loss_cons = F.mse_loss(F.softmax(out_meta_clean, dim=1), F.softmax(out_meta_deg, dim=1))
            
            loss_aux = criterion(out_a[:B], combined_lbl[:B]) + criterion(out_r[:B], combined_lbl[:B])

            loss_total = (0.40 * loss_clean) + (0.40 * loss_deg) + (0.10 * loss_cons) + (0.10 * loss_aux)

            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(wrapper_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                prob_c = torch.softmax(out_meta_clean, dim=1)
                prob_a = torch.softmax(out_meta_deg, dim=1)
                conf_c = prob_c.gather(1, combined_lbl[:B].view(-1, 1)).mean().item() * 100
                conf_a = prob_a.gather(1, combined_lbl[:B].view(-1, 1)).mean().item() * 100
            
            epoch_gaps.append(conf_c - conf_a)
            train_loss += loss_total.item()
            pbar.set_postfix({"loss": f"{loss_total.item():.4f}", "a": f"{alpha:.3f}"})
            
        mean_gap = float(np.mean(epoch_gaps))
        new_alpha = controller.update(mean_gap)
        avg_train_loss = train_loss / len(train_loader)
        
        wrapper_model.eval()
        val_loss = 0.0
        lc, pc, la, pa = [], [], [], []
        
        with torch.no_grad():
            for wv, lv in tqdm(val_loader, desc=f"Epoch {epoch+1} [Valid]", leave=False):
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
        
        print(f"End of Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | EER_c: {eer_clean:.4f}% | EER_a: {eer_aug:.4f}% | Score: {composite:.4f}%")
        
        if composite < best_composite:
            best_composite = composite
            epochs_no_improve = 0
            torch.save(wrapper_model.state_dict(), OUTPUT_WEIGHTS)
            print(f"  -> Score Improved. Saved to {OUTPUT_WEIGHTS}")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement ({epochs_no_improve}/20)")
            
        if epochs_no_improve >= 20:
            print("\nEarly stopping triggered. Model has converged.")
            break

if __name__ == "__main__":
    main()