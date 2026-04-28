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

PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PREPROCESSED_DEV_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
PROTOCOL_DEV = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"

# Weights handling
AASIST_WEIGHTS = os.path.join(MODELS_DIR, "aasist_baseline_best.pth")
RESNET_WEIGHTS = os.path.join(MODELS_DIR, "resnet_baseline_best.pth")
OUTPUT_WEIGHTS = os.path.join(MODELS_DIR, "crossattention_ensemble_baseline_best.pth")

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
        
        # The [CLS] Token
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

        # Stack into sequence: [CLS, AASIST, ResNet]
        x = torch.cat((cls_tokens, feat_a, feat_r), dim=1) 
        
        # Cross-Attention
        x = self.transformer(x)
        
        # Extract updated CLS token
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
    print(f"Initiating Warm-Start Baseline Cross-Attention Ensemble on {device}")

    # Load Pre-trained Baseline Base Models
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

    # Differential Learning Rates Setup
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

    train_dataset = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    val_dataset = ASVspoofDataset(PREPROCESSED_DEV_DIR, PROTOCOL_DEV)
    sampler = create_weighted_sampler(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)

    best_eer = float('inf')
    epochs_no_improve = 0

    for epoch in range(TOTAL_EPOCHS):
        wrapper_model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [Train]")
        for waveforms, labels in pbar:
            waveforms = waveforms.squeeze(1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            mel = mel_transform(waveforms)
            mel_db = amp_to_db(mel).unsqueeze(1)
            
            out_meta, out_a, out_r = wrapper_model(waveforms, mel_db, return_base_outs=True)
            
            loss_meta = criterion(out_meta, labels)
            loss_aux = criterion(out_a, labels) + criterion(out_r, labels)
            loss_total = loss_meta + (0.10 * loss_aux)

            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(wrapper_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss_total.item()
            pbar.set_postfix({"loss": f"{loss_total.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        wrapper_model.eval()
        val_loss = 0.0
        lc = []
        pc = []
        
        with torch.no_grad():
            for wv, lv in tqdm(val_loader, desc=f"Epoch {epoch+1} [Valid]", leave=False):
                wv = wv.squeeze(1).to(device)
                lv = lv.to(device)
                
                mel = mel_transform(wv)
                mel_db = amp_to_db(mel).unsqueeze(1)
                
                out_meta = wrapper_model(wv, mel_db)
                val_loss += criterion(out_meta, lv).item()
                
                lc.extend(lv.cpu().numpy())
                pc.extend(torch.softmax(out_meta, dim=1)[:, 1].cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_eer = compute_eer(lc, pc)
        scheduler.step()
        
        print(f"End of Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val EER: {val_eer:.4f}%")
        
        if val_eer < best_eer:
            best_eer = val_eer
            epochs_no_improve = 0
            torch.save(wrapper_model.state_dict(), OUTPUT_WEIGHTS)
            print(f"  -> EER Improved. Saved to {OUTPUT_WEIGHTS}")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement ({epochs_no_improve}/20)")
            
        if epochs_no_improve >= 20:
            print("\nEarly stopping triggered. Model has converged.")
            break

if __name__ == "__main__":
    main()