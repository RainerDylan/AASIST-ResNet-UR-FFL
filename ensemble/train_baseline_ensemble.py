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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Resolve root directory
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

# --- YOUR ACTUAL LOCAL PATHS ---
PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PREPROCESSED_DEV_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
PROTOCOL_DEV = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"

AASIST_WEIGHTS = os.path.join(MODELS_DIR, "aasist_baseline_best.pth")
RESNET_WEIGHTS = os.path.join(MODELS_DIR, "resnet_baseline_best.pth")
OUTPUT_WEIGHTS = os.path.join(MODELS_DIR, "meta_ensemble_best.pth")

# ═══════════════════════════════════════════════════════════════════════════════
# META-LEARNER ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
class MetaLearner(nn.Module):
    """
    A Multi-Layer Perceptron that acts as the meta-learner.
    Takes the concatenated 104-dim (AASIST) and 512-dim (ResNet) embeddings.
    """
    def __init__(self, input_dim=616, hidden_dim=256, num_classes=2, dropout=0.3):
        super(MetaLearner, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, emb_aasist, emb_resnet):
        # Concatenate orthogonal embeddings
        x = torch.cat([emb_aasist, emb_resnet], dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS & METRICS
# ═══════════════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initiating Meta-Learning Ensemble Training on {device}")

    # 1. Load AASIST Base Model (Frozen)
    print("Loading Baseline AASIST...")
    aasist_model = AASIST(
        stft_window=698, stft_hop=398, freq_bins=116, gat_layers=2, 
        heads=5, head_dim=104, hidden_dim=455, dropout=0.3311465671378094
    ).to(device)
    aasist_model.load_state_dict(torch.load(AASIST_WEIGHTS, map_location=device))
    aasist_model.eval()
    for param in aasist_model.parameters(): param.requires_grad = False

    # 2. Load ResNet-SimAM Base Model (Frozen)
    print("Loading Baseline ResNet-SimAM...")
    resnet_model = resnet18_simam(num_classes=2, dropout_rate=0.22489397436884667).to(device)
    resnet_model.load_state_dict(torch.load(RESNET_WEIGHTS, map_location=device))
    resnet_model.eval()
    for param in resnet_model.parameters(): param.requires_grad = False

    # 3. Setup Forward Hooks to Extract Embeddings
    buffer_aasist = [None]
    buffer_resnet = [None]
    
    def aasist_hook(module, inp, out): buffer_aasist[0] = inp[0]
    def resnet_hook(module, inp, out): buffer_resnet[0] = inp[0]
    
    aasist_model.fc.register_forward_hook(aasist_hook)
    resnet_model.fc.register_forward_hook(resnet_hook)

    # 4. Initialize Meta-Learner
    print("Initializing Meta-Learner Network...")
    meta_learner = MetaLearner(input_dim=104+512, hidden_dim=256).to(device)

    # 5. Dataloaders
    train_dataset = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    val_dataset = ASVspoofDataset(PREPROCESSED_DEV_DIR, PROTOCOL_DEV)
    sampler = create_weighted_sampler(train_dataset)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 6. Training Configuration (Only updating Meta-Learner)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(meta_learner.parameters(), lr=1e-3, weight_decay=1e-4)
    
    total_epochs = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    
    # Feature extractors for ResNet
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80).to(device)
    amp_to_db = T.AmplitudeToDB(stype='power', top_db=80).to(device)

    best_eer = float('inf')
    patience = 15
    epochs_no_improve = 0
    history = dict(train_loss=[], val_loss=[], val_acc=[], val_eer=[])
    total_training_time = 0.0

    # 7. Training Loop
    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        
        meta_learner.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
        
        for waveforms, labels in pbar:
            waveforms = waveforms.squeeze(1).to(device)
            labels = labels.to(device)
            
            # Extract features without computing gradients for base models
            with torch.no_grad():
                # AASIST 1D Forward
                _ = aasist_model(waveforms)
                emb_a = buffer_aasist[0]
                
                # ResNet 2D Forward
                mel_spec = mel_transform(waveforms)
                features_img = amp_to_db(mel_spec).unsqueeze(1)
                _ = resnet_model(features_img)
                emb_r = buffer_resnet[0]

            # Meta-Learner Forward
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = meta_learner(emb_a, emb_r)
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 8. Validation Loop
        meta_learner.eval()
        val_loss = 0.0; correct = 0; total = 0
        all_labels = []; all_probs = []
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Valid]", leave=False)
            for waveforms, labels in pbar_val:
                waveforms = waveforms.squeeze(1).to(device)
                labels = labels.to(device)
                
                _ = aasist_model(waveforms)
                emb_a = buffer_aasist[0]
                
                mel_spec = mel_transform(waveforms)
                features_img = amp_to_db(mel_spec).unsqueeze(1)
                _ = resnet_model(features_img)
                emb_r = buffer_resnet[0]

                with torch.amp.autocast('cuda'):
                    outputs = meta_learner(emb_a, emb_r)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_eer = compute_eer(all_labels, all_probs)
        scheduler.step()
        
        history["train_loss"].append(avg_train_loss); history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_accuracy); history["val_eer"].append(val_eer)
        
        epoch_duration = time.time() - epoch_start_time
        total_training_time += epoch_duration
        eta_seconds = int((total_training_time / (epoch + 1)) * (total_epochs - (epoch + 1)))
        
        print(f"End of Epoch {epoch+1} | LR: {optimizer.param_groups[0]['lr']:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | Val EER: {val_eer:.4f}% | ETA: {str(datetime.timedelta(seconds=eta_seconds))}")
        
        # Checkpointing
        if val_eer < best_eer:
            best_eer = val_eer
            epochs_no_improve = 0
            torch.save(meta_learner.state_dict(), OUTPUT_WEIGHTS)
            print(f"  -> Meta-Learner EER Improved. Saved to {OUTPUT_WEIGHTS}")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement. Early stopping counter: {epochs_no_improve}/{patience}")
            
        if epochs_no_improve >= patience:
            print("\nEarly stopping triggered. Meta-Learner has converged.")
            break

    # 9. Diagnostics Plotting
    print("Training complete. Generating learning curve graphs...")
    epochs_range = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history["train_loss"], label='Train Loss', color='blue')
    plt.plot(epochs_range, history["val_loss"], label='Val Loss', color='red', linestyle='dashed')
    plt.title('Cross-Entropy Loss (Meta-Learner)')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, ls=':', alpha=0.6)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history["val_acc"], label='Val Accuracy (%)', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True, ls=':', alpha=0.6)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history["val_eer"], label='Val EER (%)', color='purple')
    plt.title('Equal Error Rate (EER)')
    plt.xlabel('Epochs'); plt.ylabel('EER (%)'); plt.legend(); plt.grid(True, ls=':', alpha=0.6)
    
    plt.tight_layout()
    graph_path = os.path.join(RESULTS_DIR, "meta_ensemble_metrics.png")
    plt.savefig(graph_path, dpi=300)
    print(f"Graph saved to {graph_path}")

if __name__ == "__main__":
    main()