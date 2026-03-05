import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import os
import numpy as np
import time
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.data.dataset import ASVspoofDataset 
from src.models.aasist import AASIST

# --- YOUR ACTUAL LOCAL PATHS ---
PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PREPROCESSED_DEV_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
PROTOCOL_DEV = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"

def create_weighted_sampler(dataset):
    labels = dataset.labels
    class_counts = torch.bincount(torch.tensor(labels))
    total_samples = len(labels)
    
    class_weights = total_samples / class_counts.float()
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=total_samples, 
        replacement=True
    )
    return sampler

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[idx]
    return eer * 100

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initiating Phase 1 AASIST Training on {device}")
    
    model = AASIST(
        stft_window=698,
        stft_hop=398,
        freq_bins=116,
        gat_layers=2,
        heads=5,
        head_dim=104,
        hidden_dim=455,
        dropout=0.3311465671378094
    ).to(device)
    
    print("Loading datasets to memory structure...")
    train_dataset = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    val_dataset = ASVspoofDataset(PREPROCESSED_DEV_DIR, PROTOCOL_DEV)
    
    sampler = create_weighted_sampler(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=21, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=21, shuffle=False, num_workers=4)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.0003217353005291878, 
        weight_decay=0.01
    )
    
    # Strictly 80 Epochs for Phase 1 Preparation
    total_epochs = 80
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    
    best_eer = float('inf')
    
    history_train_loss = []
    history_val_loss = []
    history_val_acc = []
    history_val_eer = []
    
    total_training_time = 0.0

    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
        
        for waveforms, labels in pbar:
            waveforms = waveforms.squeeze(1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Valid]", leave=False)
            for waveforms, labels in pbar_val:
                waveforms = waveforms.squeeze(1).to(device)
                labels = labels.to(device)
                
                outputs = model(waveforms)
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
        
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)
        history_val_acc.append(val_accuracy)
        history_val_eer.append(val_eer)
        
        # Calculate ETA
        epoch_duration = time.time() - epoch_start_time
        total_training_time += epoch_duration
        avg_epoch_time = total_training_time / (epoch + 1)
        epochs_remaining = total_epochs - (epoch + 1)
        eta_seconds = int(avg_epoch_time * epochs_remaining)
        eta_string = str(datetime.timedelta(seconds=eta_seconds))
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"End of Epoch {epoch+1} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | Val EER: {val_eer:.4f}%")
        print(f"  -> Epoch Time: {epoch_duration:.1f}s | Estimated Time Left: {eta_string}")
        
        if val_eer < best_eer:
            best_eer = val_eer
            torch.save(model.state_dict(), "aasist_phase1_best.pth")
            print("  -> EER Improved! New best model saved.")

    # --- GRAPH GENERATION ---
    print("Phase 1 Training complete. Generating learning curve graphs...")
    epochs_range = range(1, total_epochs + 1)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history_train_loss, label='Train Loss', color='blue')
    plt.plot(epochs_range, history_val_loss, label='Val Loss', color='red', linestyle='dashed')
    plt.title('Cross-Entropy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history_val_acc, label='Val Accuracy (%)', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history_val_eer, label='Val EER (%)', color='purple')
    plt.title('Equal Error Rate (EER)')
    plt.xlabel('Epochs')
    plt.ylabel('EER (%)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("aasist_phase1_metrics.png", dpi=300)
    print("Graph saved as 'aasist_phase1_metrics.png'.")

if __name__ == "__main__":
    main()