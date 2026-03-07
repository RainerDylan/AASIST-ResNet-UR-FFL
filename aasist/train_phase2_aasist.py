import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
import time
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.data.dataset import ASVspoofDataset 
from src.models.aasist import AASIST
from src.ur_ffl.sensor import UncertaintySensor
from src.ur_ffl.controller import PDController
from src.ur_ffl.selector import DegradationSelector
from src.ur_ffl.actuator import DegradationActuator

PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PREPROCESSED_DEV_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
PROTOCOL_DEV = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"
PHASE1_WEIGHTS = os.path.join(MODELS_DIR, "aasist_phase1_best.pth")

def create_weighted_sampler(dataset):
    labels = dataset.labels
    class_counts = torch.bincount(torch.tensor(labels))
    total_samples = len(labels)
    class_weights = total_samples / class_counts.float()
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=total_samples, replacement=True)
    return sampler

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[idx] * 100

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initiating Phase 2 (UR-FFL) Training on {device}")
    
    model = AASIST(
        stft_window=698, stft_hop=398, freq_bins=116, gat_layers=2,
        heads=5, head_dim=104, hidden_dim=455, dropout=0.3311465671378094
    ).to(device)
    
    print(f"Loading Phase 1 Baseline Weights from {PHASE1_WEIGHTS}...")
    model.load_state_dict(torch.load(PHASE1_WEIGHTS, map_location=device))
    
    print("Applying Layer-Wise Learning Rate Decay (LLRD)...")
    frontend_modules = ['encoder', 'sinc']
    middle_modules = ['GAT_layer1', 'gat1', 'node_embedding']
    
    frontend_params = []
    middle_params = []
    backend_params = []
    
    for name, param in model.named_parameters():
        param.requires_grad = True 
        if any(fm in name for fm in frontend_modules):
            frontend_params.append(param)
        elif any(mm in name for mm in middle_modules):
            middle_params.append(param)
        else:
            backend_params.append(param)
            
    print("Loading datasets to memory structure...")
    train_dataset = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    val_dataset = ASVspoofDataset(PREPROCESSED_DEV_DIR, PROTOCOL_DEV)
    sampler = create_weighted_sampler(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    sensor = UncertaintySensor(mc_passes=10)
    controller = PDController()
    selector = DegradationSelector()
    actuator = DegradationActuator(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW([
        {'params': frontend_params, 'lr': 1e-5}, 
        {'params': middle_params, 'lr': 5e-5},   
        {'params': backend_params, 'lr': 1e-4}   
    ], weight_decay=1e-4)
    
    total_epochs = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-7)
    
    best_eer = float('inf')
    patience = 8
    epochs_no_improve = 0
    
    history_train_loss = []
    history_val_loss = []
    history_val_acc = []
    history_val_eer = []
    history_severity = []
    
    total_training_time = 0.0

    frozen_modules = ['encoder', 'sinc', 'GAT_layer1', 'gat1', 'node_embedding']

    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        
        train_loss = 0.0
        epoch_severities = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
        
        # Initialize safe starting values for the controller
        prev_clean_std = 0.02
        prev_deg_std = 0.06
        
        for waveforms, labels in pbar:
            waveforms = waveforms.squeeze(1).to(device)
            labels = labels.to(device)
            
            z_u_clean, clean_std = sensor.measure(model, waveforms)
            selections = selector.select(z_u_clean)
            
            current_alpha = controller.compute_severity(prev_clean_std, prev_deg_std)
            epoch_severities.append(current_alpha)
            
            aug_waveforms = actuator.apply(waveforms, labels, selections, current_alpha)
            
            # Close the loop: Measure Degraded StdDev for the next batch
            _, deg_std = sensor.measure(model, aug_waveforms)
            prev_clean_std = clean_std
            prev_deg_std = deg_std
            
            model.train()
            
            # Maintain BatchNorm stability against leakage
            for name, module in model.named_modules():
                if any(fm in name for fm in frozen_modules):
                    module.eval()
                    
            optimizer.zero_grad()
            
            # Unified Batching guarantees clean gradient separation without corruption
            combined_waveforms = torch.cat([waveforms, aug_waveforms], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)
            
            outputs = model(combined_waveforms)
            
            batch_size = waveforms.size(0)
            outputs_clean = outputs[:batch_size]
            outputs_deg = outputs[batch_size:]
            
            loss_clean = criterion(outputs_clean, labels)
            loss_deg = criterion(outputs_deg, labels)
            
            loss_total = 0.5 * loss_clean + 0.5 * loss_deg
            loss_total.backward()
            optimizer.step()
            
            train_loss += loss_total.item()
            pbar.set_postfix({"loss": f"{loss_total.item():.4f}", "alpha": f"{current_alpha:.2f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        avg_severity = sum(epoch_severities) / len(epoch_severities)
        
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
                
                # Model evaluation is correctly tested on degraded audio to ensure DF generalization
                val_z_scores = torch.zeros(waveforms.size(0)).to(device) 
                val_selections = selector.select(val_z_scores)
                val_aug_waveforms = actuator.apply(waveforms, labels, val_selections, alpha=0.50)
                
                outputs = model(val_aug_waveforms)
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
        history_severity.append(avg_severity)
        
        epoch_duration = time.time() - epoch_start_time
        total_training_time += epoch_duration
        avg_epoch_time = total_training_time / (epoch + 1)
        eta_seconds = int(avg_epoch_time * (total_epochs - (epoch + 1)))
        eta_string = str(datetime.timedelta(seconds=eta_seconds))
        
        current_lr = optimizer.param_groups[2]['lr']
        print(f"End of Epoch {epoch+1} | Backend LR: {current_lr:.6f} | Avg Alpha: {avg_severity:.2f} | Train Loss: {avg_train_loss:.4f} | Degraded Val Loss: {avg_val_loss:.4f} | Degraded Val Acc: {val_accuracy:.2f}% | Degraded Val EER: {val_eer:.4f}%")
        print(f"  -> Epoch Time: {epoch_duration:.1f}s | Estimated Time Left: {eta_string}")
        
        if val_eer < best_eer:
            best_eer = val_eer
            epochs_no_improve = 0
            save_path = os.path.join(MODELS_DIR, "aasist_phase2_urffl_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> UR-FFL Robust EER Improved! Saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement. Early stopping counter: {epochs_no_improve}/{patience}")
            
        if epochs_no_improve >= patience:
            print("\nEarly stopping triggered. Phase 2 has converged.")
            break

    print("Phase 2 Training complete. Generating learning curve graphs...")
    epochs_range = range(1, len(history_train_loss) + 1)
    
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(epochs_range, history_train_loss, label='Train Loss', color='blue')
    plt.plot(epochs_range, history_val_loss, label='Degraded Val Loss', color='red', linestyle='dashed')
    plt.title('Cross-Entropy Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.subplot(1, 4, 2)
    plt.plot(epochs_range, history_val_acc, label='Degraded Val Accuracy (%)', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.subplot(1, 4, 3)
    plt.plot(epochs_range, history_val_eer, label='Degraded Val EER (%)', color='purple')
    plt.title('Equal Error Rate (EER)')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(1, 4, 4)
    plt.plot(epochs_range, history_severity, label='Avg Augmentation Severity (alpha)', color='orange')
    plt.title('UR-FFL PD Controller Actions')
    plt.xlabel('Epochs')
    plt.ylabel('Alpha Level (0.1 to 0.90)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    graph_path = os.path.join(RESULTS_DIR, "aasist_phase2_metrics.png")
    plt.savefig(graph_path, dpi=300)
    print(f"Graph saved to {graph_path}")

if __name__ == "__main__":
    main()