import sys
import os

# Dynamically route the path to the parent directory to allow src imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(ROOT_DIR)

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import ASVspoofDataset
from src.config import DATASET_TRAIN_FLAC, PROTOCOL_FILE_TRAIN
from src.models.resnet_simam import resnet18_simam

# Use explicit paths to the preprocessed tensors to avoid empty datasets
PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"

def train_eval_trial(model, lr, weight_decay, target_batch_size, device, trial):
    # 1. Setup Dataset
    dataset = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    total_files = len(dataset)
    
    if total_files == 0:
        raise RuntimeError(f"Dataset is empty. Verify that {PREPROCESSED_TRAIN_DIR} contains .pt files.")
        
    subset_size = min(1000, total_files)
    train_size = int(0.8 * subset_size)
    
    random_indices = torch.randperm(total_files).tolist()[:subset_size]
    train_subset = Subset(dataset, random_indices[:train_size])
    val_subset = Subset(dataset, random_indices[train_size:])
    
    micro_batch = 16 
    accumulation_steps = max(1, target_batch_size // micro_batch)
    
    train_loader = DataLoader(train_subset, batch_size=micro_batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=micro_batch, shuffle=False, num_workers=2)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda')
    
    # 2. Feature Extractor: Raw Audio -> Log-Mel-Spectrogram (Decibels)
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80).to(device)
    amp_to_db = T.AmplitudeToDB(stype='power', top_db=80).to(device) # CRITICAL: Stabilizes AMP gradients
    
    epochs = 10 
    best_val_loss = float('inf')
    
    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        train_pbar = tqdm(train_loader, desc=f"Trial {trial.number} | Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for i, (waveforms, labels) in enumerate(train_pbar):
            waveforms = waveforms.squeeze(1).to(device)
            labels = labels.to(device)
            
            # Transform to Log-Mel Spectrogram and add Channel Dimension
            mel_spec = mel_transform(waveforms)
            features = amp_to_db(mel_spec).unsqueeze(1)
            
            with torch.amp.autocast('cuda'):
                outputs = model(features)
                loss = criterion(outputs, labels) / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})
            
        # 4. Validation Loop
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Trial {trial.number} | Epoch {epoch+1}/{epochs} [Valid]", leave=False)
        
        with torch.no_grad():
            for waveforms, labels in val_pbar:
                waveforms = waveforms.squeeze(1).to(device)
                labels = labels.to(device)
                
                mel_spec = mel_transform(waveforms)
                features = amp_to_db(mel_spec).unsqueeze(1)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(features)
                    batch_loss = criterion(outputs, labels)
                    val_loss += batch_loss.item()
                    
        val_loss /= len(val_loader)
        best_val_loss = min(best_val_loss, val_loss)
        
        # 5. Report to Optuna for HyperBand Pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return best_val_loss

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Search Space Definition
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    model = resnet18_simam(num_classes=2, dropout_rate=dropout).to(device)
    
    return train_eval_trial(model, lr, wd, batch_size, device, trial)

if __name__ == "__main__":
    # Relaxed HyperBand Pruner (Gives trials at least 3 epochs to prove themselves)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=3, 
        max_resource=10, 
        reduction_factor=3
    )
    
    # TPE Sampler (Bayesian Optimization)
    sampler = optuna.samplers.TPESampler(seed=42)
    
    # NEW STUDY NAME: Avoids loading the corrupted all-pruned history
    study = optuna.create_study(
        study_name="resnet_optimization_v2",
        storage="sqlite:///resnet_tuning.db",
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
        sampler=sampler
    )
    
    print("Initiating Bayesian Optimization + Hyperband (Optuna)...")
    study.optimize(objective, n_trials=30, timeout=14400, show_progress_bar=True)
    
    print("\n=================================================")
    print("Optimization Completed!")
    try:
        trial = study.best_trial
        print(f"Best validation loss: {trial.value:.4f}")
        print("Best hyperparameters:")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")
    except ValueError:
        print("Error: All trials were pruned. Please check if the loss is exploding to NaN.")
    print("=================================================\n")