import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random

from src.data.dataset import ASVspoofDataset
from src.models.aasist import AASIST

# --- YOUR ACTUAL LOCAL PATHS ---
PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"

def get_balanced_subsets(dataset, train_size=400, val_size=100):
    bonafide_idx = [i for i, label in enumerate(dataset.labels) if label == 1]
    spoof_idx = [i for i, label in enumerate(dataset.labels) if label == 0]
    
    random.shuffle(bonafide_idx)
    random.shuffle(spoof_idx)
    
    half_train = train_size // 2
    train_indices = bonafide_idx[:half_train] + spoof_idx[:half_train]
    
    half_val = val_size // 2
    val_indices = bonafide_idx[half_train:half_train+half_val] + spoof_idx[half_train:half_train+half_val]
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def train_eval_trial(model, lr_p1, lr_p2, target_batch_size, mc_passes, device, trial, train_subset, val_subset):
    micro_batch = 4
    accumulation_steps = max(1, target_batch_size // micro_batch)
    
    train_loader = DataLoader(train_subset, batch_size=micro_batch, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=micro_batch, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    nll_loss = nn.NLLLoss() 
    
    optimizer = optim.AdamW(model.parameters(), lr=lr_p1, weight_decay=0.01)
    model.to(device)
    
    print(f"\n[Trial {trial.number}] AASIST (Batch: {target_batch_size}, LR1: {lr_p1:.6f})")
    
    for epoch in range(3):
        if epoch == 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_p2

        # --- TRAINING PHASE ---
        model.train()
        pbar_train = tqdm(train_loader, desc=f"  Epoch {epoch+1}/3 [Train]", leave=False)
        optimizer.zero_grad()
        
        for i, (waveforms, labels) in enumerate(pbar_train):
            waveforms = waveforms.squeeze(1).to(device)
            labels = labels.to(device)
            
            outputs = model(waveforms)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()
            
            # THE FIX: Gradient Clipping to mathematically prevent the 0.693 explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                
        # --- VALIDATION PHASE ---
        model.eval() 
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train() 
                
        val_loss = 0.0
        pbar_val = tqdm(val_loader, desc=f"  Epoch {epoch+1}/3 [Val MC={mc_passes}]", leave=False)
        
        with torch.no_grad():
            for waveforms, labels in pbar_val:
                waveforms = waveforms.squeeze(1).to(device)
                labels = labels.to(device)
                
                mc_probs = 0
                for _ in range(mc_passes):
                    logits = model(waveforms)
                    mc_probs += torch.softmax(logits, dim=1)
                    
                avg_probs = mc_probs / mc_passes
                
                loss = nll_loss(torch.log(avg_probs + 1e-8), labels)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        print(f"  -> Epoch {epoch+1} Val Loss: {val_loss:.4f}")
            
        trial.report(val_loss, epoch)
        if trial.should_prune():
            print(f"  -> Trial {trial.number} pruned.")
            raise optuna.exceptions.TrialPruned()
            
    return val_loss

def objective(trial, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    stft_win = trial.suggest_int("stft_window", 256, 1024)
    stft_hop = trial.suggest_int("stft_hop", 64, 512)
    max_safe_bins = min(256, stft_win // 2)
    freq_bins = trial.suggest_int("freq_bins", 64, max_safe_bins)
    
    calculated_frames = (64600 // stft_hop) + 1
    trial.set_user_attr("temporal_frames", calculated_frames)
    
    gat_layers = trial.suggest_int("gat_layers", 2, 4)
    heads = trial.suggest_int("attention_heads", 2, 8)
    head_dim = trial.suggest_int("head_dim", 32, 128)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_int("batch_size", 16, 64)
    
    # THE FIX: Lowered the ceiling from 5e-3 to 1e-3 to ensure stable optimization
    lr_phase1 = trial.suggest_float("lr_phase1", 1e-5, 1e-3, log=True)
    lr_phase2 = trial.suggest_float("lr_phase2", 1e-6, 1e-4, log=True)
    mc_passes = trial.suggest_int("mc_passes", 10, 100)
    
    model = AASIST(stft_window=stft_win, stft_hop=stft_hop, freq_bins=freq_bins,
                   gat_layers=gat_layers, heads=heads, head_dim=head_dim, 
                   hidden_dim=hidden_dim, dropout=dropout)

    train_subset, val_subset = get_balanced_subsets(dataset)

    return train_eval_trial(model, lr_phase1, lr_phase2, batch_size, mc_passes, device, trial, train_subset, val_subset)

if __name__ == "__main__":
    print("Loading Preprocessed Dataset to memory for fast tuning...")
    full_dataset = ASVspoofDataset(PREPROCESSED_TRAIN_DIR, PROTOCOL_TRAIN)
    
    # Strictly implements the BOHB algorithm logic via TPESampler and HyperbandPruner
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
    study.optimize(lambda trial: objective(trial, full_dataset), n_trials=30)
    
    print("\nBest AASIST Config:")
    for k, v in study.best_trial.params.items(): print(f"  {k}: {v}")
    print(f"  temporal_frames: {study.best_trial.user_attrs['temporal_frames']}")