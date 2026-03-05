import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from src.data.dataset import ASVspoofDataset
from src.config import DATASET_TRAIN_FLAC, PROTOCOL_FILE_TRAIN
from src.models.rawnet2 import RawNet2

def train_eval_trial(model, input_length, lr_p1, lr_p2, weight_decay, target_batch_size, mc_passes, device, trial):
    dataset = ASVspoofDataset(DATASET_TRAIN_FLAC, PROTOCOL_FILE_TRAIN)
    
    total_files = len(dataset)
    random_indices = torch.randperm(total_files).tolist()[:500]
    
    train_subset = Subset(dataset, random_indices[:400])
    val_subset = Subset(dataset, random_indices[400:])
    
    micro_batch = 4
    accumulation_steps = max(1, target_batch_size // micro_batch)
    
    train_loader = DataLoader(train_subset, batch_size=micro_batch, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=micro_batch, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_p1, weight_decay=weight_decay)
    model.to(device)
    
    print(f"\n[Trial {trial.number}] RawNet2 (Batch: {target_batch_size}, LR1: {lr_p1:.5f}, Length: {input_length})")
    
    for epoch in range(3):
        if epoch == 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_p2

        model.train()
        pbar_train = tqdm(train_loader, desc=f"  Epoch {epoch+1}/3 [Train]", leave=False)
        optimizer.zero_grad()
        
        for i, (waveforms, labels) in enumerate(pbar_train):
            seq_len = waveforms.shape[-1]
            if seq_len > input_length:
                waveforms = waveforms[..., :input_length]
            elif seq_len < input_length:
                waveforms = F.pad(waveforms, (0, input_length - seq_len), mode='constant')

            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                
        model.train() 
        val_loss = 0.0
        pbar_val = tqdm(val_loader, desc=f"  Epoch {epoch+1}/3 [Val MC={mc_passes}]", leave=False)
        
        with torch.no_grad():
            for waveforms, labels in pbar_val:
                seq_len = waveforms.shape[-1]
                if seq_len > input_length:
                    waveforms = waveforms[..., :input_length]
                elif seq_len < input_length:
                    waveforms = F.pad(waveforms, (0, input_length - seq_len), mode='constant')

                waveforms, labels = waveforms.to(device), labels.to(device)
                mc_outputs = 0
                for _ in range(mc_passes):
                    mc_outputs += model(waveforms)
                avg_outputs = mc_outputs / mc_passes
                loss = criterion(avg_outputs, labels)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        print(f"  -> Epoch {epoch+1} Val Loss: {val_loss:.4f}")
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            print(f"  -> Trial {trial.number} pruned.")
            raise optuna.exceptions.TrialPruned()
            
    return val_loss

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_length = trial.suggest_int("input_length", 16000, 160000)
    sinc_filters = trial.suggest_int("sinc_filters", 40, 160)
    sinc_kernel = trial.suggest_int("sinc_kernel", 101, 401, step=2)
    res_blocks = trial.suggest_int("res_blocks", 3, 5)
    ch_scale = trial.suggest_float("channels_scale", 0.5, 2.0)
    conv_k = trial.suggest_categorical("conv_kernel_size", [3, 5, 7])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_int("batch_size", 16, 64)
    lr_phase1 = trial.suggest_float("lr_phase1", 1e-4, 5e-3, log=True)
    lr_phase2 = trial.suggest_float("lr_phase2", 1e-5, 5e-4, log=True)
    wd = trial.suggest_float("weight_decay", 0.0, 0.1)
    mc_passes = trial.suggest_int("mc_passes", 10, 100)
    
    model = RawNet2(sinc_filters=sinc_filters, sinc_kernel=sinc_kernel, 
                    res_blocks=res_blocks, channel_scale=ch_scale, 
                    conv_kernel=conv_k, dropout=dropout)

    return train_eval_trial(model, input_length, lr_phase1, lr_phase2, wd, batch_size, mc_passes, device, trial)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=30)
    print("\nBest RawNet2 Config:")
    for k, v in study.best_trial.params.items(): print(f"  {k}: {v}")