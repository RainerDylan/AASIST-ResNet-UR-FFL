import sys
import os
import random
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..')) if 'ensemble' in CURRENT_DIR else CURRENT_DIR
sys.path.append(ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(RESULTS_DIR, exist_ok=True)

from src.data.dataset import ASVspoofDataset
from src.models.aasist import AASIST
from src.models.resnet_simam import resnet18_simam

BASE_DATASET_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset"
PREPROCESSED_LA_DIR = os.path.join(BASE_DATASET_DIR, "preprocessed_la")
PREPROCESSED_DF_DIR = os.path.join(BASE_DATASET_DIR, "preprocessed_df")

RAW_LA_EVAL_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_eval\flac"
PROTOCOL_LA_EVAL = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt"

RAW_DF_EVAL_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2021\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\flac"
PROTOCOL_DF_EVAL = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2021\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\trial_metadata.txt" 

class MetaLearner(nn.Module):
    def __init__(self, input_dim=616, hidden_dim=256, num_classes=2, dropout=0.3):
        super(MetaLearner, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

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

        def hook_a(module, inp, out): self.emb_a[0] = inp[0]
        def hook_r(module, inp, out): self.emb_r[0] = inp[0]

        self.h_a = self.aasist.fc.register_forward_hook(hook_a)
        self.h_r = self.resnet.fc.register_forward_hook(hook_r)

    def forward(self, waveform, mel_db, return_base_outs=False):
        # AASIST runs in safe FP32
        out_a = self.aasist(waveform)
        
        # Only ResNet and MetaLearner run in FP16
        with torch.amp.autocast('cuda'):
            out_r = self.resnet(mel_db)
            out_meta = self.meta_learner(self.emb_a[0], self.emb_r[0])
            
        if return_base_outs:
            return out_meta, out_a, out_r
        return out_meta

def compute_min_dcf(fpr, fnr, p_target=0.05, c_miss=1.0, c_fa=1.0):
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    min_dcf = np.min(dcf)
    default_dcf = min(c_miss * p_target, c_fa * (1.0 - p_target))
    min_dcf_norm = min_dcf / default_dcf
    return min_dcf_norm

def create_shuffled_protocol(original_protocol, target_protocol):
    bonafide_lines = []
    spoof_lines = []
    with open(original_protocol, 'r') as f:
        for line in f:
            line_str = line.strip()
            if not line_str: 
                continue
            parts = line_str.split()
            if len(parts) < 2: 
                continue
            speaker = parts[0]
            fname = parts[1]
            is_bonafide = 'bonafide' in line_str.lower()
            label_str = 'bonafide' if is_bonafide else 'spoof'
            std_line = f"{speaker} {fname} - - {label_str}"
            if is_bonafide: 
                bonafide_lines.append(std_line)
            else: 
                spoof_lines.append(std_line)
                
    random.seed(42)
    random.shuffle(bonafide_lines)
    random.shuffle(spoof_lines)
    
    interleaved = []
    max_len = max(len(bonafide_lines), len(spoof_lines))
    for i in range(max_len):
        if i < len(bonafide_lines): interleaved.append(bonafide_lines[i])
        if i < len(spoof_lines): interleaved.append(spoof_lines[i])
            
    with open(target_protocol, 'w') as f:
        for line in interleaved: f.write(line + '\n')

def apply_vad_and_norm(waveform):
    original_waveform = waveform.clone()
    try:
        waveform = torchaudio.functional.vad(waveform, sample_rate=16000)
        if waveform.shape[-1] <= 1: waveform = original_waveform
    except Exception:
        waveform = original_waveform
    mean = waveform.mean()
    std = waveform.std()
    waveform = (waveform - mean) / (std + 1e-8)
    return waveform

def apply_preemphasis(waveform):
    alpha = 0.97
    return torch.cat([waveform[:, :1], waveform[:, 1:] - alpha * waveform[:, :-1]], dim=1)

def preprocess_evaluation(eval_dir, protocol_file, target_dir, target_total=7000, target_length=64600):
    with open(protocol_file, 'r') as f:
        lines = f.readlines()
                
    valid_lines = []
    success_bonafide = 0
    success_spoof = 0
    pbar = tqdm(total=target_total, desc="Preprocessing Eval Dataset")
    
    for line in lines:
        if (success_bonafide + success_spoof) >= target_total: break
        parts = line.strip().split()
        if len(parts) < 5: continue
            
        fname = parts[1]
        is_bonafide = (parts[4] == 'bonafide')
        file_path = os.path.join(eval_dir, f"{fname}.flac")
        if not os.path.exists(file_path): continue
            
        try:
            waveform, _ = torchaudio.load(file_path)
        except Exception:
            continue
            
        waveform = apply_vad_and_norm(waveform)
        seq_len = waveform.shape[-1]
        
        if seq_len > target_length:
            start = (seq_len - target_length) // 2
            waveform = waveform[:, start:start + target_length]
        elif seq_len < target_length:
            if seq_len <= 1:
                waveform = F.pad(waveform, (0, target_length - seq_len), mode='constant', value=0)
            else:
                while waveform.shape[-1] < target_length:
                    current_len = waveform.shape[-1]
                    pad_amount = min(target_length - current_len, current_len - 1)
                    waveform = F.pad(waveform, (0, pad_amount), mode='reflect')
                    
        pre_emphasized = apply_preemphasis(waveform)
        torch.save(pre_emphasized, os.path.join(target_dir, f"{fname}.pt"))
        valid_lines.append(line)
        
        if is_bonafide: success_bonafide += 1
        else: success_spoof += 1
        pbar.update(1)
        
    pbar.close()
    with open(protocol_file, 'w') as f:
        for line in valid_lines: f.write(line)

def build_blank_ensemble(device):
    aasist_model = AASIST(stft_window=698, stft_hop=398, freq_bins=116, gat_layers=2, heads=5, head_dim=104, hidden_dim=455, dropout=0.33)
    resnet_model = resnet18_simam(num_classes=2, dropout_rate=0.22)
    meta_learner = MetaLearner(input_dim=616)
    
    wrapper = EndToEndEnsemble(aasist_model, resnet_model, meta_learner).to(device)
    return wrapper

def calculate_metrics(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
    fnr = 1 - tpr
    auc_score = auc(fpr, tpr)
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_idx] * 100
    min_dcf_norm = compute_min_dcf(fpr, fnr)
    return eer, auc_score, min_dcf_norm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n========================================")
    print("5-FOLD STABILITY EVALUATION SCRIPT")
    print("========================================")
    print("[1] Official ASVspoof 2019 LA Evaluation")
    print("[2] Official ASVspoof 2021 DF Evaluation")
    
    choice = input("Select mode (1 or 2): ").strip()
    
    if choice == '1':
        print("\n--- OFFICIAL ASVSPOOF 2019 LA EVALUATION ---")
        raw_dir = RAW_LA_EVAL_DIR
        orig_protocol = PROTOCOL_LA_EVAL
        dataset_name = "LA"
        target_dir = PREPROCESSED_LA_DIR
    elif choice == '2':
        print("\n--- OFFICIAL ASVSPOOF 2021 DF EVALUATION ---")
        raw_dir = RAW_DF_EVAL_DIR
        orig_protocol = PROTOCOL_DF_EVAL
        dataset_name = "DF"
        target_dir = PREPROCESSED_DF_DIR
    else:
        print("Invalid selection.")
        return
        
    subset_protocol = os.path.join(target_dir, "subset_protocol.txt")
    existing_files = glob.glob(os.path.join(target_dir, '*.pt'))
    
    if len(existing_files) >= 7000 and os.path.exists(subset_protocol):
        print(f"Found existing evaluation dataset for {dataset_name}.")
    else:
        print(f"Preprocessing evaluation dataset subset for {dataset_name}...")
        for f in existing_files: os.remove(f)
        os.makedirs(target_dir, exist_ok=True)
        create_shuffled_protocol(orig_protocol, subset_protocol)
        preprocess_evaluation(raw_dir, subset_protocol, target_dir, target_total=7000)
        
    eval_dataset = ASVspoofDataset(target_dir, subset_protocol)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    wrapper = build_blank_ensemble(device)
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80).to(device)
    amp_to_db = T.AmplitudeToDB(stype='power', top_db=80).to(device)
    
    fold_eers = []
    fold_min_dcfs = []
    fold_aucs = []

    print("\nExecuting Inference Engine strictly per fold...")
    
    for fold in range(1, 6):
        weight_path = os.path.join(MODELS_DIR, f"meta_ensemble_urffl_fold_{fold}.pth")
        if not os.path.exists(weight_path):
            print(f"Missing weights for fold {fold}. Aborting.")
            return
            
        wrapper.load_state_dict(torch.load(weight_path, map_location=device))
        wrapper.eval()
        
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for waveforms, labels in tqdm(eval_loader, desc=f"Evaluating FOLD {fold}", leave=False):
                waveforms = waveforms.squeeze(1).to(device)
                
                mel = mel_transform(waveforms)
                mel_db = amp_to_db(mel).unsqueeze(1)
                
                # Removing outer autocast to prevent AASIST FP16 NaN explosion
                out_meta = wrapper(waveforms, mel_db)
                
                probs = torch.softmax(out_meta, dim=1)[:, 1]
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                
        eer, auc_score, min_dcf = calculate_metrics(all_labels, all_probs)
        fold_eers.append(eer)
        fold_aucs.append(auc_score)
        fold_min_dcfs.append(min_dcf)
        
        print(f"FOLD {fold} -> EER: {eer:.4f}% | AUC: {auc_score:.4f} | minDCF: {min_dcf:.4f}")

    print("\n========================================")
    print(f"FINAL STATISTICAL STABILITY REPORT ({dataset_name})")
    print("========================================")
    print(f"Mean Equal Error Rate (EER): {np.mean(fold_eers):.4f}% ± {np.std(fold_eers):.4f}%")
    print(f"Mean Area Under Curve (AUC): {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Mean Normalized minDCF:      {np.mean(fold_min_dcfs):.4f} ± {np.std(fold_min_dcfs):.4f}")
    print("========================================")
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(fold_eers, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.title(f'5-Fold Model Stability on Unseen {dataset_name} Evaluation Data')
    plt.ylabel('Equal Error Rate (%)')
    plt.xticks([1], ['End-to-End Ensemble Folds'])
    plt.grid(True, ls=':', alpha=0.6)
    
    graph_path = os.path.join(RESULTS_DIR, f"kfold_stability_boxplot_{dataset_name.lower()}.png")
    plt.savefig(graph_path, dpi=300)
    print(f"Stability boxplot saved to {graph_path}")

if __name__ == "__main__":
    main()