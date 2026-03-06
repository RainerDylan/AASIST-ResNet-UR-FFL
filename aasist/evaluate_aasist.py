import sys
import os
import random

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

import shutil
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, DetCurveDisplay, confusion_matrix, ConfusionMatrixDisplay

from src.data.dataset import ASVspoofDataset
from src.models.aasist import AASIST

# PATH CONFIGURATION
RAW_CUSTOM_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\Custom_Raw_Audio"
SHARED_PREPROCESSED_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\evaluation_preprocessed"

RAW_LA_EVAL_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_eval\flac"
PROTOCOL_LA_EVAL = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt"

RAW_DF_EVAL_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2021\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\flac"
PROTOCOL_DF_EVAL = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2021\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\trial_metadata.txt" 

SUBSET_PROTOCOL = os.path.join(SHARED_PREPROCESSED_DIR, "subset_protocol.txt")

def create_balanced_protocol(original_protocol, target_protocol, total_samples=10876):
    bonafide_lines = []
    spoof_lines = []
    
    with open(original_protocol, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue
                
            parts = stripped_line.split()
            
            if len(parts) > 6 and ('bonafide' in parts[5] or 'spoof' in parts[5]):
                speaker = parts[0]
                filename = parts[1]
                attack = parts[4]
                label = parts[5]
                standardized_line = f"{speaker} {filename} - {attack} {label}"
            else:
                standardized_line = stripped_line
                
            if 'bonafide' in standardized_line.split()[-1]:
                bonafide_lines.append(standardized_line)
            elif 'spoof' in standardized_line.split()[-1]:
                spoof_lines.append(standardized_line)
                
    target_per_class = total_samples // 2
    
    random.seed(42)
    n_bonafide = min(target_per_class, len(bonafide_lines))
    n_spoof = min(target_per_class, len(spoof_lines))
    
    selected_bonafide = random.sample(bonafide_lines, n_bonafide)
    selected_spoof = random.sample(spoof_lines, n_spoof)
    
    selected_all = selected_bonafide + selected_spoof
    random.shuffle(selected_all)
    
    with open(target_protocol, 'w') as f:
        for line in selected_all:
            f.write(line + '\n')

def apply_vad_and_norm(waveform):
    original_waveform = waveform.clone()
    try:
        waveform = torchaudio.functional.vad(waveform, sample_rate=16000)
        if waveform.shape[-1] <= 1:
            waveform = original_waveform
    except Exception:
        waveform = original_waveform
        
    mean = waveform.mean()
    std = waveform.std()
    waveform = (waveform - mean) / (std + 1e-8)
    return waveform

def apply_preemphasis(waveform):
    alpha = 0.97
    return torch.cat([waveform[:, :1], waveform[:, 1:] - alpha * waveform[:, :-1]], dim=1)

def preprocess_custom(target_length=64600):
    valid_extensions = ('*.wav', '*.flac', '*.mp3', '*.ogg', '*.m4a')
    audio_files = []
    for ext in valid_extensions:
        audio_files.extend(glob.glob(os.path.join(RAW_CUSTOM_DIR, ext)))
        
    if not audio_files:
        print(f"No custom audio files found in {RAW_CUSTOM_DIR}.")
        return {}

    file_to_chunks_map = {}
    skipped_files = 0
    for file_path in tqdm(audio_files, desc="Preprocessing Custom Audio"):
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception:
            skipped_files += 1
            continue
        
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        waveform = apply_vad_and_norm(waveform)
        seq_len = waveform.shape[-1]
        chunks = []
        
        if seq_len > (2 * target_length):
            step = target_length 
            for start in range(0, seq_len - target_length + 1, step):
                chunks.append(waveform[:, start:start + target_length])
        elif seq_len > target_length:
            start = (seq_len - target_length) // 2
            chunks.append(waveform[:, start:start + target_length])
        else:
            if seq_len <= 1:
                chunks.append(F.pad(waveform, (0, target_length - seq_len), mode='constant', value=0))
            else:
                while waveform.shape[-1] < target_length:
                    current_len = waveform.shape[-1]
                    pad_amount = min(target_length - current_len, current_len - 1)
                    waveform = F.pad(waveform, (0, pad_amount), mode='reflect')
                chunks.append(waveform)
                
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        saved_chunks = []
        for i, chunk in enumerate(chunks):
            pre_emphasized = apply_preemphasis(chunk)
            out_path = os.path.join(SHARED_PREPROCESSED_DIR, f"{base_name}_chunk{i}.pt")
            torch.save(pre_emphasized, out_path)
            saved_chunks.append(out_path)
            
        file_to_chunks_map[base_name] = saved_chunks
        
    if skipped_files > 0:
        print(f"\nNote: Skipped {skipped_files} corrupted custom audio files.")
        
    return file_to_chunks_map

def preprocess_evaluation(eval_dir, protocol_file, target_length=64600):
    with open(protocol_file, 'r') as f:
        lines = f.readlines()
                
    if not lines:
        print(f"No files found in protocol {protocol_file}.")
        return
        
    valid_lines = []
    skipped_files = 0
        
    for line in tqdm(lines, desc="Preprocessing Eval Dataset"):
        parts = line.strip().split()
        if not parts:
            continue
            
        fname = parts[1]
        file_path = os.path.join(eval_dir, f"{fname}.flac")
        
        if not os.path.exists(file_path):
            continue
            
        try:
            waveform, _ = torchaudio.load(file_path)
        except Exception:
            skipped_files += 1
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
        out_path = os.path.join(SHARED_PREPROCESSED_DIR, f"{fname}.pt")
        torch.save(pre_emphasized, out_path)
        valid_lines.append(line)
        
    with open(protocol_file, 'w') as f:
        for line in valid_lines:
            f.write(line)
            
    if skipped_files > 0:
        print(f"\nNote: Successfully skipped {skipped_files} corrupted FLAC files to preserve evaluation integrity.")

def initialize_model(device, weights_path):
    print(f"\nLoading trained weights from {weights_path}...")
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
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pth_files = glob.glob(os.path.join(MODELS_DIR, '*aasist*.pth'))
    if not pth_files:
        print(f"No AASIST .pth files found in {MODELS_DIR}.")
        return
        
    print("="*40)
    print("AVAILABLE AASIST MODELS")
    print("="*40)
    for i, file_path in enumerate(pth_files):
        print(f"[{i+1}] {os.path.basename(file_path)}")
        
    while True:
        try:
            model_choice = int(input("\nSelect the model to evaluate (number): ").strip())
            if 1 <= model_choice <= len(pth_files):
                selected_weights = pth_files[model_choice - 1]
                break
            else:
                print("Invalid selection. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print("\n" + "="*40)
    print("AASIST DYNAMIC EVALUATION SCRIPT")
    print("="*40)
    print("[1] Custom Audio Inference (Sliding Window)")
    print("[2] Official ASVspoof 2019 LA Evaluation (Balanced Subset)")
    print("[3] Official ASVspoof 2021 DF Evaluation (Balanced Subset)")
    
    choice = input("Select mode (1, 2, or 3): ").strip()
    
    if not os.path.exists(SHARED_PREPROCESSED_DIR):
        os.makedirs(SHARED_PREPROCESSED_DIR)

    if choice == '1':
        print("\n--- CUSTOM AUDIO MODE ---")
        print(f"Clearing {SHARED_PREPROCESSED_DIR} for new custom data...")
        for f in os.listdir(SHARED_PREPROCESSED_DIR):
            os.remove(os.path.join(SHARED_PREPROCESSED_DIR, f))
            
        file_map = preprocess_custom()
        if not file_map: return
        
        model = initialize_model(device, selected_weights)
        print("\n" + "="*40)
        print("CUSTOM INFERENCE RESULTS")
        print("="*40)
        
        with torch.no_grad():
            for base_name, chunk_paths in file_map.items():
                chunk_tensors = []
                for path in chunk_paths:
                    tensor = torch.load(path).to(device)
                    tensor = tensor.squeeze(0) if tensor.dim() == 3 else tensor
                    chunk_tensors.append(tensor)
                    
                batch = torch.stack(chunk_tensors)
                outputs = model(batch)
                probs = torch.softmax(outputs, dim=1)
                
                mean_probs = torch.mean(probs, dim=0)
                deepfake_prob = mean_probs[0].item() * 100
                bonafide_prob = mean_probs[1].item() * 100
                
                prediction = "DEEPFAKE" if deepfake_prob > bonafide_prob else "BONAFIDE"
                confidence = max(deepfake_prob, bonafide_prob)
                
                print(f"File: {base_name} | Segments: {len(chunk_paths)}")
                print(f"Result: {prediction} ({confidence:.2f}% certainty)")
                print(f"Breakdown -> Deepfake: {deepfake_prob:.2f}% | Bonafide: {bonafide_prob:.2f}%\n")

    elif choice in ['2', '3']:
        if choice == '2':
            print("\n--- OFFICIAL ASVSPOOF 2019 LA EVALUATION ---")
            raw_dir = RAW_LA_EVAL_DIR
            orig_protocol = PROTOCOL_LA_EVAL
            dataset_name = "LA"
        else:
            print("\n--- OFFICIAL ASVSPOOF 2021 DF EVALUATION ---")
            raw_dir = RAW_DF_EVAL_DIR
            orig_protocol = PROTOCOL_DF_EVAL
            dataset_name = "DF"
            
        existing_files = glob.glob(os.path.join(SHARED_PREPROCESSED_DIR, '*.pt'))
        
        if len(existing_files) > 1000 and os.path.exists(SUBSET_PROTOCOL):
            print(f"Found {len(existing_files)} preprocessed files in the folder.")
            skip = input("Do you want to skip preprocessing and evaluate immediately? (y/n): ").strip().lower()
            if skip != 'y':
                for f in existing_files: os.remove(f)
                create_balanced_protocol(orig_protocol, SUBSET_PROTOCOL)
                preprocess_evaluation(raw_dir, SUBSET_PROTOCOL)
        else:
            print("Clearing folder and preprocessing evaluation dataset subset...")
            for f in existing_files: os.remove(f)
            create_balanced_protocol(orig_protocol, SUBSET_PROTOCOL)
            preprocess_evaluation(raw_dir, SUBSET_PROTOCOL)
            
        model = initialize_model(device, selected_weights)
        print("Loading official protocol and preprocessed tensors...")
        
        eval_dataset = ASVspoofDataset(SHARED_PREPROCESSED_DIR, SUBSET_PROTOCOL)
        eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=4)
        
        all_labels = []
        all_probs = []
        all_preds = []
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for waveforms, labels in tqdm(eval_loader, desc="Evaluating Benchmarks"):
                waveforms = waveforms.squeeze(1).to(device)
                labels = labels.to(device)
                outputs = model(waveforms)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                _, predicted_classes = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted_classes == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(predicted_classes.cpu().numpy())
                
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        
        accuracy = (correct_predictions / total_samples) * 100
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs, pos_label=1)
        fnr = 1 - tpr
        auc_score = auc(fpr, tpr)
        
        eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
        eer = fpr[eer_idx] * 100
        
        print("\n" + "="*40)
        print("FINAL EVALUATION METRICS")
        print("="*40)
        print(f"Average Accuracy: {accuracy:.4f}%")
        print(f"Equal Error Rate (EER): {eer:.4f}%")
        print(f"Area Under the Curve (AUC): {auc_score:.4f}")
        print("="*40)
        
        # 1. Generate ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FAR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC Curve ({dataset_name} Evaluation)')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=':', alpha=0.6)
        roc_path = os.path.join(RESULTS_DIR, f"eval_roc_curve_{dataset_name.lower()}.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Generate DET Curve
        fig, ax = plt.subplots(figsize=(8, 6))
        display = DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name='AASIST Model')
        display.plot(ax=ax)
        plt.title(f'DET Curve ({dataset_name} Evaluation)')
        plt.grid(True, linestyle=':', alpha=0.6)
        det_path = os.path.join(RESULTS_DIR, f"eval_det_curve_{dataset_name.lower()}.png")
        plt.savefig(det_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Generate Score Distribution Histogram
        plt.figure(figsize=(8, 6))
        deepfake_scores = all_probs[all_labels == 0]
        bonafide_scores = all_probs[all_labels == 1]
        
        plt.hist(bonafide_scores, bins=50, alpha=0.6, color='#2ca02c', label='Bonafide (True Class)', edgecolor='white', linewidth=0.5)
        plt.hist(deepfake_scores, bins=50, alpha=0.6, color='#d62728', label='Deepfake (Spoof Class)', edgecolor='white', linewidth=0.5)
        plt.title(f'Model Confidence Score Distribution ({dataset_name} Evaluation)', fontsize=12, pad=15)
        plt.xlabel('Predicted Probability of being Bonafide', fontsize=10)
        plt.ylabel('Number of Audio Samples', fontsize=10)
        plt.legend(loc="upper center")
        plt.grid(True, linestyle=':', alpha=0.6)
        dist_path = os.path.join(RESULTS_DIR, f"eval_score_dist_{dataset_name.lower()}.png")
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Generate Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(7, 6))
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Deepfake', 'Bonafide'])
        cm_display.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
        plt.title(f'Confusion Matrix ({dataset_name} Evaluation)', fontsize=12, pad=15)
        plt.xlabel('Predicted Classification', fontsize=10)
        plt.ylabel('True Ground Label', fontsize=10)
        cm_path = os.path.join(RESULTS_DIR, f"eval_confusion_matrix_{dataset_name.lower()}.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"4 Visual graphics successfully generated and saved to {RESULTS_DIR}")

    else:
        print("Invalid selection. Please run the script again and type 1, 2, or 3.")

if __name__ == "__main__":
    main()