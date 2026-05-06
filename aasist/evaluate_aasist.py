import sys
import os
import random
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

from src.data.dataset import ASVspoofDataset
from src.models.aasist import AASIST

# PATH CONFIGURATION
BASE_DATASET_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset"
RAW_CUSTOM_DIR = os.path.join(BASE_DATASET_DIR, "Custom_Raw_Audio")

PREPROCESSED_CUSTOM_DIR = os.path.join(BASE_DATASET_DIR, "preprocessed_custom")
PREPROCESSED_LA_DIR = os.path.join(BASE_DATASET_DIR, "preprocessed_la")
PREPROCESSED_DF_DIR = os.path.join(BASE_DATASET_DIR, "preprocessed_df")

RAW_LA_EVAL_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_eval\flac"
PROTOCOL_LA_EVAL = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt"

RAW_DF_EVAL_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2021\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\flac"
PROTOCOL_DF_EVAL = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2021\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\trial_metadata.txt" 

def compute_min_dcf(fpr, fnr, p_target=0.05, c_miss=1.0, c_fa=1.0):
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    min_dcf = np.min(dcf)
    min_dcf_idx = np.argmin(dcf)
    default_dcf = min(c_miss * p_target, c_fa * (1.0 - p_target))
    min_dcf_norm = min_dcf / default_dcf
    return min_dcf_norm, dcf, default_dcf, min_dcf_idx

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
            
            line_lower = line_str.lower()
            is_bonafide = 'bonafide' in line_lower
            label_str = 'bonafide' if is_bonafide else 'spoof'
            
            # This strictly enforces the 5-column format required by ASVspoofDataset
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
        for line in interleaved:
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

def preprocess_custom(target_dir, target_length=64600):
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
            out_path = os.path.join(target_dir, f"{base_name}_chunk{i}.pt")
            torch.save(pre_emphasized, out_path)
            saved_chunks.append(out_path)
            
        file_to_chunks_map[base_name] = saved_chunks
        
    if skipped_files > 0:
        print(f"\nNote: Skipped {skipped_files} corrupted custom audio files.")
        
    return file_to_chunks_map

def build_custom_file_map_from_existing(target_dir):
    file_map = {}
    pt_files = glob.glob(os.path.join(target_dir, '*.pt'))
    for pt_path in pt_files:
        filename = os.path.basename(pt_path)
        base_name = filename.rsplit('_chunk', 1)[0]
        if base_name not in file_map:
            file_map[base_name] = []
        file_map[base_name].append(pt_path)
    return file_map

def preprocess_evaluation(eval_dir, protocol_file, target_dir, target_total=7000, target_length=64600):
    with open(protocol_file, 'r') as f:
        lines = f.readlines()
                
    if not lines:
        print(f"No files found in protocol {protocol_file}.")
        return
        
    valid_lines = []
    skipped_files = 0
    success_bonafide = 0
    success_spoof = 0
        
    pbar = tqdm(total=target_total, desc="Preprocessing Eval Dataset")
    
    for line in lines:
        if (success_bonafide + success_spoof) >= target_total:
            break
            
        parts = line.strip().split()
        if len(parts) < 5:
            continue
            
        fname = parts[1]
        label = parts[4]
        is_bonafide = (label == 'bonafide')
        
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
        out_path = os.path.join(target_dir, f"{fname}.pt")
        torch.save(pre_emphasized, out_path)
        valid_lines.append(line)
        
        if is_bonafide:
            success_bonafide += 1
        else:
            success_spoof += 1
            
        pbar.update(1)
        
    pbar.close()
        
    with open(protocol_file, 'w') as f:
        for line in valid_lines:
            f.write(line)
            
    print(f"\nPreprocessing Complete. Saved {success_bonafide} Bonafide and {success_spoof} Spoof samples.")
    print(f"Total valid evaluation samples extracted: {success_bonafide + success_spoof}")
    if skipped_files > 0:
        print(f"Note: Successfully bypassed {skipped_files} corrupted FLAC files to preserve dataset integrity.")

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
    
    os.makedirs(PREPROCESSED_CUSTOM_DIR, exist_ok=True)
    os.makedirs(PREPROCESSED_LA_DIR, exist_ok=True)
    os.makedirs(PREPROCESSED_DF_DIR, exist_ok=True)

    if choice == '1':
        print("\n--- CUSTOM AUDIO MODE ---")
        existing_custom = glob.glob(os.path.join(PREPROCESSED_CUSTOM_DIR, '*.pt'))
        
        if len(existing_custom) > 0:
            print(f"Found {len(existing_custom)} preprocessed files in the custom folder.")
            skip = input("Do you want to skip preprocessing and evaluate immediately? (Y/n): ").strip().lower()
            if skip not in ['n', 'no']:
                file_map = build_custom_file_map_from_existing(PREPROCESSED_CUSTOM_DIR)
            else:
                print(f"Clearing {PREPROCESSED_CUSTOM_DIR} for new custom data...")
                for f in existing_custom: os.remove(f)
                file_map = preprocess_custom(PREPROCESSED_CUSTOM_DIR)
        else:
            file_map = preprocess_custom(PREPROCESSED_CUSTOM_DIR)
            
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
            target_dir = PREPROCESSED_LA_DIR
        else:
            print("\n--- OFFICIAL ASVSPOOF 2021 DF EVALUATION ---")
            raw_dir = RAW_DF_EVAL_DIR
            orig_protocol = PROTOCOL_DF_EVAL
            dataset_name = "DF"
            target_dir = PREPROCESSED_DF_DIR
            
        subset_protocol = os.path.join(target_dir, "subset_protocol.txt")
        existing_files = glob.glob(os.path.join(target_dir, '*.pt'))
        
        if len(existing_files) >= 7000 and os.path.exists(subset_protocol):
            print(f"Found {len(existing_files)} preprocessed files in the {dataset_name} folder.")
            skip = input("Do you want to skip preprocessing and evaluate immediately? (Y/n): ").strip().lower()
            if skip not in ['n', 'no']:
                print("Skipping preprocessing. Utilizing existing dataset.")
            else:
                print(f"Clearing folder and preprocessing robust evaluation dataset subset for {dataset_name}...")
                for f in existing_files: os.remove(f)
                create_shuffled_protocol(orig_protocol, subset_protocol)
                preprocess_evaluation(raw_dir, subset_protocol, target_dir, target_total=7000)
        else:
            if len(existing_files) > 0:
                print(f"Found {len(existing_files)} files. We need exactly 7000. Clearing and restarting preprocessing for {dataset_name}...")
                for f in existing_files: os.remove(f)
            else:
                print(f"Clearing folder and preprocessing robust evaluation dataset subset for {dataset_name}...")
            
            create_shuffled_protocol(orig_protocol, subset_protocol)
            preprocess_evaluation(raw_dir, subset_protocol, target_dir, target_total=7000)
            
        model = initialize_model(device, selected_weights)
        print("Loading official protocol and preprocessed tensors...")
        
        eval_dataset = ASVspoofDataset(target_dir, subset_protocol)
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
        
        min_dcf_norm, dcf_curve, default_dcf, min_dcf_idx = compute_min_dcf(fpr, fnr)
        
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        deepfake_scores = all_probs[all_labels == 0]
        bonafide_scores = all_probs[all_labels == 1]
        
        print("\n" + "="*40)
        print("EXTENDED AI DIAGNOSTIC REPORT")
        print("="*40)
        print("--- CONFUSION MATRIX ---")
        print(f"True Positives (Correct Spoof):  {tp}")
        print(f"True Negatives (Correct Bonafide): {tn}")
        print(f"False Positives (False Spoof):   {fp}")
        print(f"False Negatives (Missed Spoof):  {fn}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
        
        print("\n--- SCORE DISTRIBUTION ---")
        print("Bonafide (Class 1) Probabilities:")
        print(f"  Mean: {np.mean(bonafide_scores):.4f} | Std: {np.std(bonafide_scores):.4f}")
        print(f"  Min:  {np.min(bonafide_scores):.4f} | Max: {np.max(bonafide_scores):.4f}")
        print("Deepfake (Class 0) Probabilities:")
        print(f"  Mean: {np.mean(deepfake_scores):.4f} | Std: {np.std(deepfake_scores):.4f}")
        print(f"  Min:  {np.min(deepfake_scores):.4f} | Max: {np.max(deepfake_scores):.4f}")
        
        print("\n--- ROC & THRESHOLD ANALYSIS ---")
        fpr_targets = [0.01, 0.05, 0.10]
        for t_fpr in fpr_targets:
            idx = np.argmin(np.abs(fpr - t_fpr))
            print(f"TPR @ FPR={t_fpr*100:.0f}%: {tpr[idx]*100:.2f}% (Threshold: {thresholds[idx]:.4f})")
            
        print("\n--- DETECTION COST FUNCTION (DCF) ---")
        print(f"Min Normalized DCF: {min_dcf_norm:.4f} at Decision Threshold: {thresholds[min_dcf_idx]:.4f}")
        
        print("\n" + "="*40)
        print("FINAL EVALUATION METRICS")
        print("="*40)
        print(f"Average Accuracy: {accuracy:.4f}%")
        print(f"Equal Error Rate (EER): {eer:.4f}%")
        print(f"Area Under the Curve (AUC): {auc_score:.4f}")
        print(f"Normalized Minimum DCF (t-DCF Proxy): {min_dcf_norm:.4f}")
        print("="*40)
        
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
        roc_path = os.path.join(RESULTS_DIR, f"aasist_roc_curve_{dataset_name.lower()}.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        display = DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name='AASIST Model')
        display.plot(ax=ax)
        plt.title(f'DET Curve ({dataset_name} Evaluation)')
        plt.grid(True, linestyle=':', alpha=0.6)
        det_path = os.path.join(RESULTS_DIR, f"aasist_det_curve_{dataset_name.lower()}.png")
        plt.savefig(det_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(8, 6))
        plt.hist(bonafide_scores, bins=50, alpha=0.6, color='#2ca02c', label='Bonafide (True Class)', edgecolor='white', linewidth=0.5)
        plt.hist(deepfake_scores, bins=50, alpha=0.6, color='#d62728', label='Deepfake (Spoof Class)', edgecolor='white', linewidth=0.5)
        plt.title(f'Model Confidence Score Distribution ({dataset_name} Evaluation)', fontsize=12, pad=15)
        plt.xlabel('Predicted Probability of being Bonafide', fontsize=10)
        plt.ylabel('Number of Audio Samples', fontsize=10)
        plt.legend(loc="upper center")
        plt.grid(True, linestyle=':', alpha=0.6)
        dist_path = os.path.join(RESULTS_DIR, f"aasist_score_dist_{dataset_name.lower()}.png")
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 6))
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Deepfake', 'Bonafide'])
        cm_display.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
        plt.title(f'Confusion Matrix ({dataset_name} Evaluation)', fontsize=12, pad=15)
        plt.xlabel('Predicted Classification', fontsize=10)
        plt.ylabel('True Ground Label', fontsize=10)
        cm_path = os.path.join(RESULTS_DIR, f"aasist_confusion_matrix_{dataset_name.lower()}.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        valid_idx = (thresholds >= 0.0) & (thresholds <= 1.0)
        valid_thresholds = thresholds[valid_idx]
        valid_dcf = (dcf_curve / default_dcf)[valid_idx]
        
        plt.figure(figsize=(8, 6))
        plt.plot(valid_thresholds, valid_dcf, color='purple', lw=2, label='Normalized DCF Curve')
        plt.plot(thresholds[min_dcf_idx], min_dcf_norm, 'ro', markersize=8, label=f'Min DCF = {min_dcf_norm:.4f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, max(valid_dcf) * 1.1])
        plt.xlabel('Decision Threshold')
        plt.ylabel('Normalized Detection Cost')
        plt.title(f'Detection Cost Function Curve ({dataset_name} Evaluation)')
        plt.legend(loc="upper center")
        plt.grid(True, linestyle=':', alpha=0.6)
        dcf_path = os.path.join(RESULTS_DIR, f"aasist_dcf_curve_{dataset_name.lower()}.png")
        plt.savefig(dcf_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"5 Visual graphics successfully generated and saved to {RESULTS_DIR}")

    else:
        print("Invalid selection. Please run the script again and type 1, 2, or 3.")

if __name__ == "__main__":
    main()