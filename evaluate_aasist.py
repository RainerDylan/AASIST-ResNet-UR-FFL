import os
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
from sklearn.metrics import roc_curve, auc, DetCurveDisplay

from src.data.dataset import ASVspoofDataset
from src.models.aasist import AASIST

# --- PATH CONFIGURATION ---
RAW_CUSTOM_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\Custom_Raw_Audio"
RAW_EVAL_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_eval\flac"
SHARED_PREPROCESSED_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\evaluation_preprocessed"
PROTOCOL_EVAL = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt"
MODEL_WEIGHTS = "aasist_baseline_best.pth"

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
    for file_path in tqdm(audio_files, desc="Preprocessing Custom Audio"):
        waveform, sr = torchaudio.load(file_path)
        
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        waveform = apply_vad_and_norm(waveform)
        seq_len = waveform.shape[-1]
        chunks = []
        
        # Sliding Window Logic for Custom Files
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
    return file_to_chunks_map

def preprocess_evaluation(target_length=64600):
    audio_files = glob.glob(os.path.join(RAW_EVAL_DIR, '*.flac'))
    if not audio_files:
        print(f"No FLAC files found in {RAW_EVAL_DIR}.")
        return
        
    for file_path in tqdm(audio_files, desc="Preprocessing Eval Dataset"):
        waveform, _ = torchaudio.load(file_path)
        waveform = apply_vad_and_norm(waveform)
        
        seq_len = waveform.shape[-1]
        
        # Strict Center Crop / Reflect Logic for Evaluation Dataset
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
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(SHARED_PREPROCESSED_DIR, f"{base_name}.pt")
        torch.save(pre_emphasized, out_path)

def initialize_model(device):
    print(f"\nLoading trained weights from {MODEL_WEIGHTS}...")
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
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*40)
    print("AASIST DYNAMIC EVALUATION SCRIPT")
    print("="*40)
    print("[1] Custom Audio Inference (Sliding Window)")
    print("[2] Official ASVspoof Evaluation (EER, AUC & Accuracy)")
    
    choice = input("Select mode (1 or 2): ").strip()
    
    if not os.path.exists(SHARED_PREPROCESSED_DIR):
        os.makedirs(SHARED_PREPROCESSED_DIR)

    if choice == '1':
        print("\n--- CUSTOM AUDIO MODE ---")
        print(f"Clearing {SHARED_PREPROCESSED_DIR} for new custom data...")
        for f in os.listdir(SHARED_PREPROCESSED_DIR):
            os.remove(os.path.join(SHARED_PREPROCESSED_DIR, f))
            
        file_map = preprocess_custom()
        if not file_map: return
        
        model = initialize_model(device)
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

    elif choice == '2':
        print("\n--- OFFICIAL EVALUATION MODE ---")
        existing_files = glob.glob(os.path.join(SHARED_PREPROCESSED_DIR, '*.pt'))
        
        if len(existing_files) > 1000:
            print(f"Found {len(existing_files)} preprocessed files in the folder.")
            skip = input("Do you want to skip preprocessing and evaluate immediately? (y/n): ").strip().lower()
            if skip != 'y':
                for f in existing_files: os.remove(f)
                preprocess_evaluation()
        else:
            print("Clearing folder and preprocessing evaluation dataset...")
            for f in existing_files: os.remove(f)
            preprocess_evaluation()
            
        model = initialize_model(device)
        print("Loading official protocol and preprocessed tensors...")
        
        eval_dataset = ASVspoofDataset(SHARED_PREPROCESSED_DIR, PROTOCOL_EVAL)
        eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=4)
        
        all_labels = []
        all_probs = []
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for waveforms, labels in tqdm(eval_loader, desc="Evaluating Benchmarks"):
                waveforms = waveforms.squeeze(1).to(device)
                labels = labels.to(device)
                outputs = model(waveforms)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                # Tracking Accuracy
                _, predicted_classes = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted_classes == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate Metrics
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
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FAR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig("eval_roc_curve.png", dpi=300)
        plt.close()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        display = DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name='AASIST Model')
        display.plot(ax=ax)
        plt.title('Detection Error Tradeoff (DET) Curve')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig("eval_det_curve.png", dpi=300)
        plt.close()
        print("Graphs saved successfully.")

    else:
        print("Invalid selection. Please run the script again and type 1 or 2.")

if __name__ == "__main__":
    main()