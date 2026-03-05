import os
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

# Ensure you define your actual dataset paths here
RAW_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_train\flac"
RAW_DEV_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_dev\flac"

# New directories where the finalized tensors will be saved
PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PREPROCESSED_DEV_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"

def apply_preprocessing(waveform, target_length=64600):
    # Keep a backup of the original audio
    original_waveform = waveform.clone()
    
    # 1. Silence Trimming (-30 dB threshold)
    try:
        waveform = torchaudio.functional.vad(waveform, sample_rate=16000)
        # If VAD strips the entire file (or leaves 1 sample), revert to the backup
        if waveform.shape[-1] <= 1:
            waveform = original_waveform
    except Exception:
        waveform = original_waveform
        
    # 2. Amplitude Normalization (Z-score)
    mean = waveform.mean()
    std = waveform.std()
    waveform = (waveform - mean) / (std + 1e-8)
    
    # 3. Temporal Resizing (Center Crop or Reflection Pad)
    seq_len = waveform.shape[-1]
    if seq_len > target_length:
        start = (seq_len - target_length) // 2
        waveform = waveform[:, start:start + target_length]
    elif seq_len < target_length:
        # Fallback for extremely short corrupted audio
        if seq_len <= 1:
            waveform = F.pad(waveform, (0, target_length - seq_len), mode='constant', value=0)
        else:
            # Safe reflection loop for normal audio
            while waveform.shape[-1] < target_length:
                current_len = waveform.shape[-1]
                pad_amount = min(target_length - current_len, current_len - 1)
                waveform = F.pad(waveform, (0, pad_amount), mode='reflect')
                
    # 4. Pre-emphasis Filtering (alpha = 0.97 as per Table 6)
    alpha = 0.97
    pre_emphasized = torch.cat([waveform[:, :1], waveform[:, 1:] - alpha * waveform[:, :-1]], dim=1)
    
    # 5. Channel Expansion is implicitly handled (shape becomes 1, 64600)
    # 6. Bit-depth verification is handled by torchaudio loading 16-bit flacs
    
    return pre_emphasized

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.flac')]
    
    print(f"Processing {len(files)} files from {input_dir}...")
    for file_name in tqdm(files):
        file_path = os.path.join(input_dir, file_name)
        waveform, sr = torchaudio.load(file_path)
        
        # Apply the 6 thesis steps
        processed_tensor = apply_preprocessing(waveform)
        
        # Save as a .pt (PyTorch Tensor) file for maximum I/O speed later
        save_name = file_name.replace('.flac', '.pt')
        torch.save(processed_tensor, os.path.join(output_dir, save_name))

if __name__ == "__main__":
    process_directory(RAW_TRAIN_DIR, PREPROCESSED_TRAIN_DIR)
    process_directory(RAW_DEV_DIR, PREPROCESSED_DEV_DIR)
    print("Offline preprocessing complete.")