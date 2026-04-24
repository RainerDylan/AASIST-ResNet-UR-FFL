import sys
import os
import torch
import torch.nn as nn
import torchaudio.transforms as T
from tqdm import tqdm

# Resolve root directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..')) if 'ensemble' in CURRENT_DIR else CURRENT_DIR
sys.path.append(ROOT_DIR)

MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")

from src.models.aasist import AASIST
from src.models.resnet_simam import resnet18_simam

# --- PATH CONFIGURATION ---
PREPROCESSED_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_preprocessed"
PREPROCESSED_DEV_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"

PROTOCOL_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
PROTOCOL_DEV = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"

AASIST_WEIGHTS = os.path.join(MODELS_DIR, "aasist_baseline_best.pth")
RESNET_WEIGHTS = os.path.join(MODELS_DIR, "resnet_baseline_best.pth")

# --- NEW DIRECTORIES FOR EMBEDDINGS ---
EMBEDDINGS_TRAIN_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_train_embeddings"
EMBEDDINGS_DEV_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_embeddings"
os.makedirs(EMBEDDINGS_TRAIN_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DEV_DIR, exist_ok=True)

def process_and_save_embeddings(protocol_file, source_dir, target_dir, aasist_model, resnet_model, mel_transform, amp_to_db, device):
    buffer_aasist = [None]
    buffer_resnet = [None]
    
    # Register hooks to intercept the embeddings
    h1 = aasist_model.fc.register_forward_hook(lambda m, i, o: buffer_aasist.__setitem__(0, i[0]))
    h2 = resnet_model.fc.register_forward_hook(lambda m, i, o: buffer_resnet.__setitem__(0, i[0]))

    with open(protocol_file, 'r') as f:
        lines = f.readlines()

    success_count = 0
    pbar = tqdm(lines, desc=f"Extracting to {os.path.basename(target_dir)}")
    
    with torch.no_grad():
        for line in pbar:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            fname = parts[1]
            source_file = os.path.join(source_dir, f"{fname}.pt")
            target_file = os.path.join(target_dir, f"{fname}.pt")
            
            # Skip if already extracted
            if os.path.exists(target_file):
                success_count += 1
                continue
                
            if not os.path.exists(source_file):
                continue

            # Load 1D waveform and add Batch dimension: (1, 64600)
            waveform = torch.load(source_file).to(device)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # 1. AASIST Forward
            _ = aasist_model(waveform)
            emb_a = buffer_aasist[0] # Shape: (1, 104)
            
            # 2. ResNet Forward
            mel_spec = mel_transform(waveform)
            features_img = amp_to_db(mel_spec).unsqueeze(1) # Shape: (1, 1, Freq, Time)
            _ = resnet_model(features_img)
            emb_r = buffer_resnet[0] # Shape: (1, 512)
            
            # 3. Concatenate and Save
            master_embedding = torch.cat([emb_a, emb_r], dim=1).squeeze(0) # Shape: (616,)
            torch.save(master_embedding.cpu(), target_file)
            success_count += 1
            
    h1.remove()
    h2.remove()
    print(f"Finished. Saved {success_count} embeddings to {target_dir}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Offline Embedding Extraction on {device}")

    print("Loading Frozen AASIST...")
    aasist_model = AASIST(
        stft_window=698, stft_hop=398, freq_bins=116, gat_layers=2, 
        heads=5, head_dim=104, hidden_dim=455, dropout=0.3311465671378094
    ).to(device)
    aasist_model.load_state_dict(torch.load(AASIST_WEIGHTS, map_location=device))
    aasist_model.eval()

    print("Loading Frozen ResNet-SimAM...")
    resnet_model = resnet18_simam(num_classes=2, dropout_rate=0.22489397436884667).to(device)
    resnet_model.load_state_dict(torch.load(RESNET_WEIGHTS, map_location=device))
    resnet_model.eval()

    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80).to(device)
    amp_to_db = T.AmplitudeToDB(stype='power', top_db=80).to(device)

    print("\n--- Processing Training Set ---")
    process_and_save_embeddings(PROTOCOL_TRAIN, PREPROCESSED_TRAIN_DIR, EMBEDDINGS_TRAIN_DIR, aasist_model, resnet_model, mel_transform, amp_to_db, device)
    
    print("\n--- Processing Validation Set ---")
    process_and_save_embeddings(PROTOCOL_DEV, PREPROCESSED_DEV_DIR, EMBEDDINGS_DEV_DIR, aasist_model, resnet_model, mel_transform, amp_to_db, device)

if __name__ == "__main__":
    main()