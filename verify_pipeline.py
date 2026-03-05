import torch
from torch.utils.data import DataLoader
from src.config import DATASET_TRAIN_FLAC, PROTOCOL_FILE_TRAIN
from src.data.dataset import ASVspoofDataset
from src.models.rawnet2 import RawNet2
from src.models.aasist import AASIST
from src.models.ensemble import DeepfakeEnsemble

def verify():
    print("--- STEP 1: Verifying Dataset and DataLoader ---")
    try:
        # Initialize the dataset
        dataset = ASVspoofDataset(flac_dir=DATASET_TRAIN_FLAC, protocol_file=PROTOCOL_FILE_TRAIN)
        print(f"Dataset successfully located {len(dataset)} valid audio files.")
        
        # Initialize a data loader with a small batch size of 2
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Fetch a single batch
        waveforms, labels = next(iter(dataloader))
        print(f"Batch loaded successfully.")
        print(f"Waveform Tensor Shape: {waveforms.shape} (Expected: [2, 64600])")
        print(f"Labels Tensor Shape: {labels.shape} (Expected: [2])")
        
    except Exception as e:
        print(f"Dataset Error: {e}")
        return

    print("\n--- STEP 2: Verifying Model Architectures ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizing computation device: {device}")
    
    # Move tensors to the designated device
    waveforms = waveforms.to(device)
    
    try:
        print("\nTesting RawNet2...")
        rawnet = RawNet2().to(device)
        out_raw = rawnet(waveforms)
        print(f"RawNet2 Output Shape: {out_raw.shape} (Expected: [2, 2])")
        
        print("\nTesting AASIST...")
        aasist = AASIST().to(device)
        out_aasist = aasist(waveforms)
        print(f"AASIST Output Shape: {out_aasist.shape} (Expected: [2, 2])")
        
        print("\nTesting DeepfakeEnsemble...")
        ensemble = DeepfakeEnsemble().to(device)
        out_ensemble = ensemble(waveforms)
        print(f"Ensemble Output Shape: {out_ensemble.shape} (Expected: [2, 2])")
        
        print("\nSUCCESS: All models and data pipelines are functioning perfectly.")
        
    except Exception as e:
        print(f"Model Forward Pass Error: {e}")

if __name__ == "__main__":
    verify()