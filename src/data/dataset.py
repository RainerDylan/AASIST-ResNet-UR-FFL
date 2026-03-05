import os
import torch
from torch.utils.data import Dataset

class ASVspoofDataset(Dataset):
    def __init__(self, preprocessed_dir, protocol_file):
        """
        Loads pre-processed .pt audio tensors.
        preprocessed_dir: Path to the folder containing .pt files
        protocol_file: Path to ASVspoof2019.LA.cm.train.trn.txt
        """
        self.preprocessed_dir = preprocessed_dir
        self.files = []
        self.labels = []
        
        # Read the protocol file to map File IDs to Labels
        with open(protocol_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                file_id = parts[1]
                label_str = parts[-1]  # 'bonafide' or 'spoof'
                
                # Assuming 1 is Bonafide, 0 is Spoof
                label = 1 if label_str == 'bonafide' else 0
                
                expected_file_path = os.path.join(self.preprocessed_dir, f"{file_id}.pt")
                if os.path.exists(expected_file_path):
                    self.files.append(expected_file_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Extremely fast load of pre-calculated tensor
        waveform = torch.load(self.files[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return waveform, label