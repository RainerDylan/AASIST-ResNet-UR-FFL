import sys
import os
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from src.data.dataset import ASVspoofDataset
from src.ur_ffl.actuator import DegradationActuator

PREPROCESSED_DEV_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"
PROTOCOL_DEV = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"

def compute_kl_divergence(clean_mel, deg_mel):
    clean_mel = clean_mel - clean_mel.min() + 1e-8
    deg_mel = deg_mel - deg_mel.min() + 1e-8
    
    B = clean_mel.size(0)
    
    clean_flat = clean_mel.reshape(B, -1)
    deg_flat = deg_mel.reshape(B, -1)
    
    p_clean = clean_flat / clean_flat.sum(dim=1, keepdim=True)
    p_deg = deg_flat / deg_flat.sum(dim=1, keepdim=True)
    
    kl_div = F.kl_div(p_deg.log(), p_clean, reduction='batchmean')
    return kl_div.item()

def save_spectrogram_comparison(clean_mel, ssi_mel, isd_mel, lnl_mel, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    
    mels = [clean_mel, ssi_mel, isd_mel, lnl_mel]
    titles = ["Clean Audio", "SSI (Noise)", "ISD (Codec)", "LnL (Smear)"]
    
    for i in range(4):
        img = mels[i][0, 0].cpu().numpy()
        axes[i].imshow(img, aspect='auto', origin='lower', cmap='viridis')
        axes[i].set_title(titles[i])
        axes[i].set_ylabel("Mel Frequency Bins")
        axes[i].set_xlabel("Time Frames")
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initiating Advanced KL Divergence Analysis on {device}")

    val_dataset = ASVspoofDataset(PREPROCESSED_DEV_DIR, PROTOCOL_DEV)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)

    actuator = DegradationActuator(device)
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80).to(device)
    amp_to_db = T.AmplitudeToDB(stype='power', top_db=80).to(device)

    kl_scores = {"LnL (Smear)": [], "ISD (Codec)": [], "SSI (Noise)": []}

    num_batches_to_test = 15
    alpha_test_level = 0.8
    saved_spectrograms = False
    
    pbar = tqdm(val_loader, total=num_batches_to_test, desc="Computing Divergence")
    for i, (waveforms, labels) in enumerate(pbar):
        if i >= num_batches_to_test:
            break
            
        waveforms = waveforms.squeeze(1).to(device)
        labels = labels.to(device)

        clean_mel = amp_to_db(mel_transform(waveforms))

        selections_lnl = ["smear"] * waveforms.size(0)
        aug_lnl = actuator.apply(waveforms, labels, selections_lnl, alpha_test_level)
        mel_lnl = amp_to_db(mel_transform(aug_lnl))
        kl_scores["LnL (Smear)"].append(compute_kl_divergence(clean_mel, mel_lnl))

        selections_isd = ["codec"] * waveforms.size(0)
        aug_isd = actuator.apply(waveforms, labels, selections_isd, alpha_test_level)
        mel_isd = amp_to_db(mel_transform(aug_isd))
        kl_scores["ISD (Codec)"].append(compute_kl_divergence(clean_mel, mel_isd))

        selections_ssi = ["flatten"] * waveforms.size(0)
        aug_ssi = actuator.apply(waveforms, labels, selections_ssi, alpha_test_level)
        mel_ssi = amp_to_db(mel_transform(aug_ssi))
        kl_scores["SSI (Noise)"].append(compute_kl_divergence(clean_mel, mel_ssi))
        
        if not saved_spectrograms:
            spec_path = os.path.join(CURRENT_DIR, "results", "degradation_spectrograms.png")
            save_spectrogram_comparison(clean_mel, mel_ssi, mel_isd, mel_lnl, spec_path)
            saved_spectrograms = True

    print("\n" + "="*60)
    print("DETAILED KL DIVERGENCE STATISTICS (nats)")
    print("="*60)
    print(f"{'Profile':<15} | {'Mean':<8} | {'Std Dev':<8} | {'Min':<8} | {'Max':<8}")
    print("-" * 60)
    
    plot_data = []
    for name, scores in kl_scores.items():
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        min_val = np.min(scores)
        max_val = np.max(scores)
        plot_data.append((name, mean_val, std_val, scores))
        print(f"{name:<15} | {mean_val:<8.4f} | {std_val:<8.4f} | {min_val:<8.4f} | {max_val:<8.4f}")
    
    plot_data.sort(key=lambda x: x[1], reverse=True)
    names = [x[0] for x in plot_data]
    means = [x[1] for x in plot_data]
    stds = [x[2] for x in plot_data]
    all_scores = [x[3] for x in plot_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.bar(names, means, yerr=stds, capsize=10, color=['darkred', 'darkorange', 'steelblue'], alpha=0.8)
    ax1.set_title('Mean KL Divergence by Profile')
    ax1.set_ylabel('KL Divergence (nats)')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    
    ax2.boxplot(all_scores, labels=names, patch_artist=True)
    ax2.set_title('Distribution of KL Divergence Scores')
    ax2.set_ylabel('KL Divergence (nats)')
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.suptitle("Forensic Hierarchy Quantification Analysis", fontsize=14)
    output_path = os.path.join(CURRENT_DIR, "results", "degradation_hierarchy_stats.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualizations saved successfully to the results folder.")

if __name__ == "__main__":
    main()