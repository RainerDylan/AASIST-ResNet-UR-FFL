import sys
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from src.data.dataset import ASVspoofDataset
from src.models.resnet_simam import resnet18_simam
from src.ur_ffl.sensor import UncertaintySensor
from src.ur_ffl.actuator import DegradationActuator

PREPROCESSED_DEV_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\ASVspoof2019_LA_dev_preprocessed"
PROTOCOL_DEV = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"
WEIGHTS_PATH = os.path.join(CURRENT_DIR, "saved_models", "resnet_unified_best.pth")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating Multi-Profile Entropy Analysis using ResNet-SimAM on {device}")

    model = resnet18_simam(num_classes=2, dropout_rate=0.2248).to(device)

    try:
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Successfully loaded ResNet weights.")
    except Exception as e:
        print(f"Warning: Could not load weights. Proceeding with uncalibrated model for testing. Error: {e}")
    
    val_dataset = ASVspoofDataset(PREPROCESSED_DEV_DIR, PROTOCOL_DEV)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)

    sensor = UncertaintySensor(mc_passes=10)
    actuator = DegradationActuator(device)
    
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80).to(device)
    amp_to_db = T.AmplitudeToDB(stype='power', top_db=80).to(device)

    entropy_data = {
        "Clean": [],
        "SSI (Mild)": [],
        "ISD (Moderate)": [],
        "LnL (Severe)": []
    }

    num_batches_to_test = 15
    alpha_intensity = 1.0
    
    pbar = tqdm(val_loader, total=num_batches_to_test, desc="Measuring Entropy Gradient")
    for i, (waveforms, labels) in enumerate(pbar):
        if i >= num_batches_to_test:
            break
            
        waveforms = waveforms.squeeze(1).to(device)
        labels = labels.to(device)

        with torch.no_grad():
            feat_clean = amp_to_db(mel_transform(waveforms)).unsqueeze(1)
            H_clean, _ = sensor.measure(model, feat_clean)
            entropy_data["Clean"].extend(H_clean.cpu().tolist())

        aug_ssi = actuator.apply(waveforms, labels, ["flatten"] * waveforms.size(0), alpha_intensity)
        with torch.no_grad():
            feat_ssi = amp_to_db(mel_transform(aug_ssi)).unsqueeze(1)
            H_ssi, _ = sensor.measure(model, feat_ssi)
            entropy_data["SSI (Mild)"].extend(H_ssi.cpu().tolist())

        aug_isd = actuator.apply(waveforms, labels, ["codec"] * waveforms.size(0), alpha_intensity)
        with torch.no_grad():
            feat_isd = amp_to_db(mel_transform(aug_isd)).unsqueeze(1)
            H_isd, _ = sensor.measure(model, feat_isd)
            entropy_data["ISD (Moderate)"].extend(H_isd.cpu().tolist())

        aug_lnl = actuator.apply(waveforms, labels, ["smear"] * waveforms.size(0), alpha_intensity)
        with torch.no_grad():
            feat_lnl = amp_to_db(mel_transform(aug_lnl)).unsqueeze(1)
            H_lnl, _ = sensor.measure(model, feat_lnl)
            entropy_data["LnL (Severe)"].extend(H_lnl.cpu().tolist())

    print("\n" + "="*65)
    print("EPISTEMIC UNCERTAINTY CALIBRATION STATISTICS (nats)")
    print("="*65)
    print(f"{'Condition':<18} | {'Mean':<8} | {'Std Dev':<8} | {'Min':<8} | {'Max':<8}")
    print("-" * 65)
    
    for name, scores in entropy_data.items():
        print(f"{name:<18} | {np.mean(scores):<8.4f} | {np.std(scores):<8.4f} | {np.min(scores):<8.4f} | {np.max(scores):<8.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['blue', 'green', 'orange', 'red']
    
    for idx, (name, scores) in enumerate(entropy_data.items()):
        ax1.hist(scores, bins=30, alpha=0.5, color=colors[idx], label=name, density=True)
    
    ax1.set_title('Predictive Entropy Distribution Gradient')
    ax1.set_xlabel('Predictive Entropy (nats)')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2.boxplot([entropy_data[k] for k in entropy_data.keys()], labels=list(entropy_data.keys()), patch_artist=True)
    ax2.set_title('Uncertainty Variance by Profile Severity')
    ax2.set_ylabel('Predictive Entropy (nats)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle("ResNet-SimAM Epistemic Uncertainty Analysis", fontsize=14)
    output_path = os.path.join(CURRENT_DIR, "results", "entropy_gradient_analysis.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nGradient Analysis successfully saved to {output_path}")

if __name__ == "__main__":
    main()