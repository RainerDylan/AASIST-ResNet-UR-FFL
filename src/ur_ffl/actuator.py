import torch
import numpy as np

class DegradationActuator:
    def __init__(self, device):
        self.device = device

    def apply(self, waveforms, labels, selections, alpha):
        aug_waveforms = waveforms.clone()
        
        for i, choice in enumerate(selections):
            if choice == 'smear':
                aug_waveforms[i] = self._apply_smear(aug_waveforms[i], alpha)
            elif choice == 'ripple':
                aug_waveforms[i] = self._apply_ripple(aug_waveforms[i], alpha)
            elif choice == 'quantize':
                aug_waveforms[i] = self._apply_quantize(aug_waveforms[i], alpha)
            elif choice == 'noise':
                aug_waveforms[i] = self._apply_noise(aug_waveforms[i], alpha)

        return aug_waveforms

    def _apply_smear(self, waveform, alpha):
        n_fft = 512
        hop_length = 256
        window = torch.hann_window(n_fft).to(self.device)
        
        stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        mag = torch.abs(stft)
        phase = torch.angle(stft)
        
        max_shift = alpha * (np.pi / 2.0)
        noise = (torch.rand_like(phase) * 2.0 - 1.0) * max_shift
        phase_deg = phase + noise
        
        stft_deg = torch.polar(mag, phase_deg)
        return torch.istft(stft_deg, n_fft=n_fft, hop_length=hop_length, window=window, length=waveform.size(-1))

    def _apply_ripple(self, waveform, alpha):
        n_fft = 512
        hop_length = 256
        window = torch.hann_window(n_fft).to(self.device)
        
        stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        mag = torch.abs(stft)
        phase = torch.angle(stft)
        
        order = max(3, min(8, int(np.floor(alpha * 10.0))))
        freqs = torch.arange(0, mag.size(0)).to(self.device)
        
        filter_gain = 1.0 + torch.sin(2.0 * np.pi * freqs * order / 128.0)
        filter_gain = filter_gain.unsqueeze(-1)
        
        mag_filtered = mag * filter_gain
        stft_deg = torch.polar(mag_filtered, phase)
        return torch.istft(stft_deg, n_fft=n_fft, hop_length=hop_length, window=window, length=waveform.size(-1))

    def _apply_quantize(self, waveform, alpha):
        bits = max(8, min(16, int(16 - np.floor(alpha * 8.0))))
        levels = 2 ** bits
        
        x_scaled = waveform * (levels - 1)
        x_quant = torch.round(x_scaled)
        return x_quant / (levels - 1)

    def _apply_noise(self, waveform, alpha):
        max_amp = torch.max(torch.abs(waveform))
        sigma = 0.01 * alpha * max_amp
        
        noise = torch.randn_like(waveform) * sigma
        x_noisy = waveform + noise
        x_deg = torch.clamp(x_noisy, min=-1.0, max=1.0)
        x_deg = x_deg - torch.mean(x_deg)
        
        return x_deg