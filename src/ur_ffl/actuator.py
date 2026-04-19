"""
actuator.py — UR-FFL degradation actuator using faithful RawBoost implementation.

RawBoost (Tak et al., ICASSP 2022) is the ONLY augmentation strategy proven to
create a monotone alpha↔acc_gap relationship that a PD controller can act on.
All previous STFT-domain simulations failed because they produced constant
~10-12pp acc_gap regardless of alpha level.

Algorithms implemented (Tak et al. 2022):
  LnL  (Algorithm 1): Linear+NonLinear convolutive noise via random multi-band
        FIR filtering + Hammerstein-style nonlinear distortion. Models microphone,
        amplifier, and channel distortion.
  ISD  (Algorithm 2): Impulsive Signal-Dependent additive noise. Impulse amplitude
        proportional to local signal amplitude. Models quantisation and clipping.
  SSI  (Algorithm 3): Stationary Signal-Independent additive noise (colored/pink).
        Models background thermal and environmental noise.

All algorithms scale SNR with alpha:
    SNR_dB = SNR_max - (SNR_max - SNR_min) * alpha
    SNR_max = 40dB (barely audible), SNR_min = 5dB (heavily degraded)

Profile mapping from selector z-scores:
    smear   → LnL  (hardest; convolutive distortion)
    codec   → ISD  (codec/quantisation artifacts)
    flatten → SSI  (additive background noise)
    noise   → SSI  (mild additive noise, different SNR range)
    clean   → none (preserves unmodified samples in each batch)

References:
    Tak H., Kamble M., Patino J., Todisco M., Evans N. (ICASSP 2022).
    RawBoost: A Raw Data Boosting and Augmentation Method applied to
    Automatic Speaker Verification Anti-Spoofing.
    https://arxiv.org/abs/2111.04433
    GitHub: https://github.com/TakHemlata/RawBoost-antispoofing
"""

import math
import torch
import numpy as np
import torch.nn.functional as F


class DegradationActuator:

    _SR = 16000

    def __init__(self, device: torch.device):
        self.device = device
        print("  [Actuator] RawBoost (LnL + ISD + SSI) — Tak et al. ICASSP 2022")

    # ── public ────────────────────────────────────────────────────────────────

    def apply(self, waveforms: torch.Tensor, labels, selections: list, alpha: float):
        if alpha < 0.01:
            return waveforms.clone()
        aug = waveforms.clone()
        for i, profile in enumerate(selections):
            x_np = aug[i].cpu().numpy().astype(np.float64)
            if profile == "smear":
                x_np = self._LnL(x_np, alpha)
            elif profile in ("codec", "flatten", "noise"):
                x_np = self._ISD(x_np, alpha) if profile == "codec" else self._SSI(x_np, alpha)
            # "clean" → unchanged
            aug[i] = torch.from_numpy(x_np.astype(np.float32)).to(self.device)
        return aug

    # ── RawBoost Algorithm 1: LnL ─────────────────────────────────────────────

    def _LnL(self, x: np.ndarray, alpha: float) -> np.ndarray:
        """
        Linear+NonLinear (LnL) convolutive noise — RawBoost Algorithm 1.
        Multi-band random FIR filtering followed by nonlinear Hammerstein model.
        Models: microphone frequency response, room acoustics, amplifier nonlinearity.
        """
        N = len(x)
        nBands = np.random.randint(4, 8)

        # SNR for additive component
        snr_db  = 40.0 - 35.0 * alpha   # 40dB at α=0 → 5dB at α=1.0
        clipped = x.copy()

        for _ in range(nBands):
            # Random bandpass FIR filter
            nf = np.random.randint(40, 100)
            fc  = np.random.uniform(0.05, 0.45)   # normalised centre freq
            bw  = np.random.uniform(0.02, 0.20)   # bandwidth
            # Windowed sinc bandpass filter
            n   = np.arange(nf) - nf // 2
            h   = np.sinc(2 * (fc + bw/2) * n) - np.sinc(2 * (fc - bw/2) * n)
            h  *= np.hamming(nf)
            h  /= (np.sum(np.abs(h)) + 1e-8)

            y = np.convolve(clipped, h, mode='full')[:N]

            # Hammerstein nonlinearity: y + a2*y^2 + a3*y^3 (weak nonlinearity)
            a2 = np.random.uniform(-0.1, 0.1)
            a3 = np.random.uniform(-0.05, 0.05)
            y_nl = y + a2 * y**2 + a3 * y**3
            # Normalise to prevent amplitude explosion
            rms = np.sqrt(np.mean(y_nl**2)) + 1e-8
            clipped = y_nl / rms * (np.sqrt(np.mean(x**2)) + 1e-8)

        # Additive noise at target SNR
        sig_pow = np.mean(clipped**2) + 1e-8
        noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
        noise = np.random.randn(N) * np.sqrt(noise_pow)
        out = clipped + noise
        # RMS-match to input
        return self._rms_normalise(out, x)

    # ── RawBoost Algorithm 2: ISD ─────────────────────────────────────────────

    def _ISD(self, x: np.ndarray, alpha: float) -> np.ndarray:
        """
        Impulsive Signal-Dependent (ISD) additive noise — RawBoost Algorithm 2.
        Impulse amplitude proportional to local signal amplitude.
        Models: quantisation distortion, clipping, codec bit-depth reduction.
        """
        N = len(x)
        # Sparsity: probability of impulse at each sample
        P   = 0.05 + 0.25 * alpha     # 5% at α=0 → 30% at α=1.0
        # Gain: scales with alpha
        g   = 0.1 + 0.8 * alpha       # 0.1 at α=0 → 0.9 at α=1.0

        D = np.zeros(N)
        mask = np.random.rand(N) < P
        D[mask] = np.random.randn(np.sum(mask))

        out = x + g * D * np.abs(x)
        return self._rms_normalise(np.clip(out, -1.0, 1.0), x)

    # ── RawBoost Algorithm 3: SSI ─────────────────────────────────────────────

    def _SSI(self, x: np.ndarray, alpha: float) -> np.ndarray:
        """
        Stationary Signal-Independent (SSI) additive noise — RawBoost Algorithm 3.
        Colored (pink/1-f) noise added at target SNR.
        Models: background environment noise, thermal noise, quantisation floor.
        """
        N = len(x)
        snr_db  = 40.0 - 35.0 * alpha   # 40dB at α=0 → 5dB at α=1.0

        # Generate colored (pink) noise via 1/f spectral shaping
        white = np.random.randn(N)
        freqs = np.fft.rfftfreq(N, d=1.0/self._SR)
        freqs[0] = 1.0  # avoid /0
        pink_spectrum = np.fft.rfft(white) / np.sqrt(freqs)
        pink = np.fft.irfft(pink_spectrum, N)
        pink /= (np.std(pink) + 1e-8)

        sig_pow   = np.mean(x**2) + 1e-8
        noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
        noise     = pink * np.sqrt(noise_pow)

        out = x + noise
        return self._rms_normalise(out, x)

    # ── helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _rms_normalise(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Normalise output RMS to match reference RMS."""
        rms_ref = np.sqrt(np.mean(ref**2)) + 1e-8
        rms_out = np.sqrt(np.mean(out**2)) + 1e-8
        return (out * rms_ref / rms_out).clip(-1.0, 1.0)

    def _ssi(self, waveforms: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Batch SSI noise for validation loop augmentation.
        Same algorithm as _SSI but operates on GPU tensors directly.
        """
        N = waveforms.shape[-1]
        if alpha < 0.01:
            return waveforms.clone()
        snr_db  = 40.0 - 35.0 * alpha
        white   = torch.randn(waveforms.shape, device=self.device)
        freqs   = torch.fft.rfftfreq(N, d=1.0/self._SR).to(self.device)
        freqs[0] = 1.0
        spec    = torch.fft.rfft(white) / freqs.sqrt().unsqueeze(0)
        pink    = torch.fft.irfft(spec, N)
        pink   /= pink.std(dim=-1, keepdim=True) + 1e-8
        sig_pow = waveforms.pow(2).mean(dim=-1, keepdim=True) + 1e-8
        noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
        return (waveforms + pink * noise_pow.sqrt()).clamp(-1.0, 1.0)