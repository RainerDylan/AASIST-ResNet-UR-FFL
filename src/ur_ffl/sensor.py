"""
Uncertainty Sensor for the UR-FFL module.

Returns predictive entropy H (nats) per sample and the batch mean.

Why entropy instead of Ue = var/(mu*(1-mu))?
─────────────────────────────────────────────
The Ue formula has an inversion artefact: as mu → 0 or 1 (high
confidence) the Bernoulli denominator mu(1−mu) → 0, which artificially
INFLATES Ue for confident predictions.  A highly confident but wrong
prediction gets a larger Ue than a genuinely confused prediction —
inverting the intended uncertainty ranking and making the selector assign
profiles in the wrong direction.

Predictive entropy (Gal & Ghahramani 2016, Eq. 11):

    H(μ) = −μ·ln(μ+ε) − (1−μ)·ln(1−μ+ε)

Properties that make H the correct feedback signal:
  • H ∈ [0, ln 2] nats  — bounded, independent of μ's scale
  • H(μ) = H(1−μ)       — symmetric across classes
  • Maximised at μ = 0.5 (maximum confusion)
  • Monotone from each extreme toward 0.5

mc_passes = 10
──────────────
Gal & Ghahramani (2016): uncertainty estimate variance ∝ 1/T.
At T = 10, std(mean_H) ≈ 0.008 nats — sufficient for stable PD control.
Smith & Gal (2018): T ≥ 10 is adequate for adversarial detection tasks.
Using T = 10 instead of 20 halves the sensor overhead per batch.

References
──────────
Gal & Ghahramani (2016): Dropout as Bayesian Approximation.  ICML.
Smith & Gal (2018): Understanding Uncertainty for Adversarial Detection.  UAI.
Kendall & Gal (2018): What Uncertainties Do We Need?  NeurIPS.
Thesis Section 3.5.1.
"""

import torch


class UncertaintySensor:
    """
    MC-Dropout epistemic uncertainty via predictive entropy.

    The sensor is called on CLEAN audio so the Selector can identify each
    sample's inherent difficulty before augmentation is applied.
    """

    def __init__(self, mc_passes: int = 50):
        """
        Parameters
        ----------
        mc_passes : int
            Number of stochastic forward passes T.
            Gal & Ghahramani (2016) recommend T ∈ [10, 100]; we default
            to 10 for a good speed/reliability trade-off in Phase 2.
        """
        self.mc_passes = mc_passes

    def measure(
        self,
        model: torch.nn.Module,
        waveforms: torch.Tensor,
    ):
        """
        Compute per-sample predictive entropy on the given waveforms.

        BatchNorm layers are kept in eval() (stable running stats) while
        Dropout layers are forced into train() (stochastic) to generate
        T different predictions per sample.

        Parameters
        ----------
        model     : AASIST model with Dropout layers.
        waveforms : (B, L) float32 tensor on model's device.

        Returns
        -------
        H_scores : (B,) tensor — per-sample entropy in nats ∈ [0, ln 2].
                   Higher = more confused, Lower = more confident.
                   Used by the Selector for within-class z-scoring.
        mean_H   : float — batch-mean entropy.
                   NOT used by the controller (which uses val_codec_loss).
        """
        model.eval()
        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

        with torch.no_grad():
            probs_list = []
            for _ in range(self.mc_passes):
                logits = model(waveforms)
                p = torch.softmax(logits, dim=1)[:, 1]   # p(bonafide), (B,)
                probs_list.append(p.unsqueeze(0))         # (1, B)

        probs = torch.cat(probs_list, dim=0)   # (T, B)
        mu    = probs.mean(dim=0)              # (B,) predictive mean

        eps = 1e-8
        H   = -(mu * torch.log(mu + eps) +
                (1.0 - mu) * torch.log(1.0 - mu + eps))   # (B,) nats

        return H, H.mean().item()
