"""
DegradationSelector — entropy-compatible profile mapping.

Bug fixed
──────────
The previous selector used z-score thresholds (−1.5, −0.5, +0.5, +1.5) but
the updated sensor.py (entropy-based, Gal & Ghahramani 2016) returns predictive
entropy H ∈ [0, ln 2] ≈ [0, 0.693] nats, which is ALWAYS non-negative.

With the old thresholds:
  H < −1.5  →  impossible → 'smear' was NEVER selected
  H < −0.5  →  impossible → 'codec' was NEVER selected
  Most H ∈ [0, 0.5] → always 'flatten' or 'noise'
  H > +1.5  →  impossible → 'clean' was NEVER selected

Result: LnL (the strongest augmentation) was never applied; the controller
saw only mild SSI noise regardless of model uncertainty.

Fix: calibrate thresholds to the actual entropy range.

Entropy reference points (binary classifier)
──────────────────────────────────────────────
H = −p·ln(p) − (1−p)·ln(1−p)

  p = 0.99  →  H = 0.056 nats   (very confident)
  p = 0.90  →  H = 0.325 nats   (confident)
  p = 0.80  →  H = 0.500 nats   (moderate uncertainty)
  p = 0.65  →  H = 0.647 nats   (near-maximum confusion)
  p = 0.51  →  H = 0.693 nats   (maximum confusion, ln 2)

Profile assignment
───────────────────
  H < 0.10  →  'smear'    LnL (hardest): model very confident → needs strong challenge
  H < 0.30  →  'codec'    ISD: moderate confidence
  H < 0.50  →  'flatten'  SSI: moderate uncertainty — the nominal training zone
  H < 0.60  →  'noise'    SSI (mild): model fairly confused — gentler augmentation
  H ≥ 0.60  →  'clean'    No augmentation: model near-maximum confusion → preserve signal

This mapping correctly assigns the hardest augmentation (LnL) to the most
confident samples and removes augmentation entirely from the most confused ones,
matching the uncertainty-difficulty inverse relationship in thesis Section 3.5.2.

References
──────────
Gal & Ghahramani (2016): predictive entropy for epistemic uncertainty.
Thesis Table 5, Section 3.5.2.
Tak et al. (2022) ICASSP: RawBoost — defines the augmentation profiles.
"""

import torch


class DegradationSelector:
    """
    Maps per-sample predictive entropy H (nats) to augmentation profiles.

    Entropy thresholds are calibrated to the range H ∈ [0, ln 2 ≈ 0.693]
    returned by the entropy-based UncertaintySensor.
    """

    # Entropy thresholds (nats) — all within [0, ln2=0.693]
    _T_SMEAR   = 0.10    # H < 0.10  →  smear (LnL, hardest)
    _T_CODEC   = 0.30    # H < 0.30  →  codec (ISD)
    _T_FLATTEN = 0.50    # H < 0.50  →  flatten (SSI)
    _T_NOISE   = 0.60    # H < 0.60  →  noise (SSI, mild)
    # H ≥ 0.60           →  clean (no augmentation)

    def select(self, entropy_scores: torch.Tensor) -> list:
        """
        Parameters
        ----------
        entropy_scores : (B,) tensor of per-sample predictive entropy values H.
                         Returned by UncertaintySensor.measure() as the first output.

        Returns
        -------
        list[str] : length B, one profile label per sample.
                    Labels match the keys in DegradationActuator.apply().
        """
        selections = []
        for h in entropy_scores.tolist():
            if h < self._T_SMEAR:
                selections.append("smear")       # very confident → LnL
            elif h < self._T_CODEC:
                selections.append("codec")       # confident → ISD
            elif h < self._T_FLATTEN:
                selections.append("flatten")     # moderate → SSI
            elif h < self._T_NOISE:
                selections.append("noise")       # uncertain → SSI mild
            else:
                selections.append("clean")       # very confused → no aug
        return selections