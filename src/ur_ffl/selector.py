import torch


class DegradationSelector:
    """
    Maps MC-dropout uncertainty z-scores to codec augmentation profiles.

    Profile   z            Technique          Research basis
    ───────── ──────────── ─────────────────  ──────────────────────────────────
    smear     z < -1.5     MP3 codec          Top-5 DF 2021 systems; Das 2021
    codec     -1.5 to -0.5 OGG/Vorbis         Complementary to MP3 (Shim 2024)
    flatten   -0.5 to +0.5 A-law + phone BW   G.711/G.722 best combo (Das 2021)
    noise      +0.5 to +1.5 SSI pink noise    RawBoost algo 3 (Tak 2022)
    clean      z > +1.5    No change          Retains unmodified samples
    """

    def select(self, z_u_scores: torch.Tensor) -> list:
        selections = []
        for zu in z_u_scores.tolist():
            if zu < -1.5:
                selections.append("smear")
            elif zu < -0.5:
                selections.append("codec")
            elif zu < 0.5:
                selections.append("flatten")
            elif zu < 1.5:
                selections.append("noise")
            else:
                selections.append("clean")
        return selections