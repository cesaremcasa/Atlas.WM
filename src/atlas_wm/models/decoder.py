"""Observation decoder for Atlas.WM — reconstructs obs from z_full.

Used during training to prevent encoder collapse: a decoder that must
reconstruct the raw observation forces the encoder to retain information,
making the trivial all-zero latent solution impossible.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """MLP decoder: ``z_full[B, d_full] → obs_hat[B, output_dim]``."""

    def __init__(self, d_full: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_full, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, z_full: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(z_full)
        return out
