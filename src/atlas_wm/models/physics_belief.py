"""Physics belief encoder: GRU over K observations → z_static_slow.

A single-timestep MLP encoder cannot infer episode-level physics (gravity,
friction) from position-only snapshots — physics only reveal themselves through
dynamics (how positions change across time).

Architecture follows the RMA / VariBAD pattern: a recurrent encoder accumulates
K consecutive observations before producing a stable z_static_slow that encodes
episode-level physical constants.

References:
    - RMA: Rapid Motor Adaptation (Kumar et al. 2021, arXiv:2107.04034)
    - VariBAD (Zintgraf et al. 2020, JMLR 2021)
    - ContraBAR (Choshen et al. ICML 2023)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsBeliefEncoder(nn.Module):
    """GRU-based encoder: obs_window[B, K, obs_dim] -> z_static_slow[B, d_slow].

    Accumulates K consecutive observations from the same episode before
    producing a representation of the episode-level physics context.
    Minimum K=5 for basic physics identification; K=10-20 for reliable
    gravity/friction estimation from position-only snapshots.
    """

    def __init__(self, obs_dim: int, d_slow: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, d_slow)

    def forward(self, obs_window: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_window: [B, K, obs_dim] — K consecutive observations

        Returns:
            z_static_slow: [B, d_slow]
        """
        _, h_n = self.gru(obs_window)  # h_n: [1, B, hidden_dim]
        return self.proj(h_n.squeeze(0))  # [B, d_slow]


class PhysicsHead(nn.Module):
    """Supervised head: z_static_slow[B, d_slow] -> physics_hat[B, n_physics].

    Used during training as an auxiliary loss to steer z_static_slow toward
    encoding physical parameters. Not used at inference time.
    """

    def __init__(self, d_slow: int, n_physics: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(d_slow, n_physics)

    def forward(self, z_static_slow: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.linear(z_static_slow)
        return out
