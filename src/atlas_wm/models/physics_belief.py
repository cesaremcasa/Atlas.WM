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
        out: torch.Tensor = self.proj(h_n.squeeze(0))
        return out  # [B, d_slow]

    def forward_sequence(self, obs_window: torch.Tensor) -> torch.Tensor:
        """Causal per-step beliefs: z_t uses only inputs up to t (v4 B12).

        Returns [B, K, d_slow] — the GRU hidden state projected at every
        step, so downstream conditioning never sees the future.
        """
        h_all, _ = self.gru(obs_window)  # [B, K, hidden]
        out: torch.Tensor = self.proj(h_all)
        return out


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


class DistributionalPhysicsHead(nn.Module):
    """Heteroscedastic head: z_slow[B, d] -> (mu[B, n], logvar[B, n]) (v4 B10).

    The belief over episode physics should carry uncertainty — windows with
    little excitation identify parameters poorly, and a point estimate hides
    that (CRAFT / Phys2Real direction). Train with :func:`gaussian_nll`.
    """

    def __init__(self, d_slow: int, n_physics: int = 3) -> None:
        super().__init__()
        self.mu = nn.Linear(d_slow, n_physics)
        self.logvar = nn.Linear(d_slow, n_physics)

    def forward(self, z_static_slow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu(z_static_slow), self.logvar(z_static_slow).clamp(-8.0, 4.0)


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Gaussian negative log-likelihood (constant terms dropped)."""
    return 0.5 * (logvar + (target - mu).pow(2) / logvar.exp()).mean()
