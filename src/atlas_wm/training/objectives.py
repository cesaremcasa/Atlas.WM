"""Stable self-predictive training objectives (v4 B7, roadmap finding H1).

The v3.x objective predicted the online encoder's own detached output and
anchored the latent scale with an L2 penalty — a band-aid over a known
degenerate family (scale runaway / collapse). Two principled replacements:

- **EMA target** (``objective: ema``, default): the prediction target comes
  from an exponential-moving-average copy of the encoder
  (BYOL/SPR/TD-MPC2 lineage). The lagged target removes the runaway
  direction structurally: inflating the online representation is penalized
  against a target that has not inflated yet.
- **VICReg regularization** (``objective: vicreg``): keeps the online
  detached target but adds a per-dimension variance hinge (anti-collapse)
  and an off-diagonal covariance penalty (anti-redundancy) on the encoder
  output — the tensor that can actually collapse. (The v3.x variance
  penalty was applied to the dynamics output instead, and fought the L2
  term head-on.)

Both retire ``lambda_latent_l2``. Empirical motivation: under the old
recipe the trained 2-frame world model scored 6× WORSE than a linear ridge
on observation-space next-frame MSE (see MODEL_CARD, v4 B6).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


def make_ema_target(model: nn.Module) -> nn.Module:
    """Create a frozen deep copy of ``model`` to serve as the EMA target."""
    target = copy.deepcopy(model)
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def ema_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    """In-place EMA update: ``target ← tau * target + (1 − tau) * online``."""
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.mul_(tau).add_(p_o.detach(), alpha=1.0 - tau)
    for b_t, b_o in zip(target.buffers(), online.buffers()):
        b_t.copy_(b_o)


def vicreg_regularizer(z: torch.Tensor, gamma: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg-style variance and covariance terms for a latent batch.

    Args:
        z: [B, d] latent batch (the ENCODER output — the tensor at risk of
           collapse; regularizing the dynamics output instead lets the
           encoder collapse while dynamics inflates spread).
        gamma: target per-dimension standard deviation.

    Returns:
        (variance_loss, covariance_loss). Variance hinges at ``gamma``
        (only under-dispersion is penalized); covariance is the mean squared
        off-diagonal entry of the batch covariance, which also anchors the
        overall scale (it grows ~scale⁴, replacing the L2 crutch's role
        against runaway).
    """
    if z.shape[0] < 2:
        zero = z.sum() * 0.0
        return zero, zero
    std = torch.sqrt(z.var(dim=0) + 1e-8)
    variance_loss = torch.relu(gamma - std).mean()
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / (z.shape[0] - 1)
    d = z.shape[1]
    off_diag_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    covariance_loss = off_diag_sq / d
    return variance_loss, covariance_loss
