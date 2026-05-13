"""Action-invariance critic for AD-3 identifiability enforcement.

AD-3: z_static_immutable must not carry action-predictive information.
We enforce this adversarially:

  Critic step: minimize action prediction loss from z_static_immutable (detached).
  Encoder step: maximize that same prediction loss (fool the critic).

The critic is a small MLP; its separate optimizer runs before the main backward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionInvarianceCritic(nn.Module):
    """Predicts action from z_static_immutable.

    Trained adversarially so the encoder produces an immutable representation
    that carries no action-predictive information.
    """

    def __init__(self, d_immutable: int, action_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_immutable, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, z_imm: torch.Tensor) -> torch.Tensor:
        return self.net(z_imm)


def critic_loss(
    critic: ActionInvarianceCritic,
    z_imm_detached: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    """MSE loss for the critic: wants to predict action from z_static_immutable.

    Args:
        critic: ActionInvarianceCritic module.
        z_imm_detached: z_static_immutable with .detach() applied (no encoder grad).
        action: [B, action_dim] raw action tensor.
    """
    action_norm = F.normalize(action, dim=-1)
    pred = critic(z_imm_detached)
    return F.mse_loss(pred, action_norm)


def encoder_adversarial_loss(
    critic: ActionInvarianceCritic,
    z_imm: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    """Adversarial loss for the encoder: wants to fool the critic.

    Returns the NEGATIVE of the critic's prediction loss so that minimizing
    this loss makes z_static_immutable less action-predictive.

    Args:
        critic: ActionInvarianceCritic module (params are frozen during this step).
        z_imm: z_static_immutable WITHOUT detach (encoder grads flow).
        action: [B, action_dim] raw action tensor.
    """
    action_norm = F.normalize(action, dim=-1)
    pred = critic(z_imm)
    return -F.mse_loss(pred, action_norm)
