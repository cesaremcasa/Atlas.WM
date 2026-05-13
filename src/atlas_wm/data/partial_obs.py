"""Partial observability transform: keep only the K nearest objects (Block 11).

In a full observation [B, n_obj, entity_dim], entity 0 is always the agent.
This module retains the agent and the K-1 nearest objects by Euclidean distance
to the agent position (assumed to be the first 2 dims of each entity).

Used as a data augmentation / wrapper during training to force the model
to generalize under incomplete information.
"""

from __future__ import annotations

import torch


def nearest_k_obs(
    x: torch.Tensor,
    k: int,
    agent_idx: int = 0,
    pos_dims: slice = slice(0, 2),
) -> torch.Tensor:
    """Return [B, k, entity_dim] keeping agent + (k-1) nearest objects.

    Args:
        x: [B, n_obj, entity_dim] full observation tensor.
        k: Number of objects to keep (including the agent). Must be ≤ n_obj.
        agent_idx: Index of the agent entity (default 0).
        pos_dims: Slice of entity_dim used as 2D position for distance computation.

    Returns:
        [B, k, entity_dim] tensor with agent at index 0 and k-1 nearest objects.
    """
    B, n_obj, _ = x.shape
    if k >= n_obj:
        return x
    if k < 1:
        raise ValueError(f"k must be ≥ 1, got {k}")

    agent_pos = x[:, agent_idx, pos_dims]  # [B, 2]
    non_agent_idx = [i for i in range(n_obj) if i != agent_idx]
    others = x[:, non_agent_idx, :]  # [B, n_obj-1, entity_dim]
    others_pos = others[:, :, pos_dims]  # [B, n_obj-1, 2]

    dists = (others_pos - agent_pos.unsqueeze(1)).norm(dim=-1)  # [B, n_obj-1]
    k_nearest = k - 1
    _, idx = dists.topk(k_nearest, dim=-1, largest=False)  # [B, k-1]

    # Gather nearest non-agent entities
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # [B, k-1, entity_dim]
    nearest = others.gather(dim=1, index=idx_expanded)  # [B, k-1, entity_dim]

    # Agent always first
    agent = x[:, agent_idx : agent_idx + 1, :]  # [B, 1, entity_dim]
    return torch.cat([agent, nearest], dim=1)  # [B, k, entity_dim]


class PartialObsWrapper:
    """Callable wrapper that applies nearest_k_obs with a fixed k.

    Example usage in a DataLoader collate_fn or as a transform::

        transform = PartialObsWrapper(k=3)
        x_partial = transform(x_full)  # [B, 3, entity_dim]
    """

    def __init__(self, k: int, agent_idx: int = 0):
        self.k = k
        self.agent_idx = agent_idx

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return nearest_k_obs(x, k=self.k, agent_idx=self.agent_idx)
