"""Entity-centric encoder for variable n_objects (Block 10).

Replaces the fixed-6D ContinuousEncoder with a permutation-equivariant
architecture that handles n_objects ∈ [3, 10] (or any range).

Each entity (agent, box_0, ..., box_{n-1}) is embedded independently by
a shared MLP, then pooled into a global context. The static/dynamic/
controllable decomposition (AD-2) is preserved.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EntityEncoder(nn.Module):
    """Permutation-equivariant encoder for multi-object observations.

    Input: [B, n_objects, entity_dim] tensor of per-object features.
    Output: same dict interface as ContinuousEncoder (z_static, z_dynamic,
            z_controllable, z_full, z_static_immutable, z_static_slow).

    The first object is treated as the agent (controllable); the rest are
    boxes (uncontrolled dynamics). Mean pooling is used for the global context.
    """

    def __init__(
        self,
        entity_dim: int = 2,
        d_static: int = 16,
        d_dynamic: int = 32,
        d_controllable: int = 16,
        hidden: int = 64,
        d_immutable: int | None = None,
    ):
        super().__init__()
        self.entity_dim = entity_dim
        self.d_static = d_static
        self.d_dynamic = d_dynamic
        self.d_controllable = d_controllable
        self.d_immutable = d_immutable if d_immutable is not None else d_static // 2
        self.d_slow = d_static - self.d_immutable

        # Per-entity embedding (shared weights across objects)
        self.entity_embed = nn.Sequential(
            nn.Linear(entity_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Global context from mean-pooled entity embeddings
        self.context_embed = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Agent-specific embedding (entity 0)
        self.agent_embed = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Heads operating on (agent_embed + context)
        combined = hidden * 2
        self.static_head = nn.Sequential(
            nn.Linear(combined, hidden), nn.ReLU(), nn.Linear(hidden, d_static)
        )
        self.dynamic_head = nn.Sequential(
            nn.Linear(combined, hidden), nn.ReLU(), nn.Linear(hidden, d_dynamic)
        )
        self.controllable_head = nn.Sequential(
            nn.Linear(combined, hidden), nn.ReLU(), nn.Linear(hidden, d_controllable)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [B, n_objects, entity_dim] — per-object coordinate features.

        Returns:
            Same dict interface as ContinuousEncoder.
        """
        B, n_obj, _ = x.shape

        # Embed all entities with shared weights: [B, n_obj, hidden]
        flat = x.view(B * n_obj, self.entity_dim)
        embedded = self.entity_embed(flat).view(B, n_obj, -1)

        # Global context via mean pooling: [B, hidden]
        context = self.context_embed(embedded.mean(dim=1))

        # Agent embedding (entity 0): [B, hidden]
        agent = self.agent_embed(embedded[:, 0, :])

        # Fuse agent + global context: [B, 2*hidden]
        fused = torch.cat([agent, context], dim=-1)

        z_static = self.static_head(fused)
        z_static_immutable = z_static[:, : self.d_immutable]
        z_static_slow = z_static[:, self.d_immutable :]
        z_dynamic = self.dynamic_head(fused)
        z_controllable = self.controllable_head(fused)

        return {
            "z_static": z_static,
            "z_static_immutable": z_static_immutable,
            "z_static_slow": z_static_slow,
            "z_dynamic": z_dynamic,
            "z_controllable": z_controllable,
            "z_full": torch.cat([z_static, z_dynamic, z_controllable], dim=-1),
        }
