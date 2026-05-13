import torch
import torch.nn as nn


class ContinuousEncoder(nn.Module):
    """Encoder for continuous state space (floats, not images)."""

    def __init__(self, input_dim=6, d_static=16, d_dynamic=32, d_controllable=16):
        super().__init__()

        self.d_static = d_static
        self.d_dynamic = d_dynamic
        self.d_controllable = d_controllable

        # Shared backbone for continuous inputs
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Separate heads
        self.static_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, d_static))

        self.dynamic_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, d_dynamic))

        self.controllable_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, d_controllable)
        )

    def forward(self, x):
        # x: [B, 6] continuous coordinates
        features = self.shared(x)

        z_static = self.static_head(features)
        z_dynamic = self.dynamic_head(features)
        z_controllable = self.controllable_head(features)

        return {
            "z_static": z_static,
            "z_dynamic": z_dynamic,
            "z_controllable": z_controllable,
            "z_full": torch.cat([z_static, z_dynamic, z_controllable], dim=-1),
        }
