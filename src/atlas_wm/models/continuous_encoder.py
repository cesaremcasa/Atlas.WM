import torch
import torch.nn as nn


class ContinuousEncoder(nn.Module):
    """Encoder for continuous state space (floats, not images).

    AD-2: z_static is split into z_static_immutable (first d_immutable dims,
    architectural passthrough in dynamics) and z_static_slow (remaining dims,
    small residual with drift penalty).
    """

    def __init__(
        self,
        input_dim: int = 6,
        d_static: int = 16,
        d_dynamic: int = 32,
        d_controllable: int = 16,
        d_immutable: int | None = None,
    ):
        super().__init__()

        self.d_static = d_static
        self.d_dynamic = d_dynamic
        self.d_controllable = d_controllable
        self.d_immutable = d_immutable if d_immutable is not None else d_static // 2
        self.d_slow = d_static - self.d_immutable

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.static_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, d_static))
        self.dynamic_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, d_dynamic))
        self.controllable_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, d_controllable)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.shared(x)

        z_static = self.static_head(features)
        z_static_immutable = z_static[:, : self.d_immutable]
        z_static_slow = z_static[:, self.d_immutable :]

        z_dynamic = self.dynamic_head(features)
        z_controllable = self.controllable_head(features)

        return {
            "z_static": z_static,
            "z_static_immutable": z_static_immutable,
            "z_static_slow": z_static_slow,
            "z_dynamic": z_dynamic,
            "z_controllable": z_controllable,
            "z_full": torch.cat([z_static, z_dynamic, z_controllable], dim=-1),
        }
