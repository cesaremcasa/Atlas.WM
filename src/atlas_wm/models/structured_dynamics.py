import torch
import torch.nn as nn


class StructuredDynamics(nn.Module):
    """Predicts next latent state with structured transitions.

    AD-2 Hybrid Static Decomposition:
    - z_static_immutable: hard architectural passthrough — identity, no network.
      Guarantees bit-for-bit identical static representation across time steps.
    - z_static_slow: soft residual with small drift allowed; drift is penalized
      in the training loss by lambda_slow_drift * delta_slow.norm().mean().
    """

    def __init__(
        self,
        d_static: int = 16,
        d_dynamic: int = 32,
        d_controllable: int = 16,
        action_dim: int = 8,
        d_immutable: int | None = None,
    ):
        super().__init__()

        self.d_static = d_static
        self.d_dynamic = d_dynamic
        self.d_controllable = d_controllable
        self.d_immutable = d_immutable if d_immutable is not None else d_static // 2
        self.d_slow = d_static - self.d_immutable

        # Slow static: small residual (soft constraint)
        self.static_slow_net = nn.Sequential(
            nn.Linear(self.d_slow, 32), nn.ReLU(), nn.Linear(32, self.d_slow)
        )

        # Dynamic: autonomous evolution
        self.dynamic_net = nn.Sequential(
            nn.Linear(d_dynamic, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, d_dynamic),
        )

        # Controllable: action-conditioned
        self.control_net = nn.Sequential(
            nn.Linear(d_controllable + action_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, d_controllable),
        )

    def forward(
        self, z_dict: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            z_dict: keys z_static_immutable, z_static_slow, z_dynamic, z_controllable.
                    Also accepts legacy z_static (split internally if sub-keys absent).
            action: [B, action_dim]
        """
        if "z_static_immutable" in z_dict:
            z_static_immutable = z_dict["z_static_immutable"]
            z_static_slow = z_dict["z_static_slow"]
        else:
            z_static = z_dict["z_static"]
            z_static_immutable = z_static[:, : self.d_immutable]
            z_static_slow = z_static[:, self.d_immutable :]

        z_dynamic = z_dict["z_dynamic"]
        z_controllable = z_dict["z_controllable"]

        # Immutable: hard passthrough — identity, no network
        z_static_immutable_next = z_static_immutable

        # Slow: soft residual (small drift, penalized by lambda_slow_drift in loss)
        delta_slow = self.static_slow_net(z_static_slow)
        z_static_slow_next = z_static_slow + delta_slow

        # Dynamic: autonomous residual evolution
        delta_dynamic = self.dynamic_net(z_dynamic)
        z_dynamic_next = z_dynamic + delta_dynamic

        # Controllable: action-conditioned residual
        control_input = torch.cat([z_controllable, action], dim=-1)
        delta_controllable = self.control_net(control_input)
        z_controllable_next = z_controllable + delta_controllable

        z_static_next = torch.cat([z_static_immutable_next, z_static_slow_next], dim=-1)

        return {
            "z_static_immutable": z_static_immutable_next,
            "z_static_slow": z_static_slow_next,
            "z_static": z_static_next,
            "z_dynamic": z_dynamic_next,
            "z_controllable": z_controllable_next,
            "z_full": torch.cat([z_static_next, z_dynamic_next, z_controllable_next], dim=-1),
            "delta_slow": delta_slow,
        }
