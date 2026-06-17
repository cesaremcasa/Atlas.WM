"""ONNX export for Atlas.WM (Block 13).

The world model is exported as two graphs with plain tensor I/O (ONNX cannot
represent the dict-based forward signatures directly):

- **encoder**: ``obs[B, input_dim] -> z_full[B, d_full]``
- **dynamics**: ``(z_full[B, d_full], action[B, action_dim]) -> z_full_next[B, d_full]``

``z_full`` is laid out as ``[z_static | z_dynamic | z_controllable]`` where
``z_static = [z_static_immutable | z_static_slow]`` — the same concatenation the
encoder produces. The dynamics wrapper slices ``z_full`` back into that
structure, so the two graphs compose: feed the encoder's ``z_full`` straight
into dynamics to roll the world model forward.

Requires the optional ``onnx`` dependency (``pip install 'atlas-wm[export]'``).
The legacy TorchScript exporter (``dynamo=False``) is used for a stable graph
with explicit dynamic batch axes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from atlas_wm.models.continuous_encoder import ContinuousEncoder
from atlas_wm.models.structured_dynamics import StructuredDynamics

DEFAULT_OPSET = 17


class EncoderONNX(nn.Module):
    """Tensor-in/tensor-out wrapper: ``obs -> z_full``."""

    def __init__(self, encoder: ContinuousEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z_full: torch.Tensor = self.encoder(obs)["z_full"]
        return z_full


class DynamicsONNX(nn.Module):
    """Tensor-in/tensor-out wrapper: ``(z_full, action) -> z_full_next``.

    Slices ``z_full`` into the structured sub-spaces the dynamics expects using
    the dynamics module's own dimensions, then returns the concatenated next
    state so the graph round-trips with :class:`EncoderONNX`.
    """

    def __init__(self, dynamics: StructuredDynamics):
        super().__init__()
        self.dynamics = dynamics
        self.d_static = int(dynamics.d_static)
        self.d_dynamic = int(dynamics.d_dynamic)
        self.d_controllable = int(dynamics.d_controllable)

    def forward(self, z_full: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        s_end = self.d_static
        d_end = s_end + self.d_dynamic
        c_end = d_end + self.d_controllable
        z_dict = {
            "z_static": z_full[:, :s_end],
            "z_dynamic": z_full[:, s_end:d_end],
            "z_controllable": z_full[:, d_end:c_end],
        }
        z_next: torch.Tensor = self.dynamics(z_dict, action)["z_full"]
        return z_next


def d_full(d_static: int, d_dynamic: int, d_controllable: int) -> int:
    """Width of the concatenated ``z_full`` vector."""
    return d_static + d_dynamic + d_controllable


def export_encoder(
    encoder: ContinuousEncoder,
    path: str,
    input_dim: int = 6,
    opset: int = DEFAULT_OPSET,
) -> None:
    """Export a ContinuousEncoder to ONNX as ``obs -> z_full``."""
    wrapper = EncoderONNX(encoder).eval()
    dummy = torch.zeros(1, input_dim)
    torch.onnx.export(
        wrapper,
        (dummy,),
        path,
        input_names=["obs"],
        output_names=["z_full"],
        dynamic_axes={"obs": {0: "batch"}, "z_full": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )


def export_dynamics(
    dynamics: StructuredDynamics,
    path: str,
    action_dim: int = 8,
    opset: int = DEFAULT_OPSET,
) -> None:
    """Export a StructuredDynamics to ONNX as ``(z_full, action) -> z_full_next``."""
    wrapper = DynamicsONNX(dynamics).eval()
    width = d_full(dynamics.d_static, dynamics.d_dynamic, dynamics.d_controllable)
    dummy_z = torch.zeros(1, width)
    dummy_a = torch.zeros(1, action_dim)
    torch.onnx.export(
        wrapper,
        (dummy_z, dummy_a),
        path,
        input_names=["z_full", "action"],
        output_names=["z_full_next"],
        dynamic_axes={
            "z_full": {0: "batch"},
            "action": {0: "batch"},
            "z_full_next": {0: "batch"},
        },
        opset_version=opset,
        dynamo=False,
    )
