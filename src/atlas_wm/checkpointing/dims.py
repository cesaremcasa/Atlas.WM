"""Infer world-model architecture dimensions from checkpoint weight shapes.

Single source of truth for reconstructing a ContinuousEncoder +
StructuredDynamics pair from a combined state dict (used by ONNX export and
the physics probes). Prefers explicit metadata when the caller has it;
weight-shape inference covers older checkpoints.
"""

from __future__ import annotations


def infer_dims(state_dict: dict) -> dict[str, int]:
    """Infer every architecture dimension from state-dict weight shapes.

    ``d_immutable`` is recovered via ``dynamics.static_slow_net.0.weight``
    (input width = d_slow, so d_immutable = d_static − d_slow). Not inferring
    it exported non-default checkpoints with the passthrough boundary at
    ``d_static // 2``, silently placing mutable dims inside the "immutable"
    slice (v4 B3, roadmap finding H6).
    """
    d_static = int(state_dict["encoder.static_head.2.weight"].shape[0])
    d_slow = int(state_dict["dynamics.static_slow_net.0.weight"].shape[1])
    d_controllable = int(state_dict["encoder.controllable_head.2.weight"].shape[0])
    return {
        "input_dim": int(state_dict["encoder.shared.0.weight"].shape[1]),
        "d_static": d_static,
        "d_immutable": d_static - d_slow,
        "d_dynamic": int(state_dict["encoder.dynamic_head.2.weight"].shape[0]),
        "d_controllable": d_controllable,
        "action_dim": int(state_dict["dynamics.control_net.0.weight"].shape[1]) - d_controllable,
    }
