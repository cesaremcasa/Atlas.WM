"""Export a trained Atlas.WM checkpoint to ONNX (Block 13).

Produces two graphs that compose into a rollout:
    encoder.onnx :  obs[B, 6]              -> z_full[B, 64]
    dynamics.onnx:  (z_full[B, 64], action[B, 8]) -> z_full_next[B, 64]

Usage:
    python scripts/export_onnx.py \
        --checkpoint checkpoints/best_model.safetensors --out-dir export/

Requires the optional onnx dependency:  pip install 'atlas-wm[export]'
"""

from __future__ import annotations

import argparse
import os

from atlas_wm.checkpointing.io import load_checkpoint
from atlas_wm.export.onnx_export import export_dynamics, export_encoder
from atlas_wm.models.continuous_encoder import ContinuousEncoder
from atlas_wm.models.structured_dynamics import StructuredDynamics

MODEL_CLASS = "ContinuousEncoder+StructuredDynamics"


def infer_dims(state_dict: dict) -> dict[str, int]:
    """Infer every architecture dimension from state-dict weight shapes.

    ``d_immutable`` is recovered via ``dynamics.static_slow_net.0.weight``
    (input width = d_slow, so d_immutable = d_static − d_slow). Previously it
    was not inferred at all: a checkpoint trained with a non-default split
    exported with the passthrough boundary at ``d_static // 2``, silently
    placing mutable dims inside the "immutable" slice (v4 B3, finding H6).
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Atlas.WM checkpoint to ONNX")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.safetensors")
    parser.add_argument("--out-dir", default="export")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    state_dict, metadata = load_checkpoint(
        args.checkpoint,
        expected_model_class=MODEL_CLASS,
        strict_env=False,
        allow_unsigned=True,
    )

    dims = infer_dims(state_dict)

    encoder = ContinuousEncoder(
        input_dim=dims["input_dim"],
        d_static=dims["d_static"],
        d_dynamic=dims["d_dynamic"],
        d_controllable=dims["d_controllable"],
        d_immutable=dims["d_immutable"],
    )
    dynamics = StructuredDynamics(
        d_static=dims["d_static"],
        d_dynamic=dims["d_dynamic"],
        d_controllable=dims["d_controllable"],
        action_dim=dims["action_dim"],
        d_immutable=dims["d_immutable"],
    )
    encoder.load_state_dict(
        {k[len("encoder.") :]: v for k, v in state_dict.items() if k.startswith("encoder.")}
    )
    dynamics.load_state_dict(
        {k[len("dynamics.") :]: v for k, v in state_dict.items() if k.startswith("dynamics.")}
    )
    encoder.eval()
    dynamics.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    encoder_path = os.path.join(args.out_dir, "encoder.onnx")
    dynamics_path = os.path.join(args.out_dir, "dynamics.onnx")

    export_encoder(encoder, encoder_path, input_dim=dims["input_dim"], opset=args.opset)
    export_dynamics(dynamics, dynamics_path, action_dim=dims["action_dim"], opset=args.opset)

    print(f"Exported encoder  -> {encoder_path}")
    print(f"Exported dynamics -> {dynamics_path}")
    print(f"Source checkpoint git_sha={metadata.get('git_sha')} opset={args.opset}")


if __name__ == "__main__":
    main()
