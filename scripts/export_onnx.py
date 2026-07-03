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

# Moved to the package so probe scripts can share it (v4 B6); re-exported
# here for backward compatibility with existing imports.
from atlas_wm.checkpointing.dims import infer_dims  # noqa: E402, F401


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
