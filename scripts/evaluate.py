"""Evaluate a trained Atlas.WM checkpoint."""

import argparse

import torch

from atlas_wm.checkpointing.dims import infer_dims
from atlas_wm.checkpointing.io import load_checkpoint
from atlas_wm.models.continuous_encoder import ContinuousEncoder
from atlas_wm.models.structured_dynamics import StructuredDynamics


def evaluate(args: argparse.Namespace) -> None:
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict, metadata = load_checkpoint(
        args.checkpoint,
        expected_model_class="ContinuousEncoder+StructuredDynamics",
        strict_env=False,
        allow_unsigned=True,
    )
    print(f"Loaded: {metadata}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dims from metadata when present, weight shapes otherwise — never
    # hardcoded (H6: a wrong d_immutable mis-slices the latent silently).
    dims = infer_dims(state_dict)
    d_immutable = int(metadata.get("d_immutable", dims["d_immutable"]))
    encoder = ContinuousEncoder(
        input_dim=dims["input_dim"],
        d_static=dims["d_static"],
        d_dynamic=dims["d_dynamic"],
        d_controllable=dims["d_controllable"],
        d_immutable=d_immutable,
    ).to(device)
    dynamics = StructuredDynamics(
        d_static=dims["d_static"],
        d_dynamic=dims["d_dynamic"],
        d_controllable=dims["d_controllable"],
        action_dim=dims["action_dim"],
        d_immutable=d_immutable,
    ).to(device)

    enc_sd = {k[len("encoder.") :]: v for k, v in state_dict.items() if k.startswith("encoder.")}
    dyn_sd = {k[len("dynamics.") :]: v for k, v in state_dict.items() if k.startswith("dynamics.")}

    encoder.load_state_dict(enc_sd)
    dynamics.load_state_dict(dyn_sd)

    encoder.eval()
    dynamics.eval()
    print("Models loaded and ready for evaluation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Atlas.WM checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to .safetensors checkpoint")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
