"""Evaluate a trained Atlas.WM checkpoint."""

import argparse

import torch

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

    encoder = ContinuousEncoder(input_dim=6, d_static=16, d_dynamic=32, d_controllable=16).to(
        device
    )
    dynamics = StructuredDynamics(d_static=16, d_dynamic=32, d_controllable=16, action_dim=8).to(
        device
    )

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
