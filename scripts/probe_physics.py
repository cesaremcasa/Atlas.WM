"""Probe a trained encoder for variable-physics identifiability (Block 12).

Loads a trained ContinuousEncoder, encodes a variable-physics dataset, and fits
linear probes from each latent sub-space to the ground-truth physics parameters.

AD-2 prediction: ``z_static_slow`` should achieve high R² (variable physics live
there), while ``z_static_immutable`` should achieve near-zero R² (it is a hard
passthrough and must not encode episode-varying quantities).

Usage:
    python scripts/probe_physics.py \
        --checkpoint checkpoints/best_model.safetensors \
        --data-dir data/processed --split val
"""

from __future__ import annotations

import argparse

import numpy as np

from atlas_wm.checkpointing.io import load_checkpoint
from atlas_wm.eval.latent_probe import probe_latent
from atlas_wm.models.continuous_encoder import ContinuousEncoder

TARGET_NAMES = ["gravity", "friction_agent", "friction_box"]


def _load_encoder(checkpoint_path: str) -> ContinuousEncoder:
    state_dict, _ = load_checkpoint(
        checkpoint_path,
        expected_model_class="ContinuousEncoder+StructuredDynamics",
        strict_env=False,
        allow_unsigned=True,
    )
    encoder_state = {
        k[len("encoder.") :]: v for k, v in state_dict.items() if k.startswith("encoder.")
    }
    encoder = ContinuousEncoder(input_dim=6, d_static=16, d_dynamic=32, d_controllable=16)
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    return encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe encoder for physics identifiability")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.safetensors")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split", default="val")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

    obs = np.load(f"{args.data_dir}/{args.split}_obs.npy")
    physics = np.load(f"{args.data_dir}/{args.split}_physics.npy")

    encoder = _load_encoder(args.checkpoint)

    print(f"Probing {len(obs)} samples from split={args.split!r}\n")
    for latent_key in ("z_static_slow", "z_static_immutable", "z_dynamic"):
        result = probe_latent(
            encoder,
            obs,
            physics,
            latent_key=latent_key,
            target_names=TARGET_NAMES,
            alpha=args.ridge_alpha,
            train_frac=args.train_frac,
        )
        print(result)
        print()

    print(
        "Expectation (AD-2): R²[z_static_slow] high, R²[z_static_immutable] ~0.\n"
        "A passthrough sub-space that encodes variable physics is a decomposition leak."
    )


if __name__ == "__main__":
    main()
