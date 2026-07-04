"""Evaluate a trained Atlas.WM checkpoint: open-loop rollout error by horizon.

Rolls the world model forward WITHOUT re-encoding (self-fed latents — the
regime a world model is actually used in) and reports observation-space
next-frame MSE at each horizon, plus the immutable-passthrough check.

Until v4 B8 this script loaded a checkpoint, printed "ready for evaluation"
and exited (roadmap finding M5).

Usage::

    python scripts/evaluate.py --checkpoint checkpoints/best_model.safetensors
    python scripts/evaluate.py --checkpoint ... --horizon 10 --split test
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from atlas_wm.checkpointing.dims import infer_dims
from atlas_wm.checkpointing.io import load_checkpoint
from atlas_wm.data.dataset import stack_window
from atlas_wm.data.episode_dataset import EpisodeATLASDataset
from atlas_wm.models.continuous_encoder import ContinuousEncoder
from atlas_wm.models.decoder import Decoder
from atlas_wm.models.structured_dynamics import StructuredDynamics


def load_world_model(
    checkpoint_path: str,
) -> tuple[ContinuousEncoder, StructuredDynamics, Decoder, dict[str, str]]:
    """Load encoder+dynamics+decoder with dims from metadata/weight shapes."""
    state_dict, meta = load_checkpoint(
        checkpoint_path,
        expected_model_class="ContinuousEncoder+StructuredDynamics",
        strict_env=False,
        allow_unsigned=True,
    )
    dims = infer_dims(state_dict)
    d_immutable = int(meta.get("d_immutable", dims["d_immutable"]))
    encoder = ContinuousEncoder(
        input_dim=dims["input_dim"],
        d_static=dims["d_static"],
        d_dynamic=dims["d_dynamic"],
        d_controllable=dims["d_controllable"],
        d_immutable=d_immutable,
    )
    dynamics = StructuredDynamics(
        d_static=dims["d_static"],
        d_dynamic=dims["d_dynamic"],
        d_controllable=dims["d_controllable"],
        action_dim=dims["action_dim"],
        d_immutable=d_immutable,
    )
    decoder = Decoder(
        d_full=dims["d_static"] + dims["d_dynamic"] + dims["d_controllable"],
        output_dim=dims["input_dim"],
    )
    encoder.load_state_dict(
        {k[len("encoder.") :]: v for k, v in state_dict.items() if k.startswith("encoder.")}
    )
    dynamics.load_state_dict(
        {k[len("dynamics.") :]: v for k, v in state_dict.items() if k.startswith("dynamics.")}
    )
    decoder.load_state_dict(
        {k[len("decoder.") :]: v for k, v in state_dict.items() if k.startswith("decoder.")}
    )
    for m in (encoder, dynamics, decoder):
        m.eval()
    return encoder, dynamics, decoder, meta


@torch.no_grad()
def rollout_mse_by_horizon(
    encoder: ContinuousEncoder,
    dynamics: StructuredDynamics,
    decoder: Decoder,
    dataset: EpisodeATLASDataset,
    frame_stack: int,
    horizon: int,
    base_dim: int,
    batch_size: int = 1024,
) -> tuple[list[float], float]:
    """Open-loop rollout over ``horizon`` steps.

    Returns:
        (mse_per_horizon, max_immutable_drift): observation-space next-frame
        MSE at horizons 1..horizon, and the maximum absolute drift of
        ``z_static_immutable`` across the rollout (must be 0 — AD-2's hard
        passthrough).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sums = [0.0] * horizon
    count = 0
    max_imm_drift = 0.0

    for batch in loader:
        obs_w = batch["obs_window"]
        act_w = batch["action_window"]
        inputs = stack_window(obs_w, frame_stack)
        z_cur = encoder(inputs[:, 0])
        z_imm_0 = z_cur["z_static_immutable"]
        for h in range(horizon):
            z_cur = dynamics(z_cur, act_w[:, h + 1])
            sums[h] += (
                nn.functional.mse_loss(
                    decoder(z_cur["z_full"])[:, -base_dim:],
                    inputs[:, h + 1][:, -base_dim:],
                    reduction="sum",
                ).item()
                / base_dim
            )
            drift = (z_cur["z_static_immutable"] - z_imm_0).abs().max().item()
            max_imm_drift = max(max_imm_drift, drift)
        count += len(obs_w)

    return [s / count for s in sums], max_imm_drift


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Atlas.WM checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to .safetensors checkpoint")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split", default="val")
    parser.add_argument("--horizon", type=int, default=10, help="Open-loop rollout length")
    args = parser.parse_args()

    encoder, dynamics, decoder, meta = load_world_model(args.checkpoint)
    frame_stack = int(meta.get("frame_stack", "1"))
    base_dim = encoder.shared[0].weight.shape[1] // frame_stack

    dataset = EpisodeATLASDataset(args.data_dir, split=args.split, window_k=args.horizon + 2)
    mse, imm_drift = rollout_mse_by_horizon(
        encoder, dynamics, decoder, dataset, frame_stack, args.horizon, base_dim
    )

    print(f"\nCheckpoint : {args.checkpoint}")
    print(
        f"objective={meta.get('objective', '?')} frame_stack={frame_stack} "
        f"rollout_k={meta.get('rollout_k', '?')} git_sha={meta.get('git_sha', '?')}"
    )
    print(f"Open-loop rollout on {args.split!r} ({len(dataset)} windows):\n")
    print("  horizon   next-frame MSE (obs space)")
    for h, m in enumerate(mse, start=1):
        print(f"  {h:>4}      {m:.6f}")
    print(f"\n  z_static_immutable max drift over rollout: {imm_drift:.2e} (AD-2: must be 0)")


if __name__ == "__main__":
    main()
