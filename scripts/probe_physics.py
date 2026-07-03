"""Probe a trained encoder for variable-physics identifiability (Block 12).

Loads a trained ContinuousEncoder, encodes a variable-physics dataset, and fits
linear probes from each latent sub-space to the ground-truth physics parameters.

AD-2 prediction: ``z_static_slow`` should achieve higher R² than
``z_static_immutable`` (variable physics live in the slow residual, while the
immutable passthrough must not encode episode-varying quantities). All three
parameters are probed; the v3.x exclusion of ``friction_agent`` was retracted
in v4 (see docs/MODEL_CARD.md and scripts/oracle_friction_agent.py).

Also supports probing the PhysicsBeliefEncoder (GRU) via --belief-checkpoint.

Usage:
    # Single-step encoder probe (baseline)
    python scripts/probe_physics.py \
        --checkpoint checkpoints/best_model.safetensors \
        --data-dir data/processed --split val

    # Belief encoder probe (expected high R²)
    python scripts/probe_physics.py \
        --checkpoint checkpoints/best_model.safetensors \
        --belief-checkpoint checkpoints/physics_belief.safetensors \
        --data-dir data/processed --split val
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from atlas_wm.checkpointing.dims import infer_dims
from atlas_wm.checkpointing.io import load_checkpoint
from atlas_wm.eval.latent_probe import probe_latent
from atlas_wm.models.continuous_encoder import ContinuousEncoder

# Full column order of the physics array (set by generate_data.py).
TARGET_NAMES = ["gravity", "friction_agent", "friction_box"]
# Fallback belief targets for checkpoints without physics_keys metadata.
# v4: the full set — the friction_agent exclusion was retracted (MODEL_CARD).
BELIEF_TARGET_KEYS = ["gravity", "friction_agent", "friction_box"]


def _load_encoder(checkpoint_path: str) -> tuple[ContinuousEncoder, dict[str, str]]:
    state_dict, meta = load_checkpoint(
        checkpoint_path,
        expected_model_class="ContinuousEncoder+StructuredDynamics",
        strict_env=False,
        allow_unsigned=True,
    )
    encoder_state = {
        k[len("encoder.") :]: v for k, v in state_dict.items() if k.startswith("encoder.")
    }
    # Dims from metadata when present, weight shapes otherwise. Hardcoding
    # them silently mis-sliced z_static_immutable/z_static_slow for
    # non-default splits — load_state_dict cannot catch it because
    # d_immutable only affects slicing, not parameter shapes (H6).
    dims = infer_dims(state_dict)
    encoder = ContinuousEncoder(
        input_dim=dims["input_dim"],
        d_static=dims["d_static"],
        d_dynamic=dims["d_dynamic"],
        d_controllable=dims["d_controllable"],
        d_immutable=int(meta.get("d_immutable", dims["d_immutable"])),
    )
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    return encoder, meta


def _probe_belief_encoder(
    belief_checkpoint: str,
    data_dir: str,
    split: str,
    ridge_alpha: float,
    train_frac: float,
) -> None:
    from atlas_wm.data.episode_dataset import EpisodeATLASDataset
    from atlas_wm.eval.latent_probe import probe_from_arrays
    from atlas_wm.models.physics_belief import PhysicsBeliefEncoder

    state_dict, meta = load_checkpoint(
        belief_checkpoint,
        expected_model_class="PhysicsBeliefEncoder",
        strict_env=False,
        allow_unsigned=True,
    )
    obs_dim: int = int(meta.get("obs_dim", "6"))
    action_dim_meta: int = int(meta.get("action_dim", "0"))
    # Derive gru_input_dim from obs_dim + action_dim when not stored explicitly,
    # so obs+action checkpoints load with the correct input size.
    gru_input_dim: int = int(meta.get("gru_input_dim", str(obs_dim + action_dim_meta)))
    d_slow: int = int(meta["d_slow"])
    window_k: int = int(meta["window_k"])

    belief_enc = PhysicsBeliefEncoder(obs_dim=gru_input_dim, d_slow=d_slow, hidden_dim=128)
    belief_enc_state = {
        k[len("belief_enc.") :]: v for k, v in state_dict.items() if k.startswith("belief_enc.")
    }
    belief_enc.load_state_dict(belief_enc_state)
    belief_enc.eval()

    ids_path = os.path.join(data_dir, f"{split}_episode_ids.npy")
    if not os.path.exists(ids_path):
        print(
            f"WARNING: episode IDs not found at {ids_path}. "
            "Re-generate data with generate_data.py + split_data.py."
        )
        return

    try:
        ds = EpisodeATLASDataset(data_dir, split=split, window_k=window_k)
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        return

    if ds.physics is None:
        print("WARNING: no physics labels in dataset — skipping belief probe")
        return

    use_actions = action_dim_meta > 0
    use_velocity = meta.get("use_velocity", "false") == "true"
    all_z, all_physics, all_eps = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            obs_w = item["obs_window"].unsqueeze(0)  # [1, K, obs_dim]
            parts = [obs_w]
            if use_velocity:
                vel_w = torch.zeros_like(obs_w)
                vel_w[:, 1:] = obs_w[:, 1:] - obs_w[:, :-1]
                parts.append(vel_w)
            if use_actions:
                parts.append(item["action_window"].unsqueeze(0))
            z = belief_enc(torch.cat(parts, dim=-1))
            all_z.append(z.squeeze(0).numpy())
            all_physics.append(item["physics"].numpy())
            all_eps.append(int(ds.episode_ids[int(ds.valid_indices[i])]))

    z_arr = np.stack(all_z)
    physics_arr = np.stack(all_physics)  # [N, 3] full physics (gravity, f_agent, f_box)
    eps_arr = np.array(all_eps)

    # Select the target columns the encoder was actually trained on.
    target_keys = json.loads(meta.get("physics_keys", json.dumps(BELIEF_TARGET_KEYS)))
    target_idx = [TARGET_NAMES.index(k) for k in target_keys]
    physics_arr = physics_arr[:, target_idx]

    print(f"\n── PhysicsBeliefEncoder probe (window_k={window_k}, {len(ds)} windows) ──")
    print(f"   targets={target_keys} | episode-grouped split over {len(np.unique(eps_arr))} eps")
    result = probe_from_arrays(
        z_arr,
        physics_arr,
        target_names=target_keys,
        latent_key="z_static_slow (GRU)",
        alpha=ridge_alpha,
        train_frac=train_frac,
        episode_ids=eps_arr,
    )
    print(result)
    print(
        "Expectation: positive R² once converged.\n"
        "(single-step encoder gives R²≈0 — physics are only observable through dynamics)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe encoder for physics identifiability")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.safetensors")
    parser.add_argument(
        "--belief-checkpoint",
        default=None,
        help=(
            "Optional PhysicsBeliefEncoder checkpoint (.safetensors) — "
            "produced by train_physics_belief.py"
        ),
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split", default="val")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

    from atlas_wm.data.dataset import ATLASDataset

    physics_path = f"{args.data_dir}/{args.split}_physics.npy"
    if not os.path.exists(physics_path):
        print(f"ERROR: physics labels not found at {physics_path}")
        print("This probe requires a variable-physics dataset (--randomize-physics).")
        return

    physics = np.load(physics_path)
    encoder, meta = _load_encoder(args.checkpoint)

    # Feed the probe the same view the encoder was trained on: ATLASDataset
    # applies the in-memory scaling (v4 B2) and, when the checkpoint was
    # trained with frame stacking (v4 B6), the same stacked frames. Row count
    # and order are unchanged by stacking, so physics labels stay aligned.
    frame_stack = int(meta.get("frame_stack", "1"))
    ds = ATLASDataset(args.data_dir, split=args.split, frame_stack=frame_stack)
    obs = ds.observations

    # Episode-grouped probe split when episode IDs exist (v4 B5, finding M3).
    eps_path = f"{args.data_dir}/{args.split}_episode_ids.npy"
    episode_ids = np.load(eps_path) if os.path.exists(eps_path) else None

    print(f"── Single-step encoder probe (baseline, {len(obs)} samples) ──")
    print(f"   split={args.split!r}, checkpoint={args.checkpoint}\n")
    for latent_key in ("z_static_slow", "z_static_immutable", "z_dynamic"):
        result = probe_latent(
            encoder,
            obs,
            physics,
            latent_key=latent_key,
            target_names=TARGET_NAMES,
            alpha=args.ridge_alpha,
            train_frac=args.train_frac,
            episode_ids=episode_ids,
        )
        print(result)
        print()

    print(
        "Expectation (AD-2): R²[z_static_slow] high, R²[z_static_immutable] ~0.\n"
        "NOTE: single-step MLP encoder gives R²≈0 by design — physics require\n"
        "temporal context. Use PhysicsBeliefEncoder (--belief-checkpoint) for\n"
        "proper physics identification."
    )

    if args.belief_checkpoint:
        _probe_belief_encoder(
            args.belief_checkpoint,
            args.data_dir,
            args.split,
            args.ridge_alpha,
            args.train_frac,
        )


if __name__ == "__main__":
    main()
