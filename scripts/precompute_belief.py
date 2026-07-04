"""Precompute causal per-row physics beliefs for a processed dataset (v4 B12).

Runs the trained PhysicsBeliefEncoder causally over every episode and stores
one belief vector per transition row ({split}_belief.npy). Rows before the
features warm up (first 2 frames of an episode) get zeros.

The world model's single-frame encoder provably cannot identify episode
physics; conditioning its z_static_slow on these beliefs is the RMA phase-2
integration the v3.x design promised.

Usage::

    python scripts/precompute_belief.py \
        --belief-checkpoint checkpoints/belief_active.safetensors \
        --data-dir data_active/processed
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from atlas_wm.checkpointing.io import load_checkpoint
from atlas_wm.data.dataset import DEFAULT_OBS_SCALE, reject_legacy_normalized
from atlas_wm.data.dynamics_features import build_dynamics_features
from atlas_wm.models.physics_belief import PhysicsBeliefEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()

    reject_legacy_normalized(args.data_dir)
    state_dict, meta = load_checkpoint(
        args.belief_checkpoint,
        expected_model_class="PhysicsBeliefEncoder",
        strict_env=False,
        allow_unsigned=True,
    )
    if meta.get("features") != "dynamics_v1":
        raise SystemExit("belief checkpoint must be a v4 B10+ (dynamics_v1 features) model")
    enc = PhysicsBeliefEncoder(
        obs_dim=int(meta["gru_input_dim"]), d_slow=int(meta["d_slow"]), hidden_dim=128
    )
    enc.load_state_dict(
        {k[len("belief_enc.") :]: v for k, v in state_dict.items() if k.startswith("belief_enc.")}
    )
    enc.eval()

    for split in ("train", "val", "test"):
        obs = np.load(f"{args.data_dir}/{split}_obs.npy").astype(np.float32) / DEFAULT_OBS_SCALE
        acts = np.load(f"{args.data_dir}/{split}_actions.npy").astype(np.float32)
        ids = np.load(f"{args.data_dir}/{split}_episode_ids.npy")
        out = np.zeros((len(obs), int(meta["d_slow"])), dtype=np.float32)
        with torch.no_grad():
            for e in np.unique(ids):
                rows = np.where(ids == e)[0]
                if len(rows) < 3:
                    continue
                ow = torch.from_numpy(obs[rows]).unsqueeze(0)
                aw = torch.from_numpy(acts[rows]).unsqueeze(0)
                feats = build_dynamics_features(ow, aw)  # [1, L-2, F]
                z_seq = enc.forward_sequence(feats).squeeze(0).numpy()  # [L-2, d]
                # feats step s describes transition ending at row s+2 —
                # belief for row r uses only frames <= r (causal).
                out[rows[2:]] = z_seq
        np.save(f"{args.data_dir}/{split}_belief.npy", out)
        print(f"{split}: wrote {out.shape} beliefs")


if __name__ == "__main__":
    main()
