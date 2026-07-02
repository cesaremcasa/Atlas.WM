"""Train the PhysicsBeliefEncoder — a GRU that infers episode-level physics.

A single-timestep MLP encoder (ContinuousEncoder) cannot infer gravity or
friction from a single position snapshot: physics only manifest in dynamics.
This script trains a separate GRU-based belief encoder that accumulates K
consecutive (obs, action) pairs from the same episode, then predicts physics
parameters (gravity, friction_agent, friction_box) via a supervised auxiliary loss.

Architecture: RMA / VariBAD pattern (multi-step recurrent + supervised distillation).
Key design choices:
  - GRU input: concat(obs_t, Δobs_t, action_t) at each step (6D + 6D + 8D = 20D).
    Physics can only be identified from DYNAMICS (how obs changes given action),
    not from observations alone; the Δobs velocity proxy exposes momentum changes.
  - Targets the RECOVERABLE physics subset {gravity, friction_box}. friction_agent
    is excluded — it is not identifiable under this env+policy (the agent is
    force-actuated every step, masking friction decay). See docs/MODEL_CARD.md.
  - Physics targets are standardized (zero mean, unit variance per parameter)
    to equalize gradient contributions across gravity (2-8) and friction (0.95-0.995).

Usage::

    # Variable-physics dataset required (episode_ids + physics labels)
    python scripts/train_physics_belief.py
    python scripts/train_physics_belief.py --config configs/experiments/v3_variable_physics.yaml
    python scripts/train_physics_belief.py --window-k 20 --epochs 100

Requires:
    data/processed/{split}_episode_ids.npy  (run generate_data.py + split_data.py first)
    data/processed/{split}_physics.npy      (requires --randomize-physics in generate_data.py)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from atlas_wm.checkpointing.io import make_metadata, save_checkpoint
from atlas_wm.data.episode_dataset import EpisodeATLASDataset
from atlas_wm.models.physics_belief import PhysicsBeliefEncoder, PhysicsHead
from atlas_wm.utils.seeding import seed_worker, set_seed

_BASE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")

# Column order of physics_params.npy (set by generate_data.py).
ALL_PHYSICS_KEYS = ["gravity", "friction_agent", "friction_box"]

# Identification targets. friction_agent is EXCLUDED: under the current
# environment + random-exploration regime it is not identifiable. The agent
# receives a 0.8 force impulse every step, which re-randomizes its velocity and
# masks the friction decay entirely — validated with an oracle probe (a large
# MLP on privileged hand-crafted dynamics features) that still scores R² < 0 for
# friction_agent, versus R² ≈ 0.15–0.45 for gravity and friction_box. We
# therefore identify only the recoverable parameters; see docs/MODEL_CARD.md.
PHYSICS_KEYS = ["gravity", "friction_box"]
TARGET_IDX = [ALL_PHYSICS_KEYS.index(k) for k in PHYSICS_KEYS]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if k == "_base":
            continue
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    base_ref: str | None = cfg.get("_base")
    if base_ref:
        base_path = os.path.normpath(os.path.join(os.path.dirname(path), base_ref))
        with open(base_path) as f:
            base: dict[str, Any] = yaml.safe_load(f)
        cfg = _deep_merge(base, cfg)
    return cfg


def _r2_per_target(gt: np.ndarray, hat: np.ndarray) -> np.ndarray:
    ss_res = ((gt - hat) ** 2).sum(axis=0)
    ss_tot = ((gt - gt.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def train_belief_encoder(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    tcfg = cfg["training"]

    # Seed before model construction so the run is reproducible (v4 B3, AD-7).
    seed: int = args.seed if getattr(args, "seed", None) is not None else tcfg.get("seed", 42)
    data_generator = set_seed(seed)

    data_dir: str = dcfg["data_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Seed: {seed}")

    window_k: int = args.window_k or cfg.get("belief_encoder", {}).get("window_k", 10)
    obs_dim: int = mcfg["input_dim"]
    action_dim: int = cfg["environment"]["action_space_size"]
    # Input: concat(obs, Δobs, action) — velocity proxy gives GRU direct access to
    # momentum changes caused by gravity/friction, which are invisible from positions alone.
    gru_input_dim: int = obs_dim * 2 + action_dim  # obs(6) + vel(6) + action(8) = 20
    d_slow: int = mcfg["d_static_slow"]
    n_physics = len(PHYSICS_KEYS)

    try:
        train_ds = EpisodeATLASDataset(data_dir, split="train", window_k=window_k)
        val_ds = EpisodeATLASDataset(data_dir, split="val", window_k=window_k)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(
            "\nTo generate the required data:\n"
            "  python scripts/generate_data.py --randomize-physics --seed 42\n"
            "  python scripts/split_data.py\n"
        )
        return

    if train_ds.physics is None:
        print("ERROR: physics labels not found in dataset.")
        print("Re-generate with: python scripts/generate_data.py --randomize-physics --seed 42")
        return

    # Compute physics normalization statistics from training data (standardize to μ=0, σ=1).
    # Select only the recoverable target columns (TARGET_IDX) from the full 3-column array.
    physics_train = train_ds.physics[train_ds.valid_indices][:, TARGET_IDX]  # [N_valid, n_targets]
    physics_mean = physics_train.mean(axis=0)
    physics_std = physics_train.std(axis=0) + 1e-8
    print(f"Physics mean: {physics_mean}, std: {physics_std}")

    phys_mean_t = torch.tensor(physics_mean, dtype=torch.float32, device=device)
    phys_std_t = torch.tensor(physics_std, dtype=torch.float32, device=device)
    target_idx_t = torch.tensor(TARGET_IDX, dtype=torch.long, device=device)

    batch_size: int = tcfg.get("batch_size", 256)
    lr: float = args.lr or tcfg["learning_rate"]
    num_epochs: int = args.epochs or 100

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=data_generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    belief_enc = PhysicsBeliefEncoder(obs_dim=gru_input_dim, d_slow=d_slow, hidden_dim=128).to(
        device
    )
    physics_head = PhysicsHead(d_slow=d_slow, n_physics=n_physics).to(device)

    params = list(belief_enc.parameters()) + list(physics_head.parameters())
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    checkpoint_path = args.output or os.path.join(
        cfg["checkpointing"]["checkpoint_dir"], "physics_belief.safetensors"
    )
    if not checkpoint_path.endswith(".safetensors"):
        checkpoint_path = checkpoint_path.rsplit(".", 1)[0] + ".safetensors"
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    best_val_r2 = -float("inf")
    print(
        f"\nTraining PhysicsBeliefEncoder — window_k={window_k}, d_slow={d_slow}, "
        f"gru_input={gru_input_dim}D (obs+vel+action)"
    )
    print(f"Predicting: {PHYSICS_KEYS}")

    for epoch in range(num_epochs):
        belief_enc.train()
        physics_head.train()
        train_loss = 0.0

        for batch in train_loader:
            obs_window = batch["obs_window"].to(device)  # [B, K, obs_dim]
            action_window = batch["action_window"].to(device)  # [B, K, action_dim]
            physics_gt = batch["physics"].to(device)[:, target_idx_t]  # [B, n_targets]

            # Normalize physics targets
            physics_norm = (physics_gt - phys_mean_t) / phys_std_t

            # Velocity proxy: Δobs_t = obs_t - obs_{t-1} (zero-padded at t=0)
            vel_window = torch.zeros_like(obs_window)
            vel_window[:, 1:] = obs_window[:, 1:] - obs_window[:, :-1]
            sa_window = torch.cat([obs_window, vel_window, action_window], dim=-1)  # [B, K, 20]

            z_slow = belief_enc(sa_window)
            physics_hat_norm = physics_head(z_slow)
            loss = nn.functional.mse_loss(physics_hat_norm, physics_norm)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        belief_enc.eval()
        physics_head.eval()
        all_hat, all_gt = [], []
        with torch.no_grad():
            for batch in val_loader:
                obs_window = batch["obs_window"].to(device)
                action_window = batch["action_window"].to(device)
                physics_gt = batch["physics"].to(device)[:, target_idx_t]
                vel_window = torch.zeros_like(obs_window)
                vel_window[:, 1:] = obs_window[:, 1:] - obs_window[:, :-1]
                sa_window = torch.cat([obs_window, vel_window, action_window], dim=-1)
                z_slow = belief_enc(sa_window)
                # Unnormalize for R² computation
                physics_hat = physics_head(z_slow) * phys_std_t + phys_mean_t
                all_hat.append(physics_hat.cpu())
                all_gt.append(physics_gt.cpu())

        hat = torch.cat(all_hat, dim=0).numpy()
        gt = torch.cat(all_gt, dim=0).numpy()

        r2_per = _r2_per_target(gt, hat)
        r2_mean = float(r2_per.mean())

        scheduler.step(-r2_mean)
        lr_now = optimizer.param_groups[0]["lr"]

        r2_str = " ".join(f"{k}={r2_per[j]:.3f}" for j, k in enumerate(PHYSICS_KEYS))
        print(
            f"Epoch {epoch + 1:3d} | train_loss: {train_loss:.6f} | "
            f"val R² {r2_str} mean={r2_mean:.3f} | LR: {lr_now:.2e}"
        )

        if r2_mean > best_val_r2:
            best_val_r2 = r2_mean
            combined = {
                **{f"belief_enc.{k}": v for k, v in belief_enc.state_dict().items()},
                **{f"physics_head.{k}": v for k, v in physics_head.state_dict().items()},
            }
            meta = make_metadata(
                model_class="PhysicsBeliefEncoder",
                config={
                    "gru_input_dim": gru_input_dim,
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "d_slow": d_slow,
                    "window_k": window_k,
                },
            )
            meta["gru_input_dim"] = str(gru_input_dim)
            meta["obs_dim"] = str(obs_dim)
            meta["action_dim"] = str(action_dim)
            meta["d_slow"] = str(d_slow)
            meta["window_k"] = str(window_k)
            meta["use_velocity"] = "true"
            meta["physics_keys"] = json.dumps(PHYSICS_KEYS)
            meta["physics_mean"] = json.dumps(physics_mean.tolist())
            meta["physics_std"] = json.dumps(physics_std.tolist())
            save_checkpoint(combined, checkpoint_path, meta)
            print(f"  -> Saved {checkpoint_path} (R²={r2_mean:.3f})")

    print(f"\nDone. Best val R² = {best_val_r2:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Atlas.WM PhysicsBeliefEncoder")
    parser.add_argument(
        "--config",
        default=os.path.normpath(_BASE_CONFIG),
        help="YAML config (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--window-k", type=int, default=None, help="GRU window length (default: 10)"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--output", default=None, help="Checkpoint path (.safetensors)")
    parser.add_argument(
        "--seed", type=int, default=None, help="Override training.seed from the config"
    )
    args = parser.parse_args()
    train_belief_encoder(args)


if __name__ == "__main__":
    main()
