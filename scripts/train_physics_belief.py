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
  - Targets all three physics parameters. (The v3.x exclusion of friction_agent
    was retracted in v4 — see docs/MODEL_CARD.md and scripts/oracle_friction_agent.py.)
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
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from atlas_wm.checkpointing.io import make_metadata, save_checkpoint
from atlas_wm.data.dynamics_features import N_FEATURES, build_dynamics_features
from atlas_wm.data.episode_dataset import EpisodeATLASDataset
from atlas_wm.models.physics_belief import (
    DistributionalPhysicsHead,
    PhysicsBeliefEncoder,
    gaussian_nll,
)
from atlas_wm.training.objectives import info_nce
from atlas_wm.utils.seeding import seed_worker, set_seed

_BASE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")

# Column order of physics_params.npy (set by generate_data.py).
ALL_PHYSICS_KEYS = ["gravity", "friction_agent", "friction_box"]

# Identification targets: all three parameters. The Block-14 exclusion of
# friction_agent was RETRACTED in v4 (see docs/MODEL_CARD.md): a robust
# median-of-ratios oracle (scripts/oracle_friction_agent.py) recovers it with
# R² = 0.85 from random-policy position-only observations — it is the MOST
# identifiable parameter (the only object under continuous known excitation).
# The original "not identifiable" verdict came from an uncommitted MSE-based
# oracle destroyed by bounce outliers, on an env whose boxes exited the grid.
PHYSICS_KEYS = ["gravity", "friction_agent", "friction_box"]
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

    # v4.1: env-aware features and targets (obs_scale.json carries the env)
    import json as _json
    import os as _os

    env_name = "cruel"
    scale_meta = _os.path.join(data_dir, "obs_scale.json")
    if _os.path.exists(scale_meta):
        with open(scale_meta) as _f:
            env_name = _json.load(_f).get("env", "cruel")

    window_k: int = args.window_k or cfg.get("belief_encoder", {}).get("window_k", 10)
    obs_dim: int = mcfg["input_dim"]
    action_dim: int = cfg["environment"]["action_space_size"]
    # v4 B10: the GRU consumes ENGINEERED dynamics features (the closed-form
    # oracle's per-step decay ratios, gravity projections, distances), not raw
    # obs sequences — the re-baseline showed raw windows score negative R^2 on
    # physics a linear estimator recovers at 0.87 from the same data.
    if env_name == "mujoco":
        from atlas_wm.data.mujoco_features import N_MJ_FEATURES, build_mujoco_features

        feature_fn = build_mujoco_features
        gru_input_dim = N_MJ_FEATURES
        # Gravity alone is structurally unidentifiable here (only mu*g enters
        # box dynamics) — measured in v4.1 part 1.
        physics_keys = ["friction", "mass"]
        all_keys = ["gravity", "friction", "mass"]
        features_tag = "mujoco_v1"
    else:
        feature_fn = build_dynamics_features
        gru_input_dim = N_FEATURES
        physics_keys = list(PHYSICS_KEYS)
        all_keys = list(ALL_PHYSICS_KEYS)
        features_tag = "dynamics_v1"
    target_idx_list = [all_keys.index(k) for k in physics_keys]
    d_slow: int = mcfg["d_static_slow"]
    n_physics = len(physics_keys)
    lam_contrastive: float = cfg.get("belief_encoder", {}).get("lambda_contrastive", 0.1)

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
    physics_train = train_ds.physics[train_ds.valid_indices][:, target_idx_list]
    physics_mean = physics_train.mean(axis=0)
    physics_std = physics_train.std(axis=0) + 1e-8
    print(f"Physics mean: {physics_mean}, std: {physics_std}")

    phys_mean_t = torch.tensor(physics_mean, dtype=torch.float32, device=device)
    phys_std_t = torch.tensor(physics_std, dtype=torch.float32, device=device)
    target_idx_t = torch.tensor(target_idx_list, dtype=torch.long, device=device)

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
    physics_head = DistributionalPhysicsHead(d_slow=d_slow, n_physics=n_physics).to(device)

    params = list(belief_enc.parameters()) + list(physics_head.parameters())
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    checkpoint_path = args.output or os.path.join(
        cfg["checkpointing"]["checkpoint_dir"], "physics_belief.safetensors"
    )
    if not checkpoint_path.endswith(".safetensors"):
        checkpoint_path = os.path.splitext(checkpoint_path)[0] + ".safetensors"
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    best_val_r2 = -float("inf")
    print(
        f"\nTraining PhysicsBeliefEncoder v2 — window_k={window_k}, d_slow={d_slow}, "
        f"gru_input={gru_input_dim}D (engineered dynamics features), "
        f"distributional head, lambda_contrastive={lam_contrastive}"
    )
    print(f"Predicting: {physics_keys} (env={env_name})")

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

            feats = feature_fn(obs_window, action_window)  # [B, K-2, F]

            z_slow = belief_enc(feats)
            mu, logvar = physics_head(z_slow)
            loss = gaussian_nll(mu, logvar, physics_norm)

            # DYSCO-style contrastive: the two halves of a same-episode window
            # must map to the same belief; other episodes in the batch are
            # negatives. Complements the supervised NLL with a label-free
            # episode-identification signal.
            if lam_contrastive > 0 and feats.shape[1] >= 4:
                half = feats.shape[1] // 2
                z_a = belief_enc(feats[:, :half])
                z_b = belief_enc(feats[:, half:])
                loss = loss + lam_contrastive * info_nce(z_a, z_b)

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
                feats = feature_fn(obs_window, action_window)
                z_slow = belief_enc(feats)
                mu, _logvar = physics_head(z_slow)
                # Unnormalize for R² computation
                physics_hat = mu * phys_std_t + phys_mean_t
                all_hat.append(physics_hat.cpu())
                all_gt.append(physics_gt.cpu())

        hat = torch.cat(all_hat, dim=0).numpy()
        gt = torch.cat(all_gt, dim=0).numpy()

        r2_per = _r2_per_target(gt, hat)
        r2_mean = float(r2_per.mean())

        scheduler.step(-r2_mean)
        lr_now = optimizer.param_groups[0]["lr"]

        r2_str = " ".join(f"{k}={r2_per[j]:.3f}" for j, k in enumerate(physics_keys))
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
            meta["features"] = features_tag
            meta["n_features"] = str(N_FEATURES)
            meta["head"] = "distributional"
            meta["physics_keys"] = json.dumps(physics_keys)
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
