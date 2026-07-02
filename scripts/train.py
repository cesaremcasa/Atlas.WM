"""Train the Atlas.WM world model.

Usage::

    python scripts/train.py                                   # default config
    python scripts/train.py --config configs/experiments/v3_variable_physics.yaml
    python scripts/train.py --max-steps 1000 --no-checkpoint  # smoke-test
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from atlas_wm.checkpointing.io import make_metadata, save_checkpoint
from atlas_wm.data.dataset import ATLASDataset
from atlas_wm.models.continuous_encoder import ContinuousEncoder
from atlas_wm.models.decoder import Decoder
from atlas_wm.models.identifiability import (
    ActionInvarianceCritic,
    critic_loss,
    encoder_adversarial_loss,
)
from atlas_wm.models.structured_dynamics import StructuredDynamics
from atlas_wm.utils.seeding import seed_worker, set_seed

_BASE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* (non-destructive)."""
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
    """Load a YAML config, resolving the optional ``_base`` key."""
    with open(path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    base_ref: str | None = cfg.get("_base")
    if base_ref:
        base_path = os.path.normpath(os.path.join(os.path.dirname(path), base_ref))
        with open(base_path) as f:
            base: dict[str, Any] = yaml.safe_load(f)
        cfg = _deep_merge(base, cfg)
    return cfg


def train(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    mcfg = cfg["model"]
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    ccfg = cfg["checkpointing"]

    # Seed everything BEFORE model construction so init weights, batch order
    # and the whole run are reproducible (v4 B3, AD-7).
    seed: int = args.seed if getattr(args, "seed", None) is not None else tcfg.get("seed", 42)
    data_generator = set_seed(seed)

    # Observation scaling happens inside ATLASDataset (in memory). The split
    # files on disk are never modified (v4 B2, roadmap finding C5).
    data_dir: str = dcfg["data_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: {args.config} | Seed: {seed}")

    train_dataset = ATLASDataset(data_dir, split="train")
    val_dataset = ATLASDataset(data_dir, split="val")

    batch_size: int = tcfg["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=data_generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    d_immutable: int = mcfg["d_static_immutable"]
    d_static: int = d_immutable + mcfg["d_static_slow"]
    d_dynamic: int = mcfg["d_dynamic"]
    d_controllable: int = mcfg["d_controllable"]
    input_dim: int = mcfg["input_dim"]
    action_dim: int = cfg["environment"]["action_space_size"]

    encoder = ContinuousEncoder(
        input_dim=input_dim,
        d_static=d_static,
        d_dynamic=d_dynamic,
        d_controllable=d_controllable,
        d_immutable=d_immutable,
    ).to(device)
    dynamics = StructuredDynamics(
        d_static=d_static,
        d_dynamic=d_dynamic,
        d_controllable=d_controllable,
        action_dim=action_dim,
        d_immutable=d_immutable,
    ).to(device)
    d_full = d_static + d_dynamic + d_controllable
    decoder = Decoder(d_full=d_full, output_dim=input_dim).to(device)
    critic = ActionInvarianceCritic(d_immutable=encoder.d_immutable, action_dim=action_dim).to(
        device
    )

    lr: float = tcfg["learning_rate"]
    weight_decay: float = tcfg.get("weight_decay", 1e-4)
    critic_lr_factor: float = tcfg.get("critic_lr_factor", 1.0)
    params = list(encoder.parameters()) + list(dynamics.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr * critic_lr_factor)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=tcfg["lr_scheduler_patience"],
        factor=tcfg["lr_scheduler_factor"],
    )

    lam_recon: float = tcfg.get("lambda_recon", 1.0)
    lam_drift: float = tcfg["lambda_slow_drift"]
    lam_adv: float = tcfg["lambda_action_invariance"]
    lam_var: float = tcfg.get("lambda_var_penalty", 0.001)
    lam_latent_l2: float = tcfg.get("lambda_latent_l2", 0.01)
    grad_clip: float = tcfg["grad_clip_norm"]
    patience: int = tcfg["early_stopping_patience"]
    num_epochs: int = tcfg["num_epochs"]
    adv_warmup: int = tcfg.get("adv_warmup_epochs", 0)

    max_steps: int | None = args.max_steps if args.max_steps else None
    checkpoint_dir: str = getattr(args, "output_checkpoint", None) or os.path.join(
        ccfg["checkpoint_dir"], "best_model.safetensors"
    )
    os.makedirs(os.path.dirname(checkpoint_dir) or ".", exist_ok=True)

    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0
    steps = 0
    history: dict[str, list[float]] = {"train": [], "val_pred": [], "val_recon": []}

    for epoch in range(num_epochs):
        encoder.train()
        dynamics.train()
        decoder.train()
        critic.train()
        train_loss = 0.0

        for batch in train_loader:
            obs = batch["obs"].to(device)
            action = batch["action"].to(device)
            next_obs = batch["next_obs"].to(device)

            z_t = encoder(obs)
            z_t1_pred = dynamics(z_t, action)

            with torch.no_grad():
                z_t1_true = encoder(next_obs)

            c_loss = critic_loss(critic, z_t["z_static_immutable"].detach(), action)
            critic_optimizer.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
            critic_optimizer.step()

            recon_loss = nn.functional.mse_loss(decoder(z_t["z_full"]), obs)
            pred_loss = nn.functional.mse_loss(z_t1_pred["z_full"], z_t1_true["z_full"])
            z_var = z_t1_pred["z_full"].var(dim=0).mean()
            var_penalty = torch.clamp(1.0 - z_var, min=0)
            drift_penalty = z_t1_pred["delta_slow"].norm(dim=-1).mean()
            # Latent-magnitude penalty. pred_loss is self-predictive (target is the
            # encoder's own detached output over next_obs), which has a degenerate
            # solution: the encoder can inflate its representation scale without bound
            # while dynamics tracks it, blowing up pred_loss ~10x/epoch after a few
            # stable epochs. This L2 anchor removes that runaway direction; validated
            # at the full 50k scale where reconstruction alone does not hold it.
            latent_l2 = z_t["z_full"].pow(2).mean()
            adv_loss = encoder_adversarial_loss(critic, z_t["z_static_immutable"], action)
            effective_lam_adv = lam_adv if epoch >= adv_warmup else 0.0
            loss = (
                lam_recon * recon_loss
                + pred_loss
                + lam_var * var_penalty
                + lam_drift * drift_penalty
                + lam_latent_l2 * latent_l2
                + effective_lam_adv * adv_loss
            )

            if torch.isnan(loss):
                print("NaN detected — skipping batch")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

            train_loss += loss.item()
            steps += 1
            if max_steps and steps >= max_steps:
                break

        train_loss /= max(len(train_loader), 1)

        encoder.eval()
        dynamics.eval()
        decoder.eval()
        val_pred_loss = 0.0
        val_recon_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                action = batch["action"].to(device)
                next_obs = batch["next_obs"].to(device)
                z_t = encoder(obs)
                z_t1_pred = dynamics(z_t, action)
                z_t1_true = encoder(next_obs)
                val_pred_loss += nn.functional.mse_loss(
                    z_t1_pred["z_full"], z_t1_true["z_full"]
                ).item()
                val_recon_loss += nn.functional.mse_loss(decoder(z_t["z_full"]), obs).item()

        val_pred_loss /= max(len(val_loader), 1)
        val_recon_loss /= max(len(val_loader), 1)
        val_loss = val_pred_loss + lam_recon * val_recon_loss
        history["train"].append(train_loss)
        history["val_pred"].append(val_pred_loss)
        history["val_recon"].append(val_recon_loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:3d} | Train: {train_loss:.6f} | "
            f"Val pred: {val_pred_loss:.6f} recon: {val_recon_loss:.6f} | LR: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            if not args.no_checkpoint:
                combined = {
                    **{f"encoder.{k}": v for k, v in encoder.state_dict().items()},
                    **{f"dynamics.{k}": v for k, v in dynamics.state_dict().items()},
                    **{f"decoder.{k}": v for k, v in decoder.state_dict().items()},
                    # Critic included so adversarial training can resume (M8).
                    **{f"critic.{k}": v for k, v in critic.state_dict().items()},
                }
                metadata = make_metadata(
                    model_class="ContinuousEncoder+StructuredDynamics",
                    config={
                        "d_static": d_static,
                        "d_immutable": d_immutable,
                        "d_dynamic": d_dynamic,
                        "d_controllable": d_controllable,
                        "lr": lr,
                        "seed": seed,
                        "epoch": epoch,
                    },
                )
                # Explicit dims + seed so loaders need no shape inference (H6/AD-7).
                metadata.update(
                    {
                        "d_static": str(d_static),
                        "d_immutable": str(d_immutable),
                        "d_dynamic": str(d_dynamic),
                        "d_controllable": str(d_controllable),
                        "seed": str(seed),
                    }
                )
                save_checkpoint(combined, checkpoint_dir, metadata)
                print(f"  -> Saved {checkpoint_dir}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        if max_steps and steps >= max_steps:
            break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    return {"history": history, "best_val_loss": best_val_loss, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Atlas.WM world model")
    parser.add_argument(
        "--config",
        default=os.path.normpath(_BASE_CONFIG),
        help="YAML config (default: configs/base.yaml)",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Limit total gradient steps")
    parser.add_argument("--no-checkpoint", action="store_true", help="Skip saving checkpoints")
    parser.add_argument("--output-checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument(
        "--seed", type=int, default=None, help="Override training.seed from the config"
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
