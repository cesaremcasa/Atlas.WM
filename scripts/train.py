import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from atlas_wm.checkpointing.io import make_metadata, save_checkpoint
from atlas_wm.data.dataset import ATLASDataset
from atlas_wm.models.continuous_encoder import ContinuousEncoder
from atlas_wm.models.structured_dynamics import StructuredDynamics


def normalize_data(data_dir: str) -> None:
    for split in ["train", "val", "test"]:
        obs = np.load(f"{data_dir}/{split}_obs.npy")
        next_obs = np.load(f"{data_dir}/{split}_next_obs.npy")
        obs = obs / 20.0
        next_obs = next_obs / 20.0
        np.save(f"{data_dir}/{split}_obs.npy", obs)
        np.save(f"{data_dir}/{split}_next_obs.npy", next_obs)
    print("Data normalized to [0, 1]")


def train(args: argparse.Namespace) -> None:
    normalize_data("data/processed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = ATLASDataset("data/processed", split="train")
    val_dataset = ATLASDataset("data/processed", split="val")

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    encoder = ContinuousEncoder(input_dim=6, d_static=16, d_dynamic=32, d_controllable=16).to(
        device
    )
    dynamics = StructuredDynamics(d_static=16, d_dynamic=32, d_controllable=16, action_dim=8).to(
        device
    )

    params = list(encoder.parameters()) + list(dynamics.parameters())
    optimizer = optim.Adam(params, lr=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    print("Starting training...")
    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0
    max_steps = args.max_steps if hasattr(args, "max_steps") and args.max_steps else None
    steps = 0

    for epoch in range(100):
        encoder.train()
        dynamics.train()
        train_loss = 0.0

        for batch in train_loader:
            obs = batch["obs"].to(device)
            action = batch["action"].to(device)
            next_obs = batch["next_obs"].to(device)

            z_t = encoder(obs)
            z_t1_pred = dynamics(z_t, action)

            with torch.no_grad():
                z_t1_true = encoder(next_obs)

            pred_loss = nn.functional.mse_loss(z_t1_pred["z_full"], z_t1_true["z_full"])
            z_var = z_t1_pred["z_full"].var(dim=0).mean()
            var_penalty = torch.clamp(1.0 - z_var, min=0)
            loss = pred_loss + 0.01 * var_penalty

            if torch.isnan(loss):
                print("NaN detected! Skipping batch...")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()

            train_loss += loss.item()
            steps += 1
            if max_steps and steps >= max_steps:
                break

        train_loss /= max(len(train_loader), 1)

        encoder.eval()
        dynamics.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                action = batch["action"].to(device)
                next_obs = batch["next_obs"].to(device)

                z_t = encoder(obs)
                z_t1_pred = dynamics(z_t, action)
                z_t1_true = encoder(next_obs)

                loss = nn.functional.mse_loss(z_t1_pred["z_full"], z_t1_true["z_full"])
                val_loss += loss.item()

        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:3d} | Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            no_checkpoint = hasattr(args, "no_checkpoint") and args.no_checkpoint
            out_path = (
                getattr(args, "output_checkpoint", None) or "checkpoints/best_model.safetensors"
            )

            if not no_checkpoint:
                combined = {
                    **{f"encoder.{k}": v for k, v in encoder.state_dict().items()},
                    **{f"dynamics.{k}": v for k, v in dynamics.state_dict().items()},
                }
                metadata = make_metadata(
                    model_class="ContinuousEncoder+StructuredDynamics",
                    config={"batch_size": 256, "lr": 3e-4, "epoch": epoch},
                )
                save_checkpoint(combined, out_path, metadata)
                print(f"  -> Best model saved to {out_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        if max_steps and steps >= max_steps:
            break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Atlas.WM")
    parser.add_argument("--config", default="configs/experiments/v3_hybrid_static.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--output-checkpoint", default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
