"""Split raw Atlas.WM data into train / val / test sets.

Usage::

    python scripts/split_data.py                          # defaults
    python scripts/split_data.py --raw-dir data/raw --processed-dir data/processed
    python scripts/split_data.py --force                  # re-split even if already done
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def split_data(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    force: bool = False,
) -> None:
    sentinel = os.path.join(processed_dir, ".split")
    if os.path.exists(sentinel) and not force:
        print("Data already split — skipping (use --force to re-split)")
        return

    obs = np.load(os.path.join(raw_dir, "observations.npy"))
    actions = np.load(os.path.join(raw_dir, "actions.npy"))
    next_obs = np.load(os.path.join(raw_dir, "next_observations.npy"))

    physics_path = os.path.join(raw_dir, "physics_params.npy")
    physics = np.load(physics_path) if os.path.exists(physics_path) else None

    episode_ids_path = os.path.join(raw_dir, "episode_ids.npy")
    episode_ids = np.load(episode_ids_path) if os.path.exists(episode_ids_path) else None

    total = len(obs)
    train_end = int(train_frac * total)
    val_end = int((train_frac + val_frac) * total)

    os.makedirs(processed_dir, exist_ok=True)

    splits = {
        "train": (slice(None, train_end), train_end),
        "val": (slice(train_end, val_end), val_end - train_end),
        "test": (slice(val_end, None), total - val_end),
    }

    for name, (sl, count) in splits.items():
        np.save(os.path.join(processed_dir, f"{name}_obs.npy"), obs[sl])
        np.save(os.path.join(processed_dir, f"{name}_actions.npy"), actions[sl])
        np.save(os.path.join(processed_dir, f"{name}_next_obs.npy"), next_obs[sl])
        if physics is not None:
            np.save(os.path.join(processed_dir, f"{name}_physics.npy"), physics[sl])
        if episode_ids is not None:
            np.save(os.path.join(processed_dir, f"{name}_episode_ids.npy"), episode_ids[sl])
        print(f"  {name}: {count} samples")

    extras = []
    if physics is not None:
        extras.append("physics labels")
    if episode_ids is not None:
        extras.append("episode IDs")
    if extras:
        print(f"Additional arrays split: {', '.join(extras)}")

    open(sentinel, "w").close()
    print(f"Done — processed data in {processed_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split raw Atlas.WM data")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory with raw .npy files")
    parser.add_argument(
        "--processed-dir", default="data/processed", help="Output directory for split files"
    )
    parser.add_argument("--force", action="store_true", help="Re-split even if already done")
    args = parser.parse_args()
    split_data(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
