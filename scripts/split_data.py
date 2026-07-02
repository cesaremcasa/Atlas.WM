"""Split raw Atlas.WM data into train / val / test sets.

Usage::

    python scripts/split_data.py                          # defaults
    python scripts/split_data.py --raw-dir data/raw --processed-dir data/processed
    python scripts/split_data.py --force                  # re-split even if already done
"""

from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np

_RAW_FILES = (
    "observations.npy",
    "actions.npy",
    "next_observations.npy",
    "physics_params.npy",
    "episode_ids.npy",
)


def raw_fingerprint(raw_dir: str) -> str:
    """SHA-256 over the raw array files (absent files are recorded as absent).

    Stored inside the ``.split`` sentinel so a re-run of generate_data.py
    (which always overwrites ``data/raw``) triggers an automatic re-split
    instead of silently training on stale processed data (v4 B2, roadmap
    finding M4).
    """
    h = hashlib.sha256()
    for name in _RAW_FILES:
        path = os.path.join(raw_dir, name)
        h.update(name.encode())
        if not os.path.exists(path):
            h.update(b":absent")
            continue
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def split_data(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    force: bool = False,
) -> None:
    sentinel = os.path.join(processed_dir, ".split")
    fingerprint = raw_fingerprint(raw_dir)
    if os.path.exists(sentinel) and not force:
        with open(sentinel) as f:
            stored = f.read().strip()
        if stored == fingerprint:
            print("Data already split and raw data unchanged — skipping")
            return
        print("Raw data changed since last split — re-splitting")

    obs = np.load(os.path.join(raw_dir, "observations.npy"))
    actions = np.load(os.path.join(raw_dir, "actions.npy"))
    next_obs = np.load(os.path.join(raw_dir, "next_observations.npy"))

    physics_path = os.path.join(raw_dir, "physics_params.npy")
    physics = np.load(physics_path) if os.path.exists(physics_path) else None

    episode_ids_path = os.path.join(raw_dir, "episode_ids.npy")
    episode_ids = np.load(episode_ids_path) if os.path.exists(episode_ids_path) else None

    total = len(obs)
    os.makedirs(processed_dir, exist_ok=True)

    if episode_ids is not None:
        # Split by episode (not by transition index) to prevent physics
        # distribution shift: seed-deterministic RNG assigns different physics
        # ranges to early vs late episodes, so a sequential 80/10/10 transition
        # split would give train and val systematically different physics ranges.
        unique_eps = np.unique(episode_ids)
        rng = np.random.default_rng(42)
        rng.shuffle(unique_eps)
        n_eps = len(unique_eps)
        train_end_eps = int(train_frac * n_eps)
        val_end_eps = int((train_frac + val_frac) * n_eps)
        train_eps = set(unique_eps[:train_end_eps].tolist())
        val_eps = set(unique_eps[train_end_eps:val_end_eps].tolist())
        test_eps = set(unique_eps[val_end_eps:].tolist())
        masks = {
            "train": np.array([e in train_eps for e in episode_ids]),
            "val": np.array([e in val_eps for e in episode_ids]),
            "test": np.array([e in test_eps for e in episode_ids]),
        }
        print(f"Episode-based split (shuffled, seed=42): {n_eps} episodes")
    else:
        train_end = int(train_frac * total)
        val_end = int((train_frac + val_frac) * total)
        masks = {
            "train": np.zeros(total, dtype=bool),
            "val": np.zeros(total, dtype=bool),
            "test": np.zeros(total, dtype=bool),
        }
        masks["train"][:train_end] = True
        masks["val"][train_end:val_end] = True
        masks["test"][val_end:] = True

    for name, mask in masks.items():
        count = int(mask.sum())
        np.save(os.path.join(processed_dir, f"{name}_obs.npy"), obs[mask])
        np.save(os.path.join(processed_dir, f"{name}_actions.npy"), actions[mask])
        np.save(os.path.join(processed_dir, f"{name}_next_obs.npy"), next_obs[mask])
        if physics is not None:
            np.save(os.path.join(processed_dir, f"{name}_physics.npy"), physics[mask])
        if episode_ids is not None:
            np.save(os.path.join(processed_dir, f"{name}_episode_ids.npy"), episode_ids[mask])
        print(f"  {name}: {count} samples")

    extras = []
    if physics is not None:
        extras.append("physics labels")
    if episode_ids is not None:
        extras.append("episode IDs")
    if extras:
        print(f"Additional arrays split: {', '.join(extras)}")

    # Legacy cleanup: pre-v4 train.py normalized the split files in place and
    # left a .normalized sentinel; the datasets now refuse to load such dirs.
    # A fresh split is unnormalized, so drop the stale marker.
    normalized_sentinel = os.path.join(processed_dir, ".normalized")
    if os.path.exists(normalized_sentinel):
        os.remove(normalized_sentinel)
        print("Removed legacy .normalized sentinel")

    with open(sentinel, "w") as f:
        f.write(fingerprint + "\n")
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
