"""Episode-windowed dataset for training PhysicsBeliefEncoder.

Returns windows of K consecutive same-episode transitions so a GRU can
accumulate evidence about episode-level physics (gravity, friction) that are
constant within an episode but vary across episodes.

Requires ``episode_ids.npy`` files produced by ``scripts/generate_data.py``.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class EpisodeATLASDataset(Dataset):
    """Windowed dataset: returns K consecutive same-episode observations.

    Each item contains:
        obs_window : [K, obs_dim]  — K consecutive obs from the same episode
        obs        : [obs_dim]     — current observation (last in window)
        action     : [action_dim]  — action taken at current obs
        next_obs   : [obs_dim]     — result of action
        physics    : [3]           — (gravity, friction_agent, friction_box),
                                     only present when physics labels exist

    Raises FileNotFoundError if episode_ids are not found. Re-run
    ``scripts/generate_data.py`` and ``scripts/split_data.py`` to generate them.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        window_k: int = 10,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        self.window_k = window_k

        ids_path = os.path.join(data_dir, f"{split}_episode_ids.npy")
        if not os.path.exists(ids_path):
            raise FileNotFoundError(
                f"Episode IDs not found at {ids_path}. "
                "Re-run: python scripts/generate_data.py --randomize-physics --seed 42 "
                "then: python scripts/split_data.py"
            )

        self.obs = np.load(os.path.join(data_dir, f"{split}_obs.npy")).astype(np.float32)
        self.actions = np.load(os.path.join(data_dir, f"{split}_actions.npy")).astype(np.float32)
        self.next_obs = np.load(os.path.join(data_dir, f"{split}_next_obs.npy")).astype(np.float32)
        self.episode_ids = np.load(ids_path).astype(np.int64)

        physics_path = os.path.join(data_dir, f"{split}_physics.npy")
        self.physics: np.ndarray | None = (
            np.load(physics_path).astype(np.float32) if os.path.exists(physics_path) else None
        )

        self.valid_indices = self._build_valid_indices()
        print(
            f"EpisodeATLASDataset({split}): {len(self.valid_indices)} valid "
            f"{window_k}-step windows out of {len(self.obs)} transitions"
        )

    def _build_valid_indices(self) -> np.ndarray:
        k = self.window_k
        n = len(self.obs)
        ids = self.episode_ids

        # Boundary at position j iff ids[j] != ids[j-1]
        boundaries = np.zeros(n, dtype=np.int32)
        boundaries[1:] = (ids[1:] != ids[:-1]).astype(np.int32)
        cumsum = np.cumsum(boundaries)

        # Index i is valid iff no boundary exists in (i-k, i] i.e. cumsum[i] == cumsum[i-k]
        i_arr = np.arange(k - 1, n)
        prev = np.where(i_arr >= k, cumsum[i_arr - k], 0)
        valid_mask = cumsum[i_arr] == prev
        result: np.ndarray = i_arr[valid_mask]
        return result

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict:
        i = int(self.valid_indices[idx])
        k = self.window_k
        obs_window = self.obs[i - k + 1 : i + 1]  # [K, obs_dim]

        action_window = self.actions[i - k + 1 : i + 1]  # [K, action_dim]

        item: dict = {
            "obs_window": torch.from_numpy(obs_window),
            "action_window": torch.from_numpy(action_window),
            "obs": torch.from_numpy(self.obs[i]),
            "action": torch.from_numpy(self.actions[i]),
            "next_obs": torch.from_numpy(self.next_obs[i]),
        }
        if self.physics is not None:
            item["physics"] = torch.from_numpy(self.physics[i])
        return item
