import numpy as np
import torch
from torch.utils.data import Dataset


class ATLASDataset(Dataset):
    """Dataset for Atlas.WM continuous-state transitions.

    Loads (obs, action, next_obs) triples from pre-split .npy files produced
    by scripts/generate_data.py + scripts/split_data.py.
    """

    def __init__(self, data_dir: str, split: str = "train") -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        obs_path = f"{data_dir}/{split}_obs.npy"
        actions_path = f"{data_dir}/{split}_actions.npy"
        next_obs_path = f"{data_dir}/{split}_next_obs.npy"

        import os
        if not os.path.exists(obs_path):
            raise FileNotFoundError(
                f"{split} data not found at {obs_path}. "
                "Run scripts/generate_data.py then scripts/split_data.py first."
            )

        self.observations = np.load(obs_path).astype(np.float32)
        self.actions = np.load(actions_path).astype(np.float32)
        self.next_observations = np.load(next_obs_path).astype(np.float32)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> dict:
        return {
            "obs": torch.from_numpy(self.observations[idx]),
            "action": torch.from_numpy(self.actions[idx]),
            "next_obs": torch.from_numpy(self.next_observations[idx]),
        }
