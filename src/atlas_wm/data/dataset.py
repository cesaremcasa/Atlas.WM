import os

import numpy as np
import torch
from torch.utils.data import Dataset

# CruelGridworld coordinates live in [0, grid_size] with grid_size = 20.
# Observations are scaled by this constant in memory at load time; the split
# files on disk are never modified (v4 B2). Every consumer of the processed
# data — ATLASDataset, EpisodeATLASDataset, probe scripts — must apply the
# same scale, and importing it from here keeps them in agreement.
DEFAULT_OBS_SCALE = 20.0


def reject_legacy_normalized(data_dir: str) -> None:
    """Fail loudly on data dirs normalized in place by pre-v4 ``train.py``.

    Older versions divided the split ``.npy`` files by 20 **on disk**, guarded
    by a hidden ``.normalized`` sentinel. Loading such a directory here would
    divide a second time, silently corrupting every downstream result — the
    exact failure mode that made belief-probe numbers unfalsifiable (roadmap
    finding C5).
    """
    if os.path.exists(os.path.join(data_dir, ".normalized")):
        raise RuntimeError(
            f"{data_dir} was normalized in place by a pre-v4 version of "
            "scripts/train.py, so its files are already divided by 20. "
            "Regenerate the splits before continuing: "
            "python scripts/split_data.py --force"
        )


class ATLASDataset(Dataset):
    """Dataset for Atlas.WM continuous-state transitions.

    Loads (obs, action, next_obs) triples from pre-split .npy files produced
    by scripts/generate_data.py + scripts/split_data.py. Observations are
    scaled to [0, 1] in memory; files on disk are never modified.
    """

    def __init__(
        self, data_dir: str, split: str = "train", obs_scale: float = DEFAULT_OBS_SCALE
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        reject_legacy_normalized(data_dir)

        obs_path = f"{data_dir}/{split}_obs.npy"
        actions_path = f"{data_dir}/{split}_actions.npy"
        next_obs_path = f"{data_dir}/{split}_next_obs.npy"

        if not os.path.exists(obs_path):
            raise FileNotFoundError(
                f"{split} data not found at {obs_path}. "
                "Run scripts/generate_data.py then scripts/split_data.py first."
            )

        self.obs_scale = obs_scale
        self.observations = np.load(obs_path).astype(np.float32) / obs_scale
        self.actions = np.load(actions_path).astype(np.float32)
        self.next_observations = np.load(next_obs_path).astype(np.float32) / obs_scale

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> dict:
        return {
            "obs": torch.from_numpy(self.observations[idx]),
            "action": torch.from_numpy(self.actions[idx]),
            "next_obs": torch.from_numpy(self.next_observations[idx]),
        }
