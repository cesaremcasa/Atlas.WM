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

    ``frame_stack=2`` (v4 B6) concatenates the previous same-episode frame to
    each observation: obs → [prev_obs | obs], next_obs → [obs | next_obs].
    A single 6-D position frame is a fundamentally ambiguous world-model
    input — velocities are unobservable, so one-step prediction has a large
    irreducible error floor (roadmap finding M2). Two frames make velocity
    observable. At an episode's first transition the frame is repeated
    (standard practice). Requires ``{split}_episode_ids.npy``.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        obs_scale: float = DEFAULT_OBS_SCALE,
        frame_stack: int = 1,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
        if frame_stack not in (1, 2):
            raise ValueError(f"frame_stack must be 1 or 2, got {frame_stack}")

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
        self.frame_stack = frame_stack
        obs = np.load(obs_path).astype(np.float32) / obs_scale
        next_obs = np.load(next_obs_path).astype(np.float32) / obs_scale
        self.actions = np.load(actions_path).astype(np.float32)

        if frame_stack == 2:
            ids_path = f"{data_dir}/{split}_episode_ids.npy"
            if not os.path.exists(ids_path):
                raise FileNotFoundError(
                    f"frame_stack=2 requires episode IDs at {ids_path}. "
                    "Re-run scripts/generate_data.py + scripts/split_data.py."
                )
            ids = np.load(ids_path)
            prev = np.empty_like(obs)
            prev[1:] = obs[:-1]
            prev[0] = obs[0]
            # First transition of each episode: no previous frame — repeat it.
            first_of_episode = np.ones(len(obs), dtype=bool)
            first_of_episode[1:] = ids[1:] != ids[:-1]
            prev[first_of_episode] = obs[first_of_episode]
            # Stacked next input: the frame preceding next_obs is obs itself.
            self.observations = np.concatenate([prev, obs], axis=1)
            self.next_observations = np.concatenate([obs, next_obs], axis=1)
        else:
            self.observations = obs
            self.next_observations = next_obs

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> dict:
        return {
            "obs": torch.from_numpy(self.observations[idx]),
            "action": torch.from_numpy(self.actions[idx]),
            "next_obs": torch.from_numpy(self.next_observations[idx]),
        }
