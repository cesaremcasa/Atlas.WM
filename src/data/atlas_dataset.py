import torch
from torch.utils.data import Dataset
import numpy as np
import os

class ATLASDataset(Dataset):
    """
    Loads pre-generated trajectories from .npy files.
    """
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        
        # Load data into memory (Small enough for MVP: ~12MB total)
        print(f"Loading data from {data_dir}...")
        self.observations = np.load(os.path.join(data_dir, "observations.npy"))
        self.actions = np.load(os.path.join(data_dir, "actions.npy"))
        self.next_observations = np.load(os.path.join(data_dir, "next_observations.npy"))
        
        # Convert to Float32 (Standard for PyTorch)
        self.observations = self.observations.astype(np.float32)
        self.actions = self.actions.astype(np.float32)
        self.next_observations = self.next_observations.astype(np.float32)
        
        print(f"Loaded {len(self.observations)} transitions.")
        
        # Validation checks
        assert self.observations.shape[0] == self.actions.shape[0], "Mismatch in data length"
        assert self.observations.shape[1:] == (3, 10, 10), "Wrong obs shape"

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        """
        Returns:
            obs: [3, 10, 10]
            action: [4]
            next_obs: [3, 10, 10]
        """
        obs = torch.from_numpy(self.observations[idx])
        action = torch.from_numpy(self.actions[idx])
        next_obs = torch.from_numpy(self.next_observations[idx])
        
        return obs, action, next_obs

if __name__ == "__main__":
    # Quick test to ensure it works
    dataset = ATLASDataset()
    obs, act, next_obs = dataset[0]
    print(f"Sample loaded: Obs {obs.shape}, Act {act.shape}")
