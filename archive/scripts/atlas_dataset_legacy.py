import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ATLASDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir="data/processed", split="train"):
        """
        Args:
            data_dir: Directory containing the split data files
            split: One of 'train', 'val', or 'test'
        """
        self.split = split
        self.data_dir = data_dir
        
        # Construct filenames based on split
        obs_file = os.path.join(data_dir, f"{split}_obs.npy")
        action_file = os.path.join(data_dir, f"{split}_actions.npy")
        next_obs_file = os.path.join(data_dir, f"{split}_next_obs.npy")
        
        # Load data into memory
        if not os.path.exists(obs_file):
            raise FileNotFoundError(f"{split} data not found in {data_dir}. Did you run split_data.py?")
            
        print(f"Loading {split} data from {data_dir}...")
        self.observations = np.load(obs_file)
        self.actions = np.load(action_file)
        self.next_observations = np.load(next_obs_file)
        
        # Convert to Float32 (Standard for PyTorch)
        self.observations = self.observations.astype(np.float32)
        self.actions = self.actions.astype(np.float32)
        self.next_observations = self.next_observations.astype(np.float32)
        
        print(f"Loaded {len(self.observations)} samples.")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return {
            'obs': torch.from_numpy(self.observations[idx]),
            'action': torch.from_numpy(self.actions[idx]),
            'next_obs': torch.from_numpy(self.next_observations[idx]),
        }
