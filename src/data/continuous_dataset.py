import numpy as np
import torch
from torch.utils.data import Dataset

class ContinuousDataset(Dataset):
    """Dataset for continuous state observations."""
    def __init__(self, data_dir, split='train'):
        print(f"Loading {split} data from {data_dir}...")
        
        self.observations = np.load(f'{data_dir}/{split}_obs.npy')
        self.actions = np.load(f'{data_dir}/{split}_actions.npy')
        self.next_observations = np.load(f'{data_dir}/{split}_next_obs.npy')
        
        print(f"Loaded {len(self.observations)} samples.")
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return {
            'obs': torch.FloatTensor(self.observations[idx]),
            'action': torch.FloatTensor(self.actions[idx]),
            'next_obs': torch.FloatTensor(self.next_observations[idx])
        }
