import os

import numpy as np

# Paths
data_dir = "data/raw"
processed_dir = "data/processed"

# Load raw data
obs = np.load(os.path.join(data_dir, "observations.npy"))
actions = np.load(os.path.join(data_dir, "actions.npy"))
next_obs = np.load(os.path.join(data_dir, "next_observations.npy"))

print(f"Raw data shape: {obs.shape}")

# Split parameters
total_samples = len(obs)
train_end = int(0.8 * total_samples)
val_end = int(0.9 * total_samples)

# Create output directory
os.makedirs(processed_dir, exist_ok=True)

# Split and save
print("Splitting and saving data...")

# Train
np.save(os.path.join(processed_dir, "train_obs.npy"), obs[:train_end])
np.save(os.path.join(processed_dir, "train_actions.npy"), actions[:train_end])
np.save(os.path.join(processed_dir, "train_next_obs.npy"), next_obs[:train_end])

# Val
np.save(os.path.join(processed_dir, "val_obs.npy"), obs[train_end:val_end])
np.save(os.path.join(processed_dir, "val_actions.npy"), actions[train_end:val_end])
np.save(os.path.join(processed_dir, "val_next_obs.npy"), next_obs[train_end:val_end])

# Test
np.save(os.path.join(processed_dir, "test_obs.npy"), obs[val_end:])
np.save(os.path.join(processed_dir, "test_actions.npy"), actions[val_end:])
np.save(os.path.join(processed_dir, "test_next_obs.npy"), next_obs[val_end:])

print("Done.")
print(f"Train: {train_end} samples")
print(f"Val: {val_end - train_end} samples")
print(f"Test: {total_samples - val_end} samples")
