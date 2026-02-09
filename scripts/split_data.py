import numpy as np
import os
from sklearn.model_selection import train_test_split

def split_data(data_dir="data/raw", split_dir="data/processed"):
    """
    Splits raw data into Train/Val/Test sets (80/10/10).
    """
    os.makedirs(split_dir, exist_ok=True)
    
    # Load raw data
    obs = np.load(os.path.join(data_dir, "observations.npy"))
    actions = np.load(os.path.join(data_dir, "actions.npy"))
    next_obs = np.load(os.path.join(data_dir, "next_observations.npy"))
    
    print(f"Total samples: {len(obs)}")
    
    # Split 80% Train, 20% Temp
    train_obs, temp_obs, train_act, temp_act, train_next, temp_next = train_test_split(
        obs, actions, next_obs, test_size=0.2, random_state=42
    )
    
    # Split Temp into 50% Val, 50% Test (10% total each)
    val_obs, test_obs, val_act, test_act, val_next, test_next = train_test_split(
        temp_obs, temp_act, temp_next, test_size=0.5, random_state=42
    )
    
    print(f"Train: {len(train_obs)}")
    print(f"Val: {len(val_obs)}")
    print(f"Test: {len(test_obs)}")
    
    # Save
    np.save(os.path.join(split_dir, "train_obs.npy"), train_obs)
    np.save(os.path.join(split_dir, "train_act.npy"), train_act)
    np.save(os.path.join(split_dir, "train_next.npy"), train_next)
    
    np.save(os.path.join(split_dir, "val_obs.npy"), val_obs)
    np.save(os.path.join(split_dir, "val_act.npy"), val_act)
    np.save(os.path.join(split_dir, "val_next.npy"), val_next)
    
    np.save(os.path.join(split_dir, "test_obs.npy"), test_obs)
    np.save(os.path.join(split_dir, "test_act.npy"), test_act)
    np.save(os.path.join(split_dir, "test_next.npy"), test_next)
    
    print(f"Data saved to {split_dir}")

if __name__ == "__main__":
    split_data()
