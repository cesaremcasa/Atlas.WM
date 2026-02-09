import sys
import os
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environments.cruel_gridworld import CruelGridworld

def generate_data(num_episodes=100, save_path="data/raw"):
    """
    Runs the environment and saves trajectories to disk.
    
    Output format:
    - observations.npy: [Total_Steps, 3, 10, 10] (Grid state)
    - actions.npy: [Total_Steps, 4] (One-hot encoded actions)
    - next_observations.npy: [Total_Steps, 3, 10, 10] (Next state)
    """
    os.makedirs(save_path, exist_ok=True)
    
    env = CruelGridworld(grid_size=10, max_steps=50)
    
    all_obs = []
    all_actions = []
    all_next_obs = []
    
    print(f"Generating {num_episodes} episodes...")
    
    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        done = False
        
        while not done:
            # Random action for data generation (0-3)
            action = np.random.randint(0, 4)
            
            # Store current obs
            all_obs.append(obs.copy())
            
            # Store action (One-hot encode for the model)
            action_one_hot = np.zeros(4)
            action_one_hot[action] = 1.0
            all_actions.append(action_one_hot)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # Store next obs
            all_next_obs.append(next_obs.copy())
            
            obs = next_obs

    # Convert to numpy arrays
    all_obs = np.array(all_obs, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)
    all_next_obs = np.array(all_next_obs, dtype=np.float32)
    
    print(f"Dataset shapes:")
    print(f"  Observations: {all_obs.shape}")
    print(f"  Actions: {all_actions.shape}")
    print(f"  Next Observations: {all_next_obs.shape}")
    
    # Save to disk
    np.save(os.path.join(save_path, "observations.npy"), all_obs)
    np.save(os.path.join(save_path, "actions.npy"), all_actions)
    np.save(os.path.join(save_path, "next_observations.npy"), all_next_obs)
    
    print(f"\nData saved to {save_path}")

if __name__ == "__main__":
    # Small dataset for MVP testing
    generate_data(num_episodes=100)
