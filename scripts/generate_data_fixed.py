import numpy as np
import torch
from src.environments.cruel_gridworld import CruelGridworld

def generate_diverse_data(num_episodes=100, steps_per_episode=50):
    env = CruelGridworld()
    
    all_obs = []
    all_actions = []
    all_next_obs = []
    
    # Use different seeds for each episode to ensure diversity
    for episode in range(num_episodes):
        # Set a unique seed per episode for reproducibility but diversity
        np.random.seed(42 + episode)
        
        obs = env.reset()
        
        for step in range(steps_per_episode):
            # Random action
            action_idx = np.random.randint(0, 4)
            action_onehot = np.zeros(4)
            action_onehot[action_idx] = 1.0
            
            # Take step
            next_obs, _, done, _ = env.step(action_idx)
            
            # Store transition
            all_obs.append(obs.copy())
            all_actions.append(action_onehot)
            all_next_obs.append(next_obs.copy())
            
            obs = next_obs
            
            if done:
                obs = env.reset()
        
        if (episode + 1) % 20 == 0:
            print(f"Generated {episode + 1}/{num_episodes} episodes...")
    
    # Convert to numpy arrays
    obs_array = np.array(all_obs, dtype=np.float32)
    actions_array = np.array(all_actions, dtype=np.float32)
    next_obs_array = np.array(all_next_obs, dtype=np.float32)
    
    print(f"\nFinal dataset size: {len(obs_array)} transitions")
    print(f"Unique observations: {len(np.unique(obs_array.reshape(obs_array.shape[0], -1), axis=0))}")
    
    # Save to raw directory
    np.save('data/raw/observations.npy', obs_array)
    np.save('data/raw/actions.npy', actions_array)
    np.save('data/raw/next_observations.npy', next_obs_array)
    
    print("\nData saved to data/raw/")

if __name__ == '__main__':
    generate_diverse_data()
