import os
import numpy as np
import sys
sys.path.insert(0, '.')
from src.environments.cruel_gridworld import CruelGridworld

def generate_data(num_episodes=2000, max_steps=100, output_dir="data/raw"):
    # FORCE Clean old files
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    env = CruelGridworld(grid_size=20)
    all_obs = []
    all_actions = []
    all_next_obs = []
    
    print(f"Generating {num_episodes} CONTINUOUS episodes...")
    print("WARNING: This will take longer due to physics calculations.")
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            action = env.action_space.sample()
            next_obs, _, _, _, _ = env.step(action)
            
            action_onehot = np.zeros(8)
            action_onehot[action] = 1.0
            
            all_obs.append(obs)
            all_actions.append(action_onehot)
            all_next_obs.append(next_obs)
            obs = next_obs
            
        if (ep + 1) % 200 == 0:
            print(f"Episode {ep+1}/{num_episodes}")

    np.save(os.path.join(output_dir, "observations.npy"), np.array(all_obs))
    np.save(os.path.join(output_dir, "actions.npy"), np.array(all_actions))
    np.save(os.path.join(output_dir, "next_observations.npy"), np.array(all_next_obs))
    print(f"Saved {len(all_obs)} transitions.")
    print(f"Shape: {np.array(all_obs).shape}")

if __name__ == "__main__":
    generate_data()
