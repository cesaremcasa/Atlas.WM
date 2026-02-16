import numpy as np
import sys
sys.path.insert(0, '.')
from src.environments.cruel_gridworld import CruelGridworld

def generate_with_exploration(num_samples=50000):
    env = CruelGridworld()
    
    all_obs = []
    all_actions = []
    all_next_obs = []
    
    obs_tuple = env.reset()
    obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
    
    for i in range(num_samples):
        if np.random.random() < 0.1:
            obs_tuple = env.reset()
            obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
        
        action_idx = np.random.randint(0, env.action_space.n)
        action_onehot = np.zeros(env.action_space.n)
        action_onehot[action_idx] = 1.0
        
        next_obs, _, _, _, _ = env.step(action_idx)
        
        all_obs.append(obs.copy())
        all_actions.append(action_onehot)
        all_next_obs.append(next_obs.copy())
        
        obs = next_obs
        
        if (i + 1) % 10000 == 0:
            print(f"Generated {i + 1}/{num_samples} samples...")
    
    obs_array = np.array(all_obs, dtype=np.float32)
    actions_array = np.array(all_actions, dtype=np.float32)
    next_obs_array = np.array(all_next_obs, dtype=np.float32)
    
    unique = len(np.unique(obs_array.reshape(obs_array.shape[0], -1), axis=0))
    diversity = 100 * unique / obs_array.shape[0]
    
    print(f"\nFinal: {len(obs_array)} samples, {unique} unique ({diversity:.1f}%)")
    
    np.save('data/raw/observations.npy', obs_array)
    np.save('data/raw/actions.npy', actions_array)
    np.save('data/raw/next_observations.npy', next_obs_array)
    
    return True

if __name__ == '__main__':
    generate_with_exploration(num_samples=50000)
