import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CruelGridworld(gym.Env):
    def __init__(self, grid_size=10):
        super().__init__()
        self.grid_size = grid_size
        
        # Action space: UP, DOWN, LEFT, RIGHT
        self.action_space = spaces.Discrete(4)
        
        # Observation: 3 channels (agent, box, wall)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(3, grid_size, grid_size), 
            dtype=np.float32
        )
        
        # Static wall positions (always same)
        self.wall_positions = [
            (grid_size // 2, grid_size - 1),  # Top middle
        ]
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # RANDOMIZE starting positions
        self.agent_pos = [
            np.random.randint(1, self.grid_size - 1),
            np.random.randint(1, self.grid_size - 2)
        ]
        
        # Box starts in random position (not same as agent)
        while True:
            self.box_pos = [
                np.random.randint(1, self.grid_size - 1),
                np.random.randint(1, self.grid_size - 2)
            ]
            if self.box_pos != self.agent_pos:
                break
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        
        # Channel 0: Agent
        obs[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        
        # Channel 1: Box
        obs[1, self.box_pos[0], self.box_pos[1]] = 1.0
        
        # Channel 2: Walls
        for wall in self.wall_positions:
            obs[2, wall[0], wall[1]] = 1.0
        
        return obs
    
    def step(self, action):
        # Action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = moves[action]
        
        new_agent_pos = [
            self.agent_pos[0] + dx,
            self.agent_pos[1] + dy
        ]
        
        # Check boundaries
        if (new_agent_pos[0] < 0 or new_agent_pos[0] >= self.grid_size or
            new_agent_pos[1] < 0 or new_agent_pos[1] >= self.grid_size):
            # Hit boundary, don't move
            return self._get_obs(), 0, False, False, {}
        
        # Check wall collision
        if tuple(new_agent_pos) in self.wall_positions:
            # Hit wall, don't move
            return self._get_obs(), 0, False, False, {}
        
        # Check if pushing box
        if new_agent_pos == self.box_pos:
            # Try to push box
            new_box_pos = [
                self.box_pos[0] + dx,
                self.box_pos[1] + dy
            ]
            
            # Check if box can move
            if (new_box_pos[0] < 0 or new_box_pos[0] >= self.grid_size or
                new_box_pos[1] < 0 or new_box_pos[1] >= self.grid_size or
                tuple(new_box_pos) in self.wall_positions):
                # Box can't move, agent can't move
                return self._get_obs(), 0, False, False, {}
            
            # Push box
            self.box_pos = new_box_pos
            self.agent_pos = new_agent_pos
        else:
            # Just move agent
            self.agent_pos = new_agent_pos
        
        return self._get_obs(), 0, False, False, {}
