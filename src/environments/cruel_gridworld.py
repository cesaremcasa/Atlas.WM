import numpy as np
from typing import Tuple, Dict, List

class CruelGridworld:
    """
    Minimal environment designed to expose world model failures.
    Deterministic physics. No stochasticity.
    """
    def __init__(self, grid_size: int = 10, max_steps: int = 50):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.action_space = 4  # Up, Down, Left, Right
        
        # Physics Constants
        self.max_object_speed = 1.0  # Max displacement per tick
        self.friction = 0.1
        
        # State placeholders
        self.agent_pos = np.array([0, 0], dtype=float)
        self.objects: List[Dict] = []
        self.walls: List[Tuple[int, int]] = []
        self.current_step = 0

    def reset(self) -> np.ndarray:
        """Resets environment to initial state."""
        self.current_step = 0
        self.agent_pos = np.array([5.0, 5.0]) # Start center
        
        # Create Boundary Walls
        self.walls = []
        for x in range(self.grid_size):
            self.walls.append((x, 0))
            self.walls.append((x, self.grid_size - 1))
            self.walls.append((0, x))
            self.walls.append((self.grid_size - 1, x))
            
        # Scenario: Teleporting Box Setup
        # Box at [5, 8], Wall at [5, 9]
        self.objects = [
            {'id': 'box', 'pos': np.array([5.0, 8.0]), 'type': 'movable', 'mass': 1.0},
            {'id': 'wall_target', 'pos': np.array([5.0, 9.0]), 'type': 'static'}
        ]
        
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Applies physics and action.
        Action: 0=Up, 1=Down, 2=Left, 3=Right
        """
        self.current_step += 1
        
        # 1. Apply Agent Movement
        move_vec = np.array([0.0, 0.0])
        if action == 0: move_vec[1] = 1.0
        elif action == 1: move_vec[1] = -1.0
        elif action == 2: move_vec[0] = -1.0
        elif action == 3: move_vec[0] = 1.0
        
        new_agent_pos = self.agent_pos + move_vec
        
        # 2. Check Wall Collision (Agent)
        if not self._check_collision(new_agent_pos):
            self.agent_pos = new_agent_pos
            
        # 3. Object Interaction (Push Mechanics)
        for obj in self.objects:
            if obj['type'] == 'movable':
                # If agent is adjacent and pushing into object
                if np.linalg.norm(self.agent_pos - obj['pos']) < 1.1:
                    # Determine push direction
                    push_dir = obj['pos'] - (self.agent_pos - move_vec)
                    
                    # Normalize and clamp speed
                    if np.linalg.norm(push_dir) > 0:
                        push_dir = push_dir / np.linalg.norm(push_dir)
                        new_obj_pos = obj['pos'] + push_dir
                        
                        # Check Wall Collision (Object)
                        if not self._check_collision(new_obj_pos):
                            obj['pos'] = new_obj_pos
                        else:
                            # Object stops (Physics Truth)
                            pass
        
        done = self.current_step >= self.max_steps
        info = {'ground_truth_physics_state': self._get_internal_state()}
        
        return self._get_observation(), 0.0, done, info

    def _check_collision(self, pos: np.ndarray) -> bool:
        """Checks if position is inside a wall."""
        x, y = int(round(pos[0])), int(round(pos[1]))
        # Boundary checks
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        # Wall list checks
        if (x, y) in self.walls:
            return True
        return False

    def _get_observation(self) -> np.ndarray:
        """
        Renders state as a simplified grid observation (3 channels).
        Ch0: Agent, Ch1: Walls, Ch2: Objects
        """
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        
        # Agent
        ax, ay = int(round(self.agent_pos[0])), int(round(self.agent_pos[1]))
        if 0 <= ax < self.grid_size and 0 <= ay < self.grid_size:
            obs[0, ay, ax] = 1.0
            
        # Walls & Objects
        for wx, wy in self.walls:
            obs[1, wy, wx] = 1.0
            
        for obj in self.objects:
            ox, oy = int(round(obj['pos'][0])), int(round(obj['pos'][1]))
            if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                obs[2, oy, ox] = 1.0
                
        return obs

    def _get_internal_state(self) -> Dict:
        """Returns exact ground truth coordinates for validation."""
        return {
            'agent_pos': self.agent_pos.copy(),
            'objects': [{'id': o['id'], 'pos': o['pos'].copy()} for o in self.objects]
        }
