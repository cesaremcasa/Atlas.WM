import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CruelGridworld(gym.Env):
    """
    CruelGridworld v6: Non-Linear Physics (Attraction).
    - Objects attract each other (Gravity).
    - Requires model to learn complex orbital/collision dynamics.
    """

    def __init__(self, grid_size=20):
        super().__init__()
        self.grid_size = grid_size
        self.dt = 0.5  # Smaller dt for stability
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=grid_size, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Use self.np_random (gymnasium-seeded) for full determinism (AD-7).
        rng = self.np_random

        # Walls
        self.walls = []
        for _ in range(int(rng.integers(4, 9))):
            self.walls.append(
                {
                    "x": float(rng.uniform(2, self.grid_size - 2)),
                    "y": float(rng.uniform(2, self.grid_size - 2)),
                    "r": float(rng.uniform(0.5, 1.5)),
                }
            )

        # Agent
        self.agent_pos = rng.uniform(2, self.grid_size - 2, size=2).astype(float)
        self.agent_vel = rng.uniform(-1, 1, size=2).astype(float)

        # Boxes
        self.box_positions = []
        self.box_vels = []
        for _ in range(2):
            pos = rng.uniform(2, self.grid_size - 2, size=2).astype(float)
            vel = rng.uniform(-1, 1, size=2).astype(float)
            # Ensure separation
            while np.linalg.norm(pos - self.agent_pos) < 3.0:
                pos = rng.uniform(2, self.grid_size - 2, size=2).astype(float)
            self.box_positions.append(pos)
            self.box_vels.append(vel)

        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.concatenate([self.agent_pos, self.box_positions[0], self.box_positions[1]])
        return obs.astype(np.float32)

    def _apply_gravity(self, p1, v1, p2, v2, G=5.0):
        """Applies gravitational attraction."""
        r_vec = p2 - p1
        dist = np.linalg.norm(r_vec)
        if 1.0 < dist < 10.0:  # Gravity range
            force_mag = G / (dist**2)
            force = (r_vec / dist) * force_mag * self.dt
            v1 += force
            v2 -= force  # Newton's 3rd law

    def step(self, action):
        forces = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        # 1. Apply Input Force
        fx, fy = forces[action]
        self.agent_vel += np.array([fx, fy]) * 0.8

        # 2. Apply Non-Linear Gravity (Agent <-> Box1, Agent <-> Box2, Box1 <-> Box2)
        self._apply_gravity(self.agent_pos, self.agent_vel, self.box_positions[0], self.box_vels[0])
        self._apply_gravity(self.agent_pos, self.agent_vel, self.box_positions[1], self.box_vels[1])
        self._apply_gravity(
            self.box_positions[0], self.box_vels[0], self.box_positions[1], self.box_vels[1]
        )

        # 3. Update Velocities
        self.agent_vel *= 0.96  # Friction
        self.box_vels[0] *= 0.98
        self.box_vels[1] *= 0.98

        # 4. Update Positions
        self.agent_pos += self.agent_vel * self.dt
        self.box_positions[0] += self.box_vels[0] * self.dt
        self.box_positions[1] += self.box_vels[1] * self.dt

        # 5. Wall Bounce
        for i in range(2):
            if self.agent_pos[i] < 0.5:
                self.agent_pos[i] = 0.5
                self.agent_vel[i] *= -0.8
            elif self.agent_pos[i] > self.grid_size - 0.5:
                self.agent_pos[i] = self.grid_size - 0.5
                self.agent_vel[i] *= -0.8

        # 6. Obstacle Collision (Simple Bounce)
        for w in self.walls:
            dist_vec = self.agent_pos - np.array([w["x"], w["y"]])
            dist = np.linalg.norm(dist_vec)
            if dist < (0.5 + w["r"]):
                n = dist_vec / dist
                self.agent_vel = self.agent_vel - 2 * np.dot(self.agent_vel, n) * n
                self.agent_pos = np.array([w["x"], w["y"]]) + n * (0.5 + w["r"])

        return self._get_obs(), 0.0, False, False, {}
