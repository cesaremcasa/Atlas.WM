import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CruelGridworld(gym.Env):
    """
    CruelGridworld v6: Non-Linear Physics (Attraction).
    - Objects attract each other (Gravity).
    - Requires model to learn complex orbital/collision dynamics.

    Block 12 — Variable Physics & Process Noise (AD-2 validation):
    - Physical parameters (gravity, friction_agent, friction_box) are
      configurable. When ``randomize_physics=True`` they are resampled per
      episode from the provided ranges using the gymnasium-seeded RNG, so
      domain-randomized rollouts remain reproducible (AD-7).
    - ``process_noise_std`` injects zero-mean Gaussian noise into velocities
      each step, also drawn from the seeded RNG.
    - The active physics parameters are exposed in the ``info`` dict returned by
      ``reset`` and ``step`` so data generation can record ground-truth labels
      for latent probing.

    Defaults reproduce the original deterministic v6 behavior bit-for-bit:
    ``randomize_physics=False`` and ``process_noise_std=0.0`` consume no extra
    RNG draws, so existing datasets and determinism canaries are unaffected.
    """

    def __init__(
        self,
        grid_size=20,
        gravity: float = 5.0,
        friction_agent: float = 0.96,
        friction_box: float = 0.98,
        process_noise_std: float = 0.0,
        randomize_physics: bool = False,
        gravity_range: tuple[float, float] = (2.0, 8.0),
        friction_agent_range: tuple[float, float] = (0.90, 0.99),
        friction_box_range: tuple[float, float] = (0.95, 0.995),
    ):
        super().__init__()
        self.grid_size = grid_size
        self.dt = 0.5  # Smaller dt for stability
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=grid_size, shape=(6,), dtype=np.float32)

        # Default / nominal physics. When randomize_physics is False these are
        # used verbatim every episode.
        self.gravity = gravity
        self.friction_agent = friction_agent
        self.friction_box = friction_box

        self.process_noise_std = process_noise_std
        self.randomize_physics = randomize_physics
        self.gravity_range = gravity_range
        self.friction_agent_range = friction_agent_range
        self.friction_box_range = friction_box_range

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Use self.np_random (gymnasium-seeded) for full determinism (AD-7).
        rng = self.np_random

        # Block 12: resample physics per episode (domain randomization).
        # Guarded by the flag so the default code path consumes no extra RNG
        # draws and remains byte-identical to the original environment.
        if self.randomize_physics:
            self.gravity = float(rng.uniform(*self.gravity_range))
            self.friction_agent = float(rng.uniform(*self.friction_agent_range))
            self.friction_box = float(rng.uniform(*self.friction_box_range))

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

        return self._get_obs(), self._physics_info()

    def _get_obs(self):
        obs = np.concatenate([self.agent_pos, self.box_positions[0], self.box_positions[1]])
        return obs.astype(np.float32)

    def _physics_info(self) -> dict[str, float]:
        """Active physical parameters — ground truth for latent probing (Block 12)."""
        return {
            "gravity": float(self.gravity),
            "friction_agent": float(self.friction_agent),
            "friction_box": float(self.friction_box),
        }

    def _apply_gravity(self, p1, v1, p2, v2, G=5.0):
        """Applies gravitational attraction."""
        r_vec = p2 - p1
        dist = np.linalg.norm(r_vec)
        if 1.0 < dist < 10.0:  # Gravity range
            force_mag = G / (dist**2)
            force = (r_vec / dist) * force_mag * self.dt
            v1 += force
            v2 -= force  # Newton's 3rd law

    def _apply_process_noise(self):
        """Inject zero-mean Gaussian noise into velocities (Block 12).

        Drawn from the gymnasium-seeded RNG so noisy rollouts stay reproducible
        under a fixed seed (AD-7). No-op when process_noise_std == 0.
        """
        if self.process_noise_std <= 0.0:
            return
        std = self.process_noise_std * self.dt
        self.agent_vel += self.np_random.normal(0.0, std, size=2)
        self.box_vels[0] += self.np_random.normal(0.0, std, size=2)
        self.box_vels[1] += self.np_random.normal(0.0, std, size=2)

    def step(self, action):
        forces = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        # 1. Apply Input Force
        fx, fy = forces[action]
        self.agent_vel += np.array([fx, fy]) * 0.8

        # 2. Apply Non-Linear Gravity (Agent <-> Box1, Agent <-> Box2, Box1 <-> Box2)
        self._apply_gravity(
            self.agent_pos, self.agent_vel, self.box_positions[0], self.box_vels[0], G=self.gravity
        )
        self._apply_gravity(
            self.agent_pos, self.agent_vel, self.box_positions[1], self.box_vels[1], G=self.gravity
        )
        self._apply_gravity(
            self.box_positions[0],
            self.box_vels[0],
            self.box_positions[1],
            self.box_vels[1],
            G=self.gravity,
        )

        # 3. Update Velocities
        self.agent_vel *= self.friction_agent  # Friction
        self.box_vels[0] *= self.friction_box
        self.box_vels[1] *= self.friction_box

        # 3b. Process noise (Block 12) — no-op unless process_noise_std > 0
        self._apply_process_noise()

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

        return self._get_obs(), 0.0, False, False, self._physics_info()
