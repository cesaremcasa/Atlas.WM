"""MuJoCo point-mass environment with per-episode physics randomization (v4 B14).

The post-toy environment tier: real MuJoCo contact/friction dynamics with
the same interface contract as CruelGridworld so the entire v4 pipeline
(datasets, belief encoder, world model) runs unchanged:

- Gymnasium API, flat float32 observation;
- 8 discrete actions = fixed planar force directions (matching the
  gridworld's action set, one-hot encoded downstream);
- per-episode physics randomization exposed in the ``info`` dict
  (``gravity``, ``friction``, ``mass`` — ground truth for belief probing);
- fully seeded through ``reset(seed=...)`` (AD-7).

Scene: an actuated ball ("agent") and two passive boxes on a bounded plane.
The agent pushes boxes via contact; sliding friction and box mass are the
episode-level latent physics. Lower process noise than CruelGridworld —
the regime where B12 belief conditioning and the B13 structured head
should show larger gains.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

_XML = """
<mujoco model="atlas_pointmass">
  <option timestep="0.02" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="1.2 1.2 .1" friction="{friction} 0.005 0.0001"/>
    <geom name="wall_n" type="box" pos="0 1.1 .05" size="1.2 .02 .06"/>
    <geom name="wall_s" type="box" pos="0 -1.1 .05" size="1.2 .02 .06"/>
    <geom name="wall_e" type="box" pos="1.1 0 .05" size=".02 1.2 .06"/>
    <geom name="wall_w" type="box" pos="-1.1 0 .05" size=".02 1.2 .06"/>
    <body name="agent" pos="0 0 .06">
      <joint name="ax" type="slide" axis="1 0 0"/>
      <joint name="ay" type="slide" axis="0 1 0"/>
      <geom name="agent_g" type="sphere" size=".06" mass="1.0" friction="{friction} 0.005 0.0001"/>
    </body>
    <body name="box0" pos=".4 .3 .06">
      <freejoint/>
      <geom name="box0_g" type="box" size=".06 .06 .06" mass="{mass0}" friction="{friction} 0.005 0.0001"/>
    </body>
    <body name="box1" pos="-.4 -.3 .06">
      <freejoint/>
      <geom name="box1_g" type="box" size=".06 .06 .06" mass="{mass1}" friction="{friction} 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="ax" gear="{gear}"/>
    <motor joint="ay" gear="{gear}"/>
  </actuator>
</mujoco>
"""

_DIRS = np.array(
    [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)], dtype=float
)
_DIRS = _DIRS / np.linalg.norm(_DIRS, axis=1, keepdims=True)


class MujocoPointMass(gym.Env):
    """Actuated ball + two passive boxes; episode-level friction/mass/gravity."""

    def __init__(
        self,
        randomize_physics: bool = False,
        friction_range: tuple[float, float] = (0.1, 0.6),
        mass_range: tuple[float, float] = (0.3, 1.5),
        gravity_range: tuple[float, float] = (7.0, 12.0),
        frame_skip: int = 5,
        include_noop: bool = False,
    ) -> None:
        super().__init__()
        self.randomize_physics = randomize_physics
        self.friction_range = friction_range
        self.mass_range = mass_range
        self.gravity_range = gravity_range
        self.frame_skip = frame_skip
        # v4.1: optional no-op action (index 8, zero ctrl) — COAST phases
        # let sliding friction reveal itself (mu*g decay), impossible under
        # continuous forcing.
        self.include_noop = include_noop
        self.action_space = spaces.Discrete(9 if include_noop else 8)
        self.observation_space = spaces.Box(low=-1.2, high=1.2, shape=(6,), dtype=np.float32)
        self._friction = 0.35
        self._mass0 = self._mass1 = 0.8
        self._gravity = 9.81
        self._build()

    def _build(self) -> None:
        xml = _XML.format(friction=self._friction, mass0=self._mass0, mass1=self._mass1, gear=8.0)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.opt.gravity[2] = -self._gravity
        self.data = mujoco.MjData(self.model)

    def _physics_info(self) -> dict[str, float]:
        return {
            "gravity": float(self._gravity),
            "friction": float(self._friction),
            "mass": float((self._mass0 + self._mass1) / 2),
        }

    def _get_obs(self) -> np.ndarray:
        agent = self.data.qpos[0:2]
        box0 = self.data.body("box0").xpos[:2]
        box1 = self.data.body("box1").xpos[:2]
        return np.concatenate([agent, box0, box1]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        if self.randomize_physics:
            self._friction = float(rng.uniform(*self.friction_range))
            self._mass0 = float(rng.uniform(*self.mass_range))
            self._mass1 = float(rng.uniform(*self.mass_range))
            self._gravity = float(rng.uniform(*self.gravity_range))
        self._build()
        mujoco.mj_resetData(self.model, self.data)
        # Randomized non-overlapping starts.
        # qpos layout: agent slides [0:2]; each box freejoint is
        # [x y z qw qx qy qz] -> box0 at [2:9], box1 at [9:16].
        self.data.qpos[0:2] = rng.uniform(-0.8, 0.8, size=2)
        self.data.qpos[2:4] = rng.uniform(-0.8, 0.8, size=2)
        self.data.qpos[9:11] = rng.uniform(-0.8, 0.8, size=2)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._physics_info()

    def step(self, action: int):
        a = int(action)
        self.data.ctrl[:] = 0.0 if (self.include_noop and a == 8) else _DIRS[a]
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        return self._get_obs(), 0.0, False, False, self._physics_info()
