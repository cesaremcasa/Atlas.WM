"""Information-seeking exploration policy for physics identification (v4 B11).

Random-policy data identifies physics poorly not because the environment
hides it, but because random actions rarely produce INFORMATIVE steps. B10
quantified exactly what makes a step informative:

- ``friction_agent``: high excitation (``||v + 0.8u||² > 0.25``), agent
  farther than the gravity gate from both boxes, no wall bounce — the
  conditions under which the per-step decay ratio is unbiased;
- ``gravity`` / ``friction_box``: boxes only reveal their physics while the
  inter-object attraction is active (1 < dist < 10) and while they are
  moving (friction is visible as decay of that motion).

This policy greedily maximizes those information-rate proxies with a
two-phase cycle (ASID-style active system identification, with the Fisher
objective replaced by the measured B10 proxies; a learned
information-maximizing policy is the natural extension):

- **MEASURE** phase: steer toward open space away from boxes and walls,
  alternating push directions to keep excitation high — clean friction
  ratios accumulate.
- **STIR** phase: approach the nearest box into the gravity band to
  energize it, then the next MEASURE phase both observes the box coasting
  (friction_box) and collects clean agent ratios again.

The policy is deterministic given its RNG and uses only the observation.
"""

from __future__ import annotations

import numpy as np

FORCES = np.array(
    [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)], dtype=float
)
_UNIT_FORCES = FORCES / np.linalg.norm(FORCES, axis=1, keepdims=True)


class InfoSeekingPolicy:
    """Two-phase information-seeking policy over CruelGridworld observations."""

    def __init__(
        self,
        rng: np.random.Generator,
        grid_size: float = 20.0,
        measure_steps: int = 30,
        stir_steps: int = 10,
        margin: float = 3.0,
        epsilon: float = 0.1,
    ) -> None:
        self.rng = rng
        self.grid_size = grid_size
        self.measure_steps = measure_steps
        self.stir_steps = stir_steps
        self.margin = margin
        self.epsilon = epsilon
        self._t = 0
        self._flip = 1.0
        self._angle = 0.0

    def reset(self) -> None:
        self._t = 0

    def _phase(self) -> str:
        cycle = self.measure_steps + self.stir_steps
        return "measure" if (self._t % cycle) < self.measure_steps else "stir"

    def act(self, obs: np.ndarray) -> int:
        """Pick one of the 8 discrete actions for the current observation."""
        agent, box0, box1 = obs[:2], obs[2:4], obs[4:6]
        phase = self._phase()
        self._t += 1

        # Exploration jitter keeps the action distribution from degenerating
        # (and decorrelates consecutive pre-vectors for ratio conditioning).
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(FORCES)))

        if phase == "stir":
            # Head at the nearest box to drive it via the gravity coupling.
            target = box0 if np.linalg.norm(agent - box0) <= np.linalg.norm(agent - box1) else box1
            desired = target - agent
        else:
            # MEASURE: oscillating dash. The per-step ratio error scales as
            # sigma_noise / ||pre||, and a random walk keeps ||pre|| ≈ 1. A
            # 3-step dash-and-reverse along the tangential axis (perpendicular
            # to the away-from-boxes direction) builds |v| ≈ 2 while keeping
            # the position local — no wall hits, boxes stay outside the gate,
            # and ratio precision doubles or better.
            away = (agent - box0) + (agent - box1)
            norm = np.linalg.norm(away)
            away = away / norm if norm > 1e-8 else np.array([1.0, 0.0])
            if self._t % 3 == 0:
                self._flip = -self._flip
                # Rotate the dash axis each reversal (golden-angle rosette):
                # a FIXED dash line that happens to cross an unobservable
                # obstacle hits it repeatedly and corrupts the ratio median
                # for the whole episode; a rosette turns such collisions into
                # sparse outliers the median absorbs.
                self._angle = (self._angle + 2.4) % (2 * np.pi)
            base = np.array([-away[1], away[0]])
            c, s_ = np.cos(self._angle), np.sin(self._angle)
            tangent = np.array([c * base[0] - s_ * base[1], s_ * base[0] + c * base[1]])
            desired = tangent * self._flip + 0.4 * away
            # Wall repulsion: push back inside the margin band.
            for i in range(2):
                if agent[i] < self.margin:
                    desired[i] += 1.5
                elif agent[i] > self.grid_size - self.margin:
                    desired[i] -= 1.5

        norm = np.linalg.norm(desired)
        if norm < 1e-8:
            return int(self.rng.integers(len(FORCES)))
        return int(np.argmax(_UNIT_FORCES @ (desired / norm)))


class CoastPushPolicy:
    """MuJoCo information-seeking policy (v4.1): PUSH -> COAST cycle.

    Random-policy MuJoCo data identifies nothing (R^2 ~ 0, measured):
    friction needs free sliding (never happens under continuous forcing)
    and mass needs contacts (rare under random actions). This policy
    manufactures both informative events:

    - PUSH: drive at the nearest box (contacts -> mass response);
    - COAST: no-op actions (index 8) after building speed -> pure sliding
      decay reveals the mu*g friction product.

    Requires MujocoPointMass(include_noop=True).
    """

    NOOP = 8

    def __init__(self, rng, push_steps: int = 14, coast_steps: int = 10, epsilon: float = 0.08):
        self.rng = rng
        self.push_steps = push_steps
        self.coast_steps = coast_steps
        self.epsilon = epsilon
        self._t = 0

    def reset(self) -> None:
        self._t = 0

    def act(self, obs) -> int:
        agent, box0, box1 = obs[:2], obs[2:4], obs[4:6]
        t = self._t
        self._t += 1
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(9))
        cycle = self.push_steps + self.coast_steps
        if (t % cycle) >= self.push_steps:
            return self.NOOP  # COAST
        target = box0 if np.linalg.norm(agent - box0) <= np.linalg.norm(agent - box1) else box1
        desired = target - agent
        n = np.linalg.norm(desired)
        if n < 1e-8:
            return int(self.rng.integers(8))
        return int(np.argmax(_UNIT_FORCES @ (desired / n)))
