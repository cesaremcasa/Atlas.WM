"""Chaos physics tripwire (v4 B4).

Referenced by .github/workflows/chaos-physics.yml since Block 8, but never
committed — the workflow failed on every scheduled run. This implements it:
a randomized sweep over seeds and physics parameters asserting the invariants
that the deterministic contract tests (tests/test_physics.py) cannot cover
exhaustively.

Invariants checked per episode:
  1. Containment: every observation stays inside the declared space.
  2. Finiteness: no NaN/inf positions.
  3. Determinism: the same seed replays to a bit-identical trajectory.
  4. Dissipation: with no input force and no nearby boxes, agent speed decays.

Exit code 1 on any violation.

Usage::

    python scripts/chaos_physics.py --episodes 200 --steps 150
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from atlas_wm.environments.cruel_gridworld import CruelGridworld


def _rollout(seed: int, steps: int, rng_actions: np.ndarray) -> np.ndarray:
    env = CruelGridworld(randomize_physics=True)
    obs, _ = env.reset(seed=seed)
    traj = [obs.copy()]
    for a in rng_actions[:steps]:
        obs, *_ = env.step(int(a))
        traj.append(obs.copy())
    return np.array(traj)


def run_chaos(episodes: int, steps: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    violations: list[str] = []

    for ep in range(episodes):
        ep_seed = int(rng.integers(2**31))
        actions = rng.integers(0, 8, size=steps)

        env = CruelGridworld(randomize_physics=True)
        obs, _ = env.reset(seed=ep_seed)
        traj = [obs.copy()]
        for a in actions:
            obs, *_ = env.step(int(a))
            traj.append(obs.copy())
            if not np.all(np.isfinite(obs)):
                violations.append(f"ep {ep} (seed {ep_seed}): non-finite obs {obs}")
                break
            if not env.observation_space.contains(obs):
                violations.append(f"ep {ep} (seed {ep_seed}): obs left the grid: {obs}")
                break

        # Determinism: replay with the same seed and actions must be identical.
        replay = _rollout(ep_seed, len(traj) - 1, actions)
        if not np.array_equal(np.array(traj), replay):
            violations.append(f"ep {ep} (seed {ep_seed}): replay diverged from original")

    # Dissipation: an isolated coasting agent must slow down.
    env = CruelGridworld()
    env.reset(seed=seed)
    env.walls = []
    env.agent_pos = np.array([10.0, 10.0])
    env.agent_vel = np.array([3.0, 0.0])
    env.box_positions = [np.array([1.0, 1.0]), np.array([19.0, 19.0])]
    env.box_vels = [np.zeros(2), np.zeros(2)]
    speed0 = float(np.linalg.norm(env.agent_vel))
    # action pairs (0, 4) = (-1,0) and (1,0) cancel over two steps
    env.step(0)
    env.step(4)
    if float(np.linalg.norm(env.agent_vel)) >= speed0:
        violations.append("dissipation: coasting agent did not lose speed")

    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    violations = run_chaos(args.episodes, args.steps, args.seed)
    if violations:
        print(f"CHAOS TRIPWIRE FAILED — {len(violations)} violation(s):")
        for v in violations[:20]:
            print(f"  - {v}")
        sys.exit(1)
    print(
        f"Chaos tripwire passed: {args.episodes} episodes x {args.steps} steps, "
        "containment + finiteness + determinism + dissipation all hold."
    )


if __name__ == "__main__":
    main()
