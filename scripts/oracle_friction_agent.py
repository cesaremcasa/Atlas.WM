"""Oracle probe: closed-form identifiability check for ``friction_agent``.

This is the reproducible evidence behind the v4 retraction of the Block-14
claim that ``friction_agent`` is "not identifiable" (see docs/MODEL_CARD.md).
That claim rested on an MLP oracle that was never committed to the repo and
whose MSE objective is destroyed by heavy-tailed bounce outliers.

Method — median of per-step decay ratios, position-only observations:

    During step k the env computes (absent bounces and gravity)
        v_k = friction_agent * (v_{k-1} + 0.8 * u_k)
    and positions integrate as pos_k = pos_{k-1} + v_k * dt, so
    v_k = (pos_k - pos_{k-1}) / dt is exactly recoverable from observations.
    The per-step ratio
        r_k = <v_k, v_{k-1} + 0.8 u_k> / ||v_{k-1} + 0.8 u_k||^2
    equals friction_agent * (1 + gravity projection). Robustness comes from
    three filters plus the median:
      - steps whose start/end position sits on the wall-clamp boundary are
        dropped (observable wall bounces reflect velocity);
      - steps with weak excitation (||pre||^2 < 0.25) are dropped;
      - steps with a box within ``dist_gate`` are dropped (gravity bias);
      - the MEDIAN over the surviving ratios ignores the remaining outliers
        (unobservable obstacle bounces). A least-squares fit does not — that
        is what broke the original oracle.

Measured on the post-B1 environment (boxes contained), 400 random-policy
episodes x 50 steps, physics randomized per episode:

    dist_gate=5.0 (default):  R^2 = 0.85, MAE = 0.004, 94% episode coverage
    dist_gate=8.0 (strict):   R^2 = 0.98, MAE = 0.001, 68% episode coverage
    dist_gate disabled:       R^2 = 0.62, MAE = 0.008, 100% coverage

Usage::

    python scripts/oracle_friction_agent.py --episodes 400 --steps 50
"""

from __future__ import annotations

import argparse

import numpy as np

from atlas_wm.environments.cruel_gridworld import CruelGridworld

FORCES = np.array(
    [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)], dtype=float
)
ACTION_GAIN = 0.8
MIN_EXCITATION = 0.25  # ||v_{k-1} + 0.8 u_k||^2 below this is ill-conditioned
MIN_RATIOS = 8


def estimate_friction_agent(
    obs_traj: np.ndarray,
    actions: np.ndarray,
    dt: float = 0.5,
    grid_size: float = 20.0,
    dist_gate: float | None = 5.0,
) -> float | None:
    """Median-of-ratios estimate from one episode's positions and actions.

    Args:
        obs_traj: [T+1, 6] raw (unnormalized) observations, obs_traj[0] = reset.
        actions:  [T] integer actions; actions[k] produced obs_traj[k+1].
        dist_gate: drop steps with a box nearer than this (None disables).

    Returns the estimate, or None if fewer than MIN_RATIOS usable steps.
    """
    pos = obs_traj[:, :2]
    vel = np.diff(pos, axis=0) / dt  # vel[k] = velocity applied during step k+1

    ratios = []
    for k in range(1, len(vel)):
        # Wall bounces are observable: the clamp pins a coordinate to the
        # boundary band. Drop steps touching it (velocity was reflected).
        endpoints = np.concatenate([pos[k], pos[k + 1]])
        if np.any(np.isclose(endpoints, 0.5)) or np.any(np.isclose(endpoints, grid_size - 0.5)):
            continue

        pre = vel[k - 1] + ACTION_GAIN * FORCES[actions[k]]
        den = float(pre @ pre)
        if den < MIN_EXCITATION:
            continue

        if dist_gate is not None:
            start = obs_traj[k]
            d_box0 = float(np.linalg.norm(start[:2] - start[2:4]))
            d_box1 = float(np.linalg.norm(start[:2] - start[4:6]))
            if min(d_box0, d_box1) <= dist_gate:
                continue

        ratios.append(float(pre @ vel[k]) / den)

    if len(ratios) < MIN_RATIOS:
        return None
    return float(np.median(ratios))


def run_oracle(
    episodes: int = 400,
    steps: int = 50,
    seed: int = 0,
    dist_gate: float | None = 5.0,
) -> dict[str, float]:
    """Roll random-policy episodes with randomized physics; report R² and MAE."""
    rng = np.random.default_rng(seed)
    truths, estimates = [], []
    skipped = 0

    for _ in range(episodes):
        env = CruelGridworld(randomize_physics=True)
        obs, info = env.reset(seed=int(rng.integers(2**31)))
        traj = [obs.astype(float)]
        actions = []
        for _ in range(steps):
            a = int(rng.integers(8))
            obs, *_ = env.step(a)
            traj.append(obs.astype(float))
            actions.append(a)

        est = estimate_friction_agent(np.array(traj), np.array(actions), dist_gate=dist_gate)
        if est is None:
            skipped += 1
            continue
        truths.append(info["friction_agent"])
        estimates.append(est)

    gt = np.array(truths)
    hat = np.clip(np.array(estimates), 0.5, 1.1)
    ss_res = float(((gt - hat) ** 2).sum())
    ss_tot = float(((gt - gt.mean()) ** 2).sum())
    return {
        "r2": 1.0 - ss_res / ss_tot,
        "mae": float(np.abs(gt - hat).mean()),
        "episodes_used": float(len(gt)),
        "episodes_skipped": float(skipped),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist-gate",
        type=float,
        default=5.0,
        help="Drop steps with a box nearer than this (0 disables; 8.0 = strict)",
    )
    args = parser.parse_args()

    gate = args.dist_gate if args.dist_gate > 0 else None
    result = run_oracle(episodes=args.episodes, steps=args.steps, seed=args.seed, dist_gate=gate)
    print(
        f"friction_agent oracle over {int(result['episodes_used'])} episodes "
        f"({int(result['episodes_skipped'])} skipped, insufficient usable steps):\n"
        f"  R²  = {result['r2']:.3f}\n"
        f"  MAE = {result['mae']:.4f}"
    )
    if result["r2"] > 0.5:
        print(
            "\nfriction_agent IS identifiable under random-policy data. "
            "The Block-14 exclusion is retracted; see docs/MODEL_CARD.md."
        )


if __name__ == "__main__":
    main()
