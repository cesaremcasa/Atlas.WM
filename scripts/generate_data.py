"""Generate transition data from CruelGridworld.

Default invocation reproduces the original fixed-physics dataset exactly
(global RNG, no per-episode randomization). Block 12 adds opt-in variable
physics: with ``--randomize-physics`` the environment resamples gravity and
friction every episode, and the active parameters are recorded per transition
in ``physics_params.npy`` as ground-truth labels for latent probing.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from atlas_wm.environments.cruel_gridworld import CruelGridworld

PHYSICS_KEYS = ("gravity", "friction_agent", "friction_box")


def generate_with_exploration(
    num_samples: int = 50000,
    randomize_physics: bool = False,
    process_noise_std: float = 0.0,
    seed: int | None = None,
    out_dir: str = "data/raw",
    episode_reset_prob: float = 0.1,
) -> bool:
    """Generate (obs, action, next_obs) transitions via random exploration.

    Args:
        num_samples: Number of transitions to generate.
        randomize_physics: Resample gravity/friction per episode (domain randomization).
        process_noise_std: Std of Gaussian process noise injected each step.
        seed: If set, makes generation reproducible (env resets + action/reset
            sampling are seeded). If None, legacy global-RNG behavior is used.
        out_dir: Output directory for the .npy files.
        episode_reset_prob: Probability of resetting the environment at each step.
            Default 0.1 gives mean episode length ~10. Use 0.02 for mean ~50 steps
            (better for PhysicsBeliefEncoder training with longer windows).

    Returns:
        True on success.
    """
    env = CruelGridworld(randomize_physics=randomize_physics, process_noise_std=process_noise_std)

    # Seeded RNG makes the variable-physics dataset reproducible (AD-7). When no
    # seed is given we fall back to the original global-RNG code path so the
    # default fixed-physics dataset is byte-identical to prior runs.
    rng = np.random.default_rng(seed) if seed is not None else None
    episode = 0

    def reset_env() -> dict:
        nonlocal episode
        episode += 1
        if seed is not None:
            _, info = env.reset(seed=seed + episode)
        else:
            _, info = env.reset()
        return info

    all_obs = []
    all_actions = []
    all_next_obs = []
    all_physics = []
    all_episode_ids: list[int] = []

    info = reset_env()
    obs = env._get_obs()

    for i in range(num_samples):
        reset_draw = rng.random() if rng is not None else np.random.random()
        if reset_draw < episode_reset_prob:
            info = reset_env()
            obs = env._get_obs()

        if rng is not None:
            action_idx = int(rng.integers(0, env.action_space.n))
        else:
            action_idx = np.random.randint(0, env.action_space.n)
        action_onehot = np.zeros(env.action_space.n)
        action_onehot[action_idx] = 1.0

        next_obs, _, _, _, info = env.step(action_idx)

        all_obs.append(obs.copy())
        all_actions.append(action_onehot)
        all_next_obs.append(next_obs.copy())
        all_physics.append([info[k] for k in PHYSICS_KEYS])
        all_episode_ids.append(episode)

        obs = next_obs

        if (i + 1) % 10000 == 0:
            print(f"Generated {i + 1}/{num_samples} samples...")

    obs_array = np.array(all_obs, dtype=np.float32)
    actions_array = np.array(all_actions, dtype=np.float32)
    next_obs_array = np.array(all_next_obs, dtype=np.float32)
    physics_array = np.array(all_physics, dtype=np.float32)
    episode_ids_array = np.array(all_episode_ids, dtype=np.int64)

    unique = len(np.unique(obs_array.reshape(obs_array.shape[0], -1), axis=0))
    diversity = 100 * unique / obs_array.shape[0]

    n_episodes = len(np.unique(episode_ids_array))
    avg_ep_len = len(obs_array) / max(n_episodes, 1)
    print(f"\nFinal: {len(obs_array)} samples, {unique} unique ({diversity:.1f}%)")
    print(f"Episodes: {n_episodes} (avg length {avg_ep_len:.1f} steps)")
    if randomize_physics:
        for j, key in enumerate(PHYSICS_KEYS):
            col = physics_array[:, j]
            print(f"  {key}: [{col.min():.3f}, {col.max():.3f}] (variable)")

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "observations.npy"), obs_array)
    np.save(os.path.join(out_dir, "actions.npy"), actions_array)
    np.save(os.path.join(out_dir, "next_observations.npy"), next_obs_array)
    np.save(os.path.join(out_dir, "physics_params.npy"), physics_array)
    np.save(os.path.join(out_dir, "episode_ids.npy"), episode_ids_array)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Atlas.WM transition data")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument(
        "--randomize-physics",
        action="store_true",
        help="Resample gravity/friction per episode (Block 12 domain randomization).",
    )
    parser.add_argument(
        "--process-noise-std",
        type=float,
        default=0.0,
        help="Std of Gaussian process noise injected into velocities each step.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible generation. Omit for legacy global-RNG behavior.",
    )
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument(
        "--episode-reset-prob",
        type=float,
        default=0.1,
        help="Probability of resetting per step. 0.1 → mean ~10 steps/episode (default). "
        "Use 0.02 for mean ~50 steps (needed for PhysicsBeliefEncoder with window_k>=20).",
    )
    args = parser.parse_args()

    generate_with_exploration(
        num_samples=args.num_samples,
        randomize_physics=args.randomize_physics,
        process_noise_std=args.process_noise_std,
        seed=args.seed,
        out_dir=args.out_dir,
        episode_reset_prob=args.episode_reset_prob,
    )


if __name__ == "__main__":
    main()
