"""Variable physics & process noise tests (Block 12).

Covers AD-2 domain randomization and AD-7 determinism:
  - Default (fixed-physics) env is byte-identical to the original behavior.
  - Per-episode randomization stays within configured ranges.
  - Seeded randomization and seeded process noise are reproducible.
  - Process noise perturbs trajectories only when enabled.
  - Active physics parameters are surfaced in the info dict for probing.
"""

import numpy as np

from atlas_wm.environments.cruel_gridworld import CruelGridworld

SEED = 7


class TestDefaultPhysicsUnchanged:
    def test_default_constants_are_canonical(self):
        env = CruelGridworld()
        assert env.gravity == 5.0
        assert env.friction_agent == 0.96
        assert env.friction_box == 0.98
        assert env.process_noise_std == 0.0
        assert env.randomize_physics is False

    def test_fixed_physics_constant_across_resets(self):
        env = CruelGridworld()
        for seed in range(5):
            _, info = env.reset(seed=seed)
            assert info["gravity"] == 5.0
            assert info["friction_agent"] == 0.96
            assert info["friction_box"] == 0.98

    def test_default_trajectory_matches_zero_noise_env(self):
        """A default env and an explicit zero-noise env must be byte-identical."""
        a = CruelGridworld()
        b = CruelGridworld(process_noise_std=0.0, randomize_physics=False)
        obs_a, _ = a.reset(seed=SEED)
        obs_b, _ = b.reset(seed=SEED)
        np.testing.assert_array_equal(obs_a, obs_b)
        for action in range(8):
            oa, *_ = a.step(action)
            ob, *_ = b.step(action)
            np.testing.assert_array_equal(oa, ob)


class TestRandomizedPhysics:
    def test_randomized_physics_within_ranges(self):
        env = CruelGridworld(
            randomize_physics=True,
            gravity_range=(2.0, 8.0),
            friction_agent_range=(0.90, 0.99),
            friction_box_range=(0.95, 0.995),
        )
        for seed in range(20):
            _, info = env.reset(seed=seed)
            assert 2.0 <= info["gravity"] <= 8.0
            assert 0.90 <= info["friction_agent"] <= 0.99
            assert 0.95 <= info["friction_box"] <= 0.995

    def test_randomized_physics_varies_across_seeds(self):
        env = CruelGridworld(randomize_physics=True)
        gravities = {round(env.reset(seed=s)[1]["gravity"], 6) for s in range(10)}
        assert len(gravities) > 1, "Randomization produced identical gravity for all seeds"

    def test_randomized_physics_reproducible_for_same_seed(self):
        env = CruelGridworld(randomize_physics=True)
        _, info1 = env.reset(seed=SEED)
        _, info2 = env.reset(seed=SEED)
        assert info1 == info2

    def test_step_info_reports_active_physics(self):
        env = CruelGridworld(randomize_physics=True)
        _, reset_info = env.reset(seed=SEED)
        _, _, _, _, step_info = env.step(0)
        assert step_info == reset_info


class TestProcessNoise:
    def test_zero_noise_is_deterministic(self):
        def rollout():
            env = CruelGridworld(process_noise_std=0.0)
            env.reset(seed=SEED)
            return [env.step(a)[0].copy() for a in range(8)]

        for o1, o2 in zip(rollout(), rollout()):
            np.testing.assert_array_equal(o1, o2)

    def test_noise_reproducible_for_same_seed(self):
        def rollout():
            env = CruelGridworld(process_noise_std=0.1)
            env.reset(seed=SEED)
            return [env.step(a)[0].copy() for a in range(8)]

        for o1, o2 in zip(rollout(), rollout()):
            np.testing.assert_array_equal(o1, o2)

    def test_noise_changes_trajectory(self):
        clean = CruelGridworld(process_noise_std=0.0)
        noisy = CruelGridworld(process_noise_std=0.2)
        clean.reset(seed=SEED)
        noisy.reset(seed=SEED)
        diverged = False
        for action in range(8):
            oc, *_ = clean.step(action)
            on, *_ = noisy.step(action)
            if not np.allclose(oc, on):
                diverged = True
        assert diverged, "Process noise did not perturb the trajectory"
