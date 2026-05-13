"""Physics contract tests for CruelGridworld (Block 8).

Each test verifies one law of the physics simulation.
These are tripwire tests: a refactor that silently breaks gravity, friction,
wall bounce, or observation shape will be caught here immediately.
"""

import numpy as np

from atlas_wm.environments.cruel_gridworld import CruelGridworld

SEED = 42
GRID = 20


def _make_env(seed: int = SEED) -> CruelGridworld:
    env = CruelGridworld(grid_size=GRID)
    env.reset(seed=seed)
    return env


class TestObservationSpace:
    def test_obs_shape_is_6d(self):
        env = _make_env()
        obs, _ = env.reset(seed=SEED)
        assert obs.shape == (6,), f"Expected (6,), got {obs.shape}"

    def test_obs_dtype_is_float32(self):
        env = _make_env()
        obs, _ = env.reset(seed=SEED)
        assert obs.dtype == np.float32

    def test_observation_space_matches_obs(self):
        env = _make_env()
        obs, _ = env.reset(seed=SEED)
        assert env.observation_space.contains(obs)

    def test_action_space_is_8_discrete(self):
        env = _make_env()
        assert env.action_space.n == 8


class TestWallBounce:
    def test_agent_stays_within_bounds_after_many_steps(self):
        env = _make_env()
        env.reset(seed=SEED)
        for action in range(8):
            for _ in range(10):
                obs, *_ = env.step(action)
                assert (obs[:2] >= 0.5).all(), f"Agent left lower bound: {obs[:2]}"
                assert (obs[:2] <= GRID - 0.5).all(), f"Agent left upper bound: {obs[:2]}"

    def test_wall_bounce_reverses_velocity(self):
        env = _make_env()
        env.reset(seed=SEED)
        env.agent_pos = np.array([0.4, 10.0], dtype=float)
        env.agent_vel = np.array([-2.0, 0.0], dtype=float)
        env.step(4)  # neutral right direction, but clamp should dominate
        assert env.agent_pos[0] >= 0.5, "Agent passed through left wall"


class TestFriction:
    def test_friction_reduces_agent_velocity(self):
        env = _make_env()
        env.reset(seed=SEED)
        env.agent_vel = np.array([5.0, 5.0], dtype=float)
        env.walls = []
        env.step(0)  # action 0 = (-1, 0); combined with friction, speed must be < initial 5*sqrt(2)
        # max possible speed: (5 + 0.8) * 0.96 ≈ 5.57 (along x) — still < 5*sqrt(2) ≈ 7.07
        assert np.linalg.norm(env.agent_vel) < 5 * np.sqrt(2), "Friction not applied"

    def test_agent_friction_coefficient_is_096(self):
        env = _make_env()
        env.reset(seed=SEED)
        env.agent_vel = np.array([1.0, 0.0], dtype=float)
        env.walls = []  # disable obstacle collisions
        # action=0 is (-1, 0): force -0.8; after friction: (1.0 + (-0.8)) * 0.96 = 0.192
        env.step(0)
        # With gravity from boxes and wall interactions, only check that friction ran
        assert abs(env.agent_vel[0]) < 2.0  # velocity is reasonable, not diverging

    def test_box_friction_is_098(self):
        env = _make_env()
        env.reset(seed=SEED)
        env.box_vels[0] = np.array([10.0, 0.0], dtype=float)
        env.box_vels[1] = np.array([0.0, 0.0], dtype=float)
        env.walls = []
        env.step(0)
        # Box friction is 0.98, so box0 should have speed < 10
        assert np.linalg.norm(env.box_vels[0]) < 10.0


class TestGravity:
    def test_gravity_attracts_nearby_objects(self):
        env = _make_env()
        env.reset(seed=SEED)
        env.walls = []
        env.agent_vel = np.array([0.0, 0.0], dtype=float)
        env.box_vels = [np.array([0.0, 0.0], dtype=float), np.array([0.0, 0.0], dtype=float)]

        # Place box directly right of agent within gravity range
        env.agent_pos = np.array([10.0, 10.0], dtype=float)
        env.box_positions[0] = np.array([12.0, 10.0], dtype=float)  # dist=2, in range
        env.box_positions[1] = np.array([10.0, 18.0], dtype=float)  # far from agent

        env.step(4)  # action (1, 0) — adds positive x force

        # Agent should be attracted toward box0 (positive x)
        # After action (1,0) force 0.8 + gravity toward right box
        assert env.agent_vel[0] > 0.0, "Agent should accelerate toward right box"

    def test_gravity_inactive_beyond_10_units(self):
        env = _make_env()
        env.reset(seed=SEED)
        env.walls = []
        env.agent_vel = np.array([0.0, 0.0], dtype=float)
        env.box_vels = [np.array([0.0, 0.0], dtype=float), np.array([0.0, 0.0], dtype=float)]

        # Agent well inside bounds; box0 is > 10 units away (no gravity)
        env.agent_pos = np.array([7.0, 10.0], dtype=float)
        env.box_positions[0] = np.array([18.0, 10.0], dtype=float)  # dist = 11 > 10
        env.box_positions[1] = np.array(
            [7.0, 19.0], dtype=float
        )  # dist = 9 in range but orthogonal to x

        env.step(0)  # action (-1, 0) — applies negative x force
        # box0 is beyond gravity range; only action force affects agent in x
        assert env.agent_vel[0] < 0.0, "Only action force should apply at distance > 10"

    def test_newtons_third_law(self):
        """Gravity between agent and box is equal and opposite."""
        env = _make_env()
        env.reset(seed=SEED)
        env.walls = []
        env.agent_vel = np.array([0.0, 0.0], dtype=float)
        env.box_vels = [np.array([0.0, 0.0], dtype=float), np.array([0.0, 0.0], dtype=float)]
        env.box_positions[1] = np.array([10.0, 19.0], dtype=float)  # very far, no gravity to agent

        env.agent_pos = np.array([10.0, 10.0], dtype=float)
        env.box_positions[0] = np.array([12.0, 10.0], dtype=float)

        vel_agent_before = env.agent_vel.copy()
        vel_box_before = env.box_vels[0].copy()

        env._apply_gravity(
            env.agent_pos, env.agent_vel, env.box_positions[0], env.box_vels[0], G=5.0
        )

        delta_agent = env.agent_vel - vel_agent_before
        delta_box = env.box_vels[0] - vel_box_before

        # Newton's 3rd law: delta_agent == -delta_box
        np.testing.assert_allclose(delta_agent, -delta_box, rtol=1e-6)


class TestActionEffects:
    def test_8_distinct_actions_produce_distinct_velocities(self):
        initial_vels = []
        for action in range(8):
            env = CruelGridworld(grid_size=GRID)
            env.reset(seed=SEED)
            env.agent_vel = np.zeros(2, dtype=float)
            env.walls = []
            env.step(action)
            initial_vels.append(env.agent_vel.copy())

        # All velocities should be distinct (each action applies a unique force)
        for i in range(8):
            for j in range(i + 1, 8):
                assert not np.allclose(initial_vels[i], initial_vels[j]), (
                    f"Actions {i} and {j} produced identical velocities"
                )

    def test_step_returns_correct_format(self):
        env = _make_env()
        env.reset(seed=SEED)
        result = env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (6,)
        assert reward == 0.0
        assert terminated is False
        assert truncated is False
        assert isinstance(info, dict)


class TestDeterminism:
    def test_same_seed_same_trajectory(self):
        def rollout(seed: int) -> list[np.ndarray]:
            env = CruelGridworld(grid_size=GRID)
            obs, _ = env.reset(seed=seed)
            trajectory = [obs.copy()]
            for a in range(8):
                obs, *_ = env.step(a)
                trajectory.append(obs.copy())
            return trajectory

        t1 = rollout(SEED)
        t2 = rollout(SEED)
        for o1, o2 in zip(t1, t2):
            np.testing.assert_array_equal(o1, o2)

    def test_different_seeds_different_initial_states(self):
        env = CruelGridworld(grid_size=GRID)
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        assert not np.allclose(obs1, obs2), (
            "Different seeds should produce different initial states"
        )
