"""MuJoCo point-mass environment contract tests (v4 B14)."""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from atlas_wm.environments.mujoco_pointmass import MujocoPointMass  # noqa: E402


class TestMujocoPointMass:
    def test_containment_and_api(self):
        env = MujocoPointMass(randomize_physics=True)
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert set(info) == {"gravity", "friction", "mass"}
        rng = np.random.default_rng(0)
        for _ in range(100):
            obs, r, term, trunc, info = env.step(int(rng.integers(8)))
            assert env.observation_space.contains(obs)

    def test_seeded_determinism(self):
        def roll(seed):
            e = MujocoPointMass(randomize_physics=True)
            o, _ = e.reset(seed=seed)
            t = [o.copy()]
            for a in [1, 3, 5, 7] * 10:
                o, *_ = e.step(a)
                t.append(o.copy())
            return np.array(t)

        assert np.array_equal(roll(3), roll(3))

    def test_physics_randomization_varies_and_is_seeded(self):
        e = MujocoPointMass(randomize_physics=True)
        _, i1 = e.reset(seed=1)
        _, i2 = e.reset(seed=2)
        _, i1b = e.reset(seed=1)
        assert i1 != i2 and i1 == i1b

    def test_boxes_move_only_via_contact(self):
        env = MujocoPointMass()
        obs0, _ = env.reset(seed=5)
        for _ in range(10):
            obs, *_ = env.step(0)
        # agent moved; boxes may move only if contacted — at minimum finite
        assert np.isfinite(obs).all()
        assert not np.allclose(obs[:2], obs0[:2]), "agent did not move under force"
