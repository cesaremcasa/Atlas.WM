"""Tests for the engineered dynamics features and belief v2 pieces (v4 B10)."""

import numpy as np
import torch

from atlas_wm.data.dynamics_features import (
    ACTION_GAIN,
    N_FEATURES,
    build_dynamics_features,
)
from atlas_wm.models.physics_belief import DistributionalPhysicsHead, gaussian_nll
from atlas_wm.training.objectives import info_nce

FORCES = np.array(
    [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)], dtype=float
)


def _clean_agent_trajectory(friction=0.94, steps=12, dt=0.5, seed=0):
    """Agent-only dynamics, far from boxes and walls: v' = f*(v + 0.8u)."""
    rng = np.random.default_rng(seed)
    pos = np.array([10.0, 10.0])
    vel = np.array([0.6, -0.4])
    boxes = np.array([1.0, 1.0, 19.0, 19.0])  # parked far away (> gravity range)
    obs, actions = [np.concatenate([pos, boxes])], []
    for _ in range(steps):
        a = int(rng.integers(8))
        vel = friction * (vel + ACTION_GAIN * FORCES[a])
        pos = pos + vel * dt
        obs.append(np.concatenate([pos, boxes]))
        actions.append(a)
    obs_w = torch.tensor(np.array(obs) / 20.0, dtype=torch.float32).unsqueeze(0)
    act_w = torch.zeros(1, steps + 1, 8)
    for k, a in enumerate(actions):
        act_w[0, k, a] = 1.0  # dataset convention: action taken AT frame k
    return obs_w, act_w


class TestDynamicsFeatures:
    def test_shapes(self):
        obs_w, act_w = _clean_agent_trajectory()
        feats = build_dynamics_features(obs_w, act_w)
        assert feats.shape == (1, obs_w.shape[1] - 2, N_FEATURES)
        assert torch.isfinite(feats).all()

    def test_friction_ratio_recovers_exact_friction(self):
        # On clean dynamics (no gravity in range, no bounces) every valid
        # per-step ratio IS the friction coefficient — the oracle's identity.
        for friction in (0.90, 0.94, 0.99):
            obs_w, act_w = _clean_agent_trajectory(friction=friction)
            feats = build_dynamics_features(obs_w, act_w)
            ratio, valid = feats[0, :, 6], feats[0, :, 7].bool()
            assert valid.any(), "no valid steps on a clean trajectory"
            np.testing.assert_allclose(
                ratio[valid].numpy(),
                friction,
                rtol=1e-4,
                err_msg=f"ratio feature does not recover friction={friction}",
            )

    def test_boundary_steps_flagged_invalid(self):
        obs_w, act_w = _clean_agent_trajectory()
        # Pin one frame's agent position onto the wall-clamp boundary.
        obs_w = obs_w.clone()
        obs_w[0, 5, 0] = 0.5 / 20.0
        feats = build_dynamics_features(obs_w, act_w)
        valid = feats[0, :, 7].bool()
        # Steps whose start or end touches frame 5 must be invalid.
        assert not valid[3] and not valid[4], "boundary bounce not filtered"

    def test_scale_invariance_of_ratio(self):
        # The ratio is computed in raw units internally; feeding the same
        # trajectory must give identical ratios regardless of tensor dtype.
        obs_w, act_w = _clean_agent_trajectory(friction=0.93)
        f32 = build_dynamics_features(obs_w, act_w)[0, :, 6]
        f64 = build_dynamics_features(obs_w.double().float(), act_w)[0, :, 6]
        torch.testing.assert_close(f32, f64)


class TestDistributionalHead:
    def test_shapes_and_nll_gradient(self):
        head = DistributionalPhysicsHead(d_slow=8, n_physics=3)
        z = torch.randn(16, 8, requires_grad=True)
        mu, logvar = head(z)
        assert mu.shape == (16, 3) and logvar.shape == (16, 3)
        loss = gaussian_nll(mu, logvar, torch.randn(16, 3))
        loss.backward()
        assert z.grad is not None

    def test_nll_prefers_calibrated_uncertainty(self):
        target = torch.zeros(64, 1)
        mu = torch.ones(64, 1)  # constant error of 1
        nll_overconfident = gaussian_nll(mu, torch.full((64, 1), -4.0), target)
        nll_calibrated = gaussian_nll(mu, torch.zeros(64, 1), target)  # sigma=1
        assert nll_calibrated < nll_overconfident, (
            "NLL must penalize overconfidence under fixed error"
        )


class TestInfoNCE:
    def test_aligned_views_beat_shuffled_views(self):
        g = torch.Generator().manual_seed(0)
        z = torch.randn(32, 8, generator=g)
        aligned = info_nce(z, z + 0.01 * torch.randn(32, 8, generator=g))
        shuffled = info_nce(z, z[torch.randperm(32, generator=g)])
        assert aligned < shuffled
