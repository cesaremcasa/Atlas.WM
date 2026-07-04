"""Active-exploration policy tests (v4 B11).

The information-seeking policy must beat random exploration on the measured
information-rate proxies from B10 — that advantage is the block's entire
point, so it is locked in as a regression test.
"""

import numpy as np
import torch

from atlas_wm.data.dynamics_features import build_dynamics_features
from atlas_wm.data.exploration import InfoSeekingPolicy
from atlas_wm.environments.cruel_gridworld import CruelGridworld


def _rollout_metrics(policy_name: str, episodes: int = 40, steps: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    med_errs, valid_fracs = [], []
    for _ in range(episodes):
        env = CruelGridworld(randomize_physics=True, process_noise_std=0.05)
        obs, info = env.reset(seed=int(rng.integers(2**31)))
        pol = InfoSeekingPolicy(rng=np.random.default_rng(int(rng.integers(2**31))))
        traj, acts = [obs.copy()], []
        for _ in range(steps):
            a = pol.act(obs) if policy_name == "active" else int(rng.integers(8))
            obs, *_ = env.step(a)
            traj.append(obs.copy())
            acts.append(a)
        obs_w = torch.tensor(np.array(traj) / 20.0, dtype=torch.float32).unsqueeze(0)
        act_w = torch.zeros(1, len(traj), 8)
        for k, a in enumerate(acts):
            act_w[0, k, a] = 1.0
        feats = build_dynamics_features(obs_w, act_w)
        valid_fracs.append(float(feats[0, :, 7].mean()))
        final_median = float(feats[0, -1, 26])
        if final_median > 0:
            med_errs.append(abs(final_median - info["friction_agent"]))
    return float(np.mean(med_errs)), float(np.mean(valid_fracs))


class TestInfoSeekingPolicy:
    def test_actions_are_valid(self):
        pol = InfoSeekingPolicy(rng=np.random.default_rng(0))
        rng = np.random.default_rng(1)
        for _ in range(200):
            obs = rng.uniform(0, 20, size=6)
            a = pol.act(obs)
            assert 0 <= a < 8

    def test_deterministic_given_rng(self):
        obs_seq = np.random.default_rng(3).uniform(0, 20, size=(50, 6))
        a1 = [InfoSeekingPolicy(rng=np.random.default_rng(5)).act(o) for o in obs_seq[:1]]
        p1 = InfoSeekingPolicy(rng=np.random.default_rng(5))
        p2 = InfoSeekingPolicy(rng=np.random.default_rng(5))
        assert [p1.act(o) for o in obs_seq] == [p2.act(o) for o in obs_seq]
        assert a1  # silence unused warning

    def test_active_beats_random_on_friction_precision(self):
        # The block's core claim: information-seeking data yields materially
        # more precise per-episode friction estimates (measured 2.4x at scale).
        # MEAN error is the right statistic: random exploration's failure mode
        # is a heavy tail of uninformative episodes, which medians hide.
        err_active, _ = _rollout_metrics("active")
        err_random, _ = _rollout_metrics("random")
        assert err_active < 0.75 * err_random, (
            f"active exploration lost its information advantage: "
            f"active mean |err| = {err_active:.4f} vs random {err_random:.4f}"
        )


class TestCoastPushPolicy:
    """v4.1: MuJoCo COAST/PUSH policy contract."""

    def test_emits_noop_during_coast_and_valid_actions(self):
        from atlas_wm.data.exploration import CoastPushPolicy

        pol = CoastPushPolicy(rng=np.random.default_rng(0), epsilon=0.0)
        obs = np.array([0.0, 0.0, 0.5, 0.5, -0.5, -0.5])
        acts = [pol.act(obs) for _ in range(24)]
        assert all(0 <= a <= 8 for a in acts)
        assert all(a == 8 for a in acts[14:24]), "coast phase must be no-ops"
        assert all(a != 8 for a in acts[:14]), "push phase must command"
