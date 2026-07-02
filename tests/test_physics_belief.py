"""Tests for PhysicsBeliefEncoder, PhysicsHead, EpisodeATLASDataset, and probe_from_arrays.

Block 14 additions: GRU-based physics belief encoder and associated dataset/probe utilities.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from atlas_wm.models.physics_belief import PhysicsBeliefEncoder, PhysicsHead


class TestPhysicsBeliefEncoder:
    def test_output_shape(self):
        enc = PhysicsBeliefEncoder(obs_dim=14, d_slow=8, hidden_dim=64)
        x = torch.randn(4, 10, 14)  # [B=4, K=10, obs_dim=14]
        z = enc(x)
        assert z.shape == (4, 8), f"Expected (4, 8), got {z.shape}"

    def test_batch_size_one(self):
        enc = PhysicsBeliefEncoder(obs_dim=6, d_slow=8)
        x = torch.randn(1, 5, 6)
        z = enc(x)
        assert z.shape == (1, 8)

    def test_different_window_lengths(self):
        enc = PhysicsBeliefEncoder(obs_dim=14, d_slow=16, hidden_dim=128)
        for k in (1, 5, 20, 50):
            x = torch.randn(2, k, 14)
            z = enc(x)
            assert z.shape == (2, 16)

    def test_gradients_flow(self):
        enc = PhysicsBeliefEncoder(obs_dim=14, d_slow=8)
        x = torch.randn(3, 10, 14, requires_grad=False)
        z = enc(x)
        loss = z.sum()
        loss.backward()
        for p in enc.parameters():
            assert p.grad is not None

    def test_different_inputs_give_different_outputs(self):
        enc = PhysicsBeliefEncoder(obs_dim=6, d_slow=8)
        enc.eval()
        with torch.no_grad():
            z1 = enc(torch.randn(1, 10, 6))
            z2 = enc(torch.randn(1, 10, 6))
        assert not torch.allclose(z1, z2), "Different inputs should give different outputs"

    def test_deterministic_in_eval_mode(self):
        enc = PhysicsBeliefEncoder(obs_dim=6, d_slow=8)
        enc.eval()
        x = torch.randn(2, 10, 6)
        with torch.no_grad():
            z1 = enc(x)
            z2 = enc(x)
        torch.testing.assert_close(z1, z2)


class TestPhysicsHead:
    def test_output_shape_default(self):
        head = PhysicsHead(d_slow=8, n_physics=3)
        z = torch.randn(5, 8)
        out = head(z)
        assert out.shape == (5, 3)

    def test_output_shape_custom_n(self):
        head = PhysicsHead(d_slow=16, n_physics=2)
        z = torch.randn(10, 16)
        out = head(z)
        assert out.shape == (10, 2)

    def test_gradients_flow(self):
        head = PhysicsHead(d_slow=8, n_physics=3)
        z = torch.randn(4, 8)
        out = head(z)
        out.sum().backward()
        assert head.linear.weight.grad is not None

    def test_combined_encoder_head(self):
        enc = PhysicsBeliefEncoder(obs_dim=14, d_slow=8)
        head = PhysicsHead(d_slow=8, n_physics=3)
        x = torch.randn(4, 10, 14)
        z = enc(x)
        physics_hat = head(z)
        assert physics_hat.shape == (4, 3)


class TestEpisodeATLASDatasetBuildValidIndices:
    """Test the vectorized boundary detection logic directly without disk I/O."""

    def _make_dummy_dataset(
        self,
        tmp_path,
        n: int,
        episode_ids: np.ndarray,
        window_k: int = 3,
        include_physics: bool = False,
    ):
        from atlas_wm.data.episode_dataset import EpisodeATLASDataset

        split = "train"
        obs = np.random.randn(n, 6).astype(np.float32)
        actions = np.random.randn(n, 8).astype(np.float32)
        next_obs = np.random.randn(n, 6).astype(np.float32)

        np.save(tmp_path / f"{split}_obs.npy", obs)
        np.save(tmp_path / f"{split}_actions.npy", actions)
        np.save(tmp_path / f"{split}_next_obs.npy", next_obs)
        np.save(tmp_path / f"{split}_episode_ids.npy", episode_ids)
        if include_physics:
            physics = np.random.randn(n, 3).astype(np.float32)
            np.save(tmp_path / f"{split}_physics.npy", physics)

        return EpisodeATLASDataset(str(tmp_path), split=split, window_k=window_k)

    def test_single_episode_all_valid(self, tmp_path):
        n = 20
        episode_ids = np.ones(n, dtype=np.int64)
        ds = self._make_dummy_dataset(tmp_path, n, episode_ids, window_k=3)
        # With k=3, indices 2..19 are valid (all same episode) → 18 windows
        assert len(ds) == n - 2

    def test_two_episodes_no_cross_boundary(self, tmp_path):
        # Episode 1: indices 0-9, Episode 2: indices 10-19
        episode_ids = np.array([1] * 10 + [2] * 10, dtype=np.int64)
        ds = self._make_dummy_dataset(tmp_path, 20, episode_ids, window_k=3)
        # Episode 1: i=2..9 valid (8 windows).
        # Episode 2: window [i-2..i] must lie fully inside the episode, so the
        # first valid index is i=12 (window [10,11,12]), giving i=12..19
        # (8 windows). The pre-v4 condition compared one row before the window
        # and wrongly dropped i=12 (roadmap finding H5).
        assert len(ds) == 8 + 8

    def test_window_exactly_episode_length(self, tmp_path):
        # 5-step episodes, window_k=5: each episode contributes exactly one
        # window — its full length. Episode 1: i=4 (window [0..4]);
        # episode 2: i=9 (window [5..9]). The pre-v4 condition yielded zero
        # windows for every non-first episode of length exactly K (H5).
        episode_ids = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.int64)
        ds = self._make_dummy_dataset(tmp_path, 10, episode_ids, window_k=5)
        assert len(ds) == 2

    def test_windows_never_cross_boundaries_exhaustive(self, tmp_path):
        # Property check over an irregular episode layout: every returned
        # window is fully same-episode, and every fully-inside index is kept.
        episode_ids = np.array([1] * 7 + [2] * 4 + [3] * 5 + [4] * 3, dtype=np.int64)
        k = 4
        ds = self._make_dummy_dataset(tmp_path, len(episode_ids), episode_ids, window_k=k)
        valid = set(ds.valid_indices.tolist())
        for i in range(len(episode_ids)):
            window_ok = i >= k - 1 and len(set(episode_ids[i - k + 1 : i + 1])) == 1
            assert (i in valid) == window_ok, f"index {i}: expected valid={window_ok}"

    def test_window_larger_than_episode(self, tmp_path):
        # Episodes shorter than window → no valid windows
        episode_ids = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)
        ds = self._make_dummy_dataset(tmp_path, 6, episode_ids, window_k=5)
        assert len(ds) == 0

    def test_getitem_returns_correct_keys_no_physics(self, tmp_path):
        episode_ids = np.ones(20, dtype=np.int64)
        ds = self._make_dummy_dataset(tmp_path, 20, episode_ids, window_k=3)
        item = ds[0]
        assert "obs_window" in item
        assert "action_window" in item
        assert "obs" in item
        assert "action" in item
        assert "next_obs" in item
        assert "physics" not in item

    def test_getitem_returns_physics_when_present(self, tmp_path):
        episode_ids = np.ones(20, dtype=np.int64)
        ds = self._make_dummy_dataset(tmp_path, 20, episode_ids, window_k=3, include_physics=True)
        item = ds[0]
        assert "physics" in item
        assert item["physics"].shape == (3,)

    def test_getitem_obs_window_shape(self, tmp_path):
        episode_ids = np.ones(20, dtype=np.int64)
        k = 5
        ds = self._make_dummy_dataset(tmp_path, 20, episode_ids, window_k=k)
        item = ds[0]
        assert item["obs_window"].shape == (k, 6)
        assert item["action_window"].shape == (k, 8)
        assert item["obs"].shape == (6,)
        assert item["action"].shape == (8,)
        assert item["next_obs"].shape == (6,)

    def test_all_windows_same_episode(self, tmp_path):
        n = 30
        episode_ids = np.array([1] * 10 + [2] * 10 + [3] * 10, dtype=np.int64)
        k = 3
        ds = self._make_dummy_dataset(tmp_path, n, episode_ids, window_k=k)
        # Each window must span a single episode
        for idx in range(len(ds)):
            i = int(ds.valid_indices[idx])
            window_ids = ds.episode_ids[i - k + 1 : i + 1]
            assert len(set(window_ids.tolist())) == 1, (
                f"Window at i={i} spans multiple episodes: {window_ids}"
            )

    def test_missing_episode_ids_raises(self, tmp_path):
        from atlas_wm.data.episode_dataset import EpisodeATLASDataset

        n = 10
        obs = np.random.randn(n, 6).astype(np.float32)
        np.save(tmp_path / "train_obs.npy", obs)
        np.save(tmp_path / "train_actions.npy", np.random.randn(n, 8).astype(np.float32))
        np.save(tmp_path / "train_next_obs.npy", obs)
        # No episode_ids file → should raise FileNotFoundError

        with pytest.raises(FileNotFoundError):
            EpisodeATLASDataset(str(tmp_path), split="train", window_k=3)

    def test_invalid_split_raises(self, tmp_path):
        from atlas_wm.data.episode_dataset import EpisodeATLASDataset

        with pytest.raises(ValueError):
            EpisodeATLASDataset(str(tmp_path), split="bogus", window_k=3)


class TestProbeFromArrays:
    def test_identifiable_scores_high(self):
        from atlas_wm.eval.latent_probe import probe_from_arrays

        rng = np.random.default_rng(42)
        feats = rng.normal(size=(500, 8)).astype(np.float64)
        w = rng.normal(size=(8, 3))
        targets = feats @ w + rng.normal(size=(500, 3)) * 0.01  # near-linear relationship

        result = probe_from_arrays(
            feats,
            targets,
            latent_key="z_test",
            target_names=["a", "b", "c"],
            alpha=1e-4,
        )
        assert result.r2_mean > 0.95, f"Expected high R², got {result.r2_mean:.3f}"

    def test_uninformative_scores_near_zero(self):
        from atlas_wm.eval.latent_probe import probe_from_arrays

        rng = np.random.default_rng(0)
        feats = rng.normal(size=(400, 8))
        targets = rng.normal(size=(400, 2))  # independent of feats

        result = probe_from_arrays(feats, targets, alpha=1.0)
        assert result.r2_mean < 0.2

    def test_output_structure(self):
        from atlas_wm.eval.latent_probe import ProbeResult, probe_from_arrays

        rng = np.random.default_rng(1)
        feats = rng.normal(size=(200, 4))
        targets = rng.normal(size=(200, 3))

        result = probe_from_arrays(
            feats,
            targets,
            latent_key="my_latent",
            target_names=["x", "y", "z"],
        )
        assert isinstance(result, ProbeResult)
        assert result.latent_key == "my_latent"
        assert result.target_names == ["x", "y", "z"]
        assert result.r2_per_target.shape == (3,)
        assert isinstance(result.r2_mean, float)

    def test_1d_target_handled(self):
        from atlas_wm.eval.latent_probe import probe_from_arrays

        rng = np.random.default_rng(2)
        feats = rng.normal(size=(200, 4))
        targets = rng.normal(size=200)  # 1D, not 2D

        result = probe_from_arrays(feats, targets)
        assert result.r2_per_target.shape == (1,)

    def test_result_str_contains_latent_key(self):
        from atlas_wm.eval.latent_probe import probe_from_arrays

        rng = np.random.default_rng(3)
        feats = rng.normal(size=(100, 4))
        targets = rng.normal(size=(100, 2))

        result = probe_from_arrays(feats, targets, latent_key="z_gru")
        assert "z_gru" in str(result)
        assert "R²" in str(result)
