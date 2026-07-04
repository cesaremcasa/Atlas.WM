"""Belief-conditioning tests (v4 B12)."""

import numpy as np
import torch

from atlas_wm.models.physics_belief import PhysicsBeliefEncoder


class TestForwardSequence:
    def test_causal_prefix_property(self):
        # z_t from the full pass must equal z from the truncated input —
        # i.e. per-step beliefs never see the future.
        torch.manual_seed(0)
        enc = PhysicsBeliefEncoder(obs_dim=27, d_slow=8, hidden_dim=16)
        x = torch.randn(2, 12, 27)
        full = enc.forward_sequence(x)
        for t in (1, 5, 11):
            torch.testing.assert_close(full[:, t], enc(x[:, : t + 1]))

    def test_last_step_matches_forward(self):
        torch.manual_seed(1)
        enc = PhysicsBeliefEncoder(obs_dim=27, d_slow=8, hidden_dim=16)
        x = torch.randn(3, 9, 27)
        torch.testing.assert_close(enc.forward_sequence(x)[:, -1], enc(x))


class TestBeliefWindow:
    def test_dataset_alignment(self, tmp_path):
        from atlas_wm.data.episode_dataset import EpisodeATLASDataset

        n, k = 30, 5
        rng = np.random.default_rng(0)
        np.save(tmp_path / "train_obs.npy", rng.uniform(0, 20, (n, 6)).astype(np.float32))
        np.save(tmp_path / "train_actions.npy", rng.standard_normal((n, 8)).astype(np.float32))
        np.save(tmp_path / "train_next_obs.npy", rng.uniform(0, 20, (n, 6)).astype(np.float32))
        np.save(tmp_path / "train_episode_ids.npy", np.repeat([0, 1, 2], 10))
        belief = np.arange(n, dtype=np.float32)[:, None].repeat(8, axis=1)
        np.save(tmp_path / "train_belief.npy", belief)
        ds = EpisodeATLASDataset(str(tmp_path), split="train", window_k=k)
        item = ds[0]
        i = int(ds.valid_indices[0])
        np.testing.assert_array_equal(item["belief_window"].numpy(), belief[i - k + 1 : i + 1])
