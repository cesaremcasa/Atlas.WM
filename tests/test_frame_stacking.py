"""Frame-stacking tests (v4 B6, roadmap finding M2).

A single 6-D position frame makes one-step latent prediction ill-posed:
velocities are unobservable. frame_stack=2 concatenates the previous
same-episode frame so velocity is recoverable by the encoder.
"""

import argparse

import numpy as np
import pytest
import yaml

from atlas_wm.checkpointing.io import load_checkpoint
from atlas_wm.data.dataset import ATLASDataset


def _write_split(tmp_path, n_eps=4, ep_len=8):
    rng = np.random.default_rng(0)
    n = n_eps * ep_len
    obs = rng.uniform(0, 20, size=(n, 6)).astype(np.float32)
    next_obs = rng.uniform(0, 20, size=(n, 6)).astype(np.float32)
    np.save(tmp_path / "train_obs.npy", obs)
    np.save(tmp_path / "train_actions.npy", rng.standard_normal((n, 8)).astype(np.float32))
    np.save(tmp_path / "train_next_obs.npy", next_obs)
    ids = np.repeat(np.arange(n_eps), ep_len)
    np.save(tmp_path / "train_episode_ids.npy", ids)
    return obs / 20.0, next_obs / 20.0, ids


class TestStackedDataset:
    def test_shapes_and_row_count_unchanged(self, tmp_path):
        obs, _, _ = _write_split(tmp_path)
        ds = ATLASDataset(str(tmp_path), split="train", frame_stack=2)
        assert len(ds) == len(obs)
        assert ds.observations.shape == (len(obs), 12)
        assert ds.next_observations.shape == (len(obs), 12)

    def test_stacking_semantics(self, tmp_path):
        obs, next_obs, ids = _write_split(tmp_path)
        ds = ATLASDataset(str(tmp_path), split="train", frame_stack=2)
        for i in range(len(obs)):
            prev_half, cur_half = ds.observations[i, :6], ds.observations[i, 6:]
            np.testing.assert_allclose(cur_half, obs[i], rtol=1e-6)
            if i == 0 or ids[i] != ids[i - 1]:
                # First transition of an episode repeats the frame.
                np.testing.assert_allclose(prev_half, obs[i], rtol=1e-6)
            else:
                np.testing.assert_allclose(prev_half, obs[i - 1], rtol=1e-6)
            # Stacked next input = [current obs | next_obs].
            np.testing.assert_allclose(ds.next_observations[i, :6], obs[i], rtol=1e-6)
            np.testing.assert_allclose(ds.next_observations[i, 6:], next_obs[i], rtol=1e-6)

    def test_frames_never_cross_episode_boundaries(self, tmp_path):
        obs, _, ids = _write_split(tmp_path, n_eps=6, ep_len=5)
        ds = ATLASDataset(str(tmp_path), split="train", frame_stack=2)
        boundary_rows = np.where(np.diff(ids) != 0)[0] + 1
        for i in boundary_rows:
            np.testing.assert_allclose(
                ds.observations[i, :6],
                obs[i],
                rtol=1e-6,
                err_msg=f"row {i} pulled a frame from the previous episode",
            )

    def test_requires_episode_ids(self, tmp_path):
        _write_split(tmp_path)
        (tmp_path / "train_episode_ids.npy").unlink()
        with pytest.raises(FileNotFoundError, match="episode IDs"):
            ATLASDataset(str(tmp_path), split="train", frame_stack=2)

    def test_invalid_frame_stack_rejected(self, tmp_path):
        _write_split(tmp_path)
        with pytest.raises(ValueError, match="frame_stack"):
            ATLASDataset(str(tmp_path), split="train", frame_stack=3)


class TestStackedTraining:
    def test_train_records_frame_stack_and_widens_encoder(self, tmp_path):
        from train import train

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        rng = np.random.default_rng(0)
        for split, n in (("train", 64), ("val", 16)):
            obs = rng.uniform(0, 20, size=(n, 6)).astype(np.float32)
            np.save(data_dir / f"{split}_obs.npy", obs)
            np.save(
                data_dir / f"{split}_actions.npy",
                np.eye(8, dtype=np.float32)[rng.integers(0, 8, size=n)],
            )
            np.save(data_dir / f"{split}_next_obs.npy", obs + 0.05)
            np.save(data_dir / f"{split}_episode_ids.npy", np.repeat(np.arange(4), n // 4))

        cfg = {
            "model": {
                "d_static_immutable": 8,
                "d_static_slow": 8,
                "d_dynamic": 32,
                "d_controllable": 16,
                "input_dim": 6,
                "frame_stack": 2,
            },
            "environment": {"action_space_size": 8},
            "data": {"data_dir": str(data_dir)},
            "training": {
                "seed": 42,
                "batch_size": 16,
                "learning_rate": 3.0e-4,
                "num_epochs": 1,
                "early_stopping_patience": 15,
                "grad_clip_norm": 0.5,
                "lr_scheduler_patience": 3,
                "lr_scheduler_factor": 0.5,
                "lambda_recon": 1.0,
                "lambda_slow_drift": 0.1,
                "lambda_action_invariance": 0.001,
                "adv_warmup_epochs": 0,
            },
            "checkpointing": {"checkpoint_dir": str(tmp_path / "ckpt")},
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg))
        ckpt = str(tmp_path / "model.safetensors")
        train(
            argparse.Namespace(
                config=str(config_path),
                max_steps=None,
                no_checkpoint=False,
                output_checkpoint=ckpt,
                seed=None,
            )
        )

        state_dict, meta = load_checkpoint(
            ckpt,
            expected_model_class="ContinuousEncoder+StructuredDynamics",
            strict_env=False,
            allow_unsigned=True,
        )
        assert meta["frame_stack"] == "2"
        assert state_dict["encoder.shared.0.weight"].shape[1] == 12
        # Decoder reconstructs the stacked input.
        assert state_dict["decoder.net.0.weight"] is not None
