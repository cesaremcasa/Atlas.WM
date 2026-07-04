"""Multi-step rollout training and evaluation tests (v4 B8)."""

import argparse

import numpy as np
import torch
import yaml
from evaluate import load_world_model, rollout_mse_by_horizon
from train import train

from atlas_wm.checkpointing.io import load_checkpoint
from atlas_wm.data.dataset import stack_window
from atlas_wm.data.episode_dataset import EpisodeATLASDataset


class TestStackWindow:
    def test_single_frame_drops_leading_position(self):
        obs_w = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
        out = stack_window(obs_w, frame_stack=1)
        assert out.shape == (2, 4, 3)
        torch.testing.assert_close(out, obs_w[:, 1:])

    def test_two_frames_concat_previous(self):
        obs_w = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
        out = stack_window(obs_w, frame_stack=2)
        assert out.shape == (2, 4, 6)
        # position s+1 input = [frame_s | frame_{s+1}]
        torch.testing.assert_close(out[:, 0, :3], obs_w[:, 0])
        torch.testing.assert_close(out[:, 0, 3:], obs_w[:, 1])
        torch.testing.assert_close(out[:, 3, :3], obs_w[:, 3])
        torch.testing.assert_close(out[:, 3, 3:], obs_w[:, 4])


def _make_setup(tmp_path, rollout_k=3, n=120, ep_len=20):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for split, rows in (("train", n), ("val", 40)):
        obs = rng.uniform(0, 20, size=(rows, 6)).astype(np.float32)
        np.save(data_dir / f"{split}_obs.npy", obs)
        np.save(
            data_dir / f"{split}_actions.npy",
            np.eye(8, dtype=np.float32)[rng.integers(0, 8, size=rows)],
        )
        np.save(data_dir / f"{split}_next_obs.npy", obs + 0.05)
        np.save(
            data_dir / f"{split}_episode_ids.npy",
            np.repeat(np.arange(rows // ep_len), ep_len),
        )
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
            "objective": "vicreg",
            "rollout_k": rollout_k,
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
    return config_path, str(data_dir)


class TestRolloutTraining:
    def test_trains_and_records_rollout_metadata(self, tmp_path):
        config, _ = _make_setup(tmp_path, rollout_k=3)
        ckpt = str(tmp_path / "model.safetensors")
        result = train(
            argparse.Namespace(
                config=str(config),
                max_steps=None,
                no_checkpoint=False,
                output_checkpoint=ckpt,
                seed=None,
            )
        )
        assert np.isfinite(result["best_val_loss"])
        assert len(result["history"]["val_rollout_mse"]) == 1
        _, meta = load_checkpoint(
            ckpt,
            expected_model_class="ContinuousEncoder+StructuredDynamics",
            strict_env=False,
            allow_unsigned=True,
        )
        assert meta["rollout_k"] == "3"

    def test_multi_horizon_eval_and_immutable_passthrough(self, tmp_path):
        config, data_dir = _make_setup(tmp_path, rollout_k=2)
        ckpt = str(tmp_path / "model.safetensors")
        train(
            argparse.Namespace(
                config=str(config),
                max_steps=None,
                no_checkpoint=False,
                output_checkpoint=ckpt,
                seed=None,
            )
        )
        encoder, dynamics, decoder, meta = load_world_model(ckpt)
        horizon = 5
        ds = EpisodeATLASDataset(data_dir, split="val", window_k=horizon + 2)
        mse, imm_drift = rollout_mse_by_horizon(
            encoder,
            dynamics,
            decoder,
            ds,
            frame_stack=int(meta["frame_stack"]),
            horizon=horizon,
            base_dim=6,
        )
        assert len(mse) == horizon
        assert all(np.isfinite(m) for m in mse)
        # AD-2 hard passthrough: bit-exact across the whole open-loop rollout.
        assert imm_drift == 0.0
