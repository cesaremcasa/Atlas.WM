"""Training reproducibility canary (v4 B3, AD-7).

The pre-v4 determinism canary only checked eval-mode forward passes of
freshly constructed models; training itself was unseeded, so no checkpoint
was reproducible. These tests run the REAL training loop twice end-to-end
and require bit-identical loss traces.

Also covers finding H6: ``d_static_immutable``/``d_static_slow`` from the
config must actually reach the models and the exported-checkpoint dims.
"""

import argparse
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, "scripts")
from export_onnx import infer_dims  # noqa: E402
from train import train  # noqa: E402

from atlas_wm.checkpointing.io import load_checkpoint  # noqa: E402


def _make_config(tmp_path, d_immutable=8, d_slow=8, seed=42):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for split, n in (("train", 64), ("val", 16)):
        obs = rng.uniform(0, 20, size=(n, 6)).astype(np.float32)
        actions = np.eye(8, dtype=np.float32)[rng.integers(0, 8, size=n)]
        np.save(data_dir / f"{split}_obs.npy", obs)
        np.save(data_dir / f"{split}_actions.npy", actions)
        np.save(data_dir / f"{split}_next_obs.npy", obs + 0.05)

    cfg = {
        "model": {
            "d_static_immutable": d_immutable,
            "d_static_slow": d_slow,
            "d_dynamic": 32,
            "d_controllable": 16,
            "input_dim": 6,
        },
        "environment": {"action_space_size": 8},
        "data": {"data_dir": str(data_dir)},
        "training": {
            "seed": seed,
            "batch_size": 16,
            "learning_rate": 3.0e-4,
            "num_epochs": 2,
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
    return config_path


def _run(config_path, checkpoint=None, seed=None):
    args = argparse.Namespace(
        config=str(config_path),
        max_steps=None,
        no_checkpoint=checkpoint is None,
        output_checkpoint=checkpoint,
        seed=seed,
    )
    return train(args)


class TestTrainingCanary:
    def test_same_seed_identical_loss_trace(self, tmp_path):
        config = _make_config(tmp_path)
        first = _run(config)
        second = _run(config)
        assert first["history"] == second["history"], (
            "Two runs with the same seed produced different loss traces — "
            "training is not reproducible (AD-7)"
        )
        assert first["best_val_loss"] == second["best_val_loss"]

    def test_different_seeds_different_traces(self, tmp_path):
        config = _make_config(tmp_path)
        a = _run(config, seed=1)
        b = _run(config, seed=2)
        assert a["history"]["train"] != b["history"]["train"], (
            "Different seeds produced identical traces — seeding is not being applied"
        )


class TestDImmutablePlumbing:
    @pytest.mark.parametrize("d_immutable,d_slow", [(4, 12), (12, 4)])
    def test_config_split_reaches_checkpoint(self, tmp_path, d_immutable, d_slow):
        config = _make_config(tmp_path, d_immutable=d_immutable, d_slow=d_slow)
        ckpt = str(tmp_path / "model.safetensors")
        _run(config, checkpoint=ckpt)

        state_dict, meta = load_checkpoint(
            ckpt,
            expected_model_class="ContinuousEncoder+StructuredDynamics",
            strict_env=False,
            allow_unsigned=True,
        )
        # static_slow_net input width == d_slow proves the non-default split
        # reached StructuredDynamics (pre-v4 it silently trained 8/8, H6).
        assert state_dict["dynamics.static_slow_net.0.weight"].shape[1] == d_slow
        assert meta["d_immutable"] == str(d_immutable)

        dims = infer_dims(state_dict)
        assert dims["d_immutable"] == d_immutable
        assert dims["d_static"] == d_immutable + d_slow

    def test_critic_state_is_checkpointed(self, tmp_path):
        config = _make_config(tmp_path)
        ckpt = str(tmp_path / "model.safetensors")
        _run(config, checkpoint=ckpt)
        state_dict, _ = load_checkpoint(
            ckpt,
            expected_model_class="ContinuousEncoder+StructuredDynamics",
            strict_env=False,
            allow_unsigned=True,
        )
        assert any(k.startswith("critic.") for k in state_dict), (
            "Critic weights missing from checkpoint — adversarial training cannot resume (M8)"
        )
