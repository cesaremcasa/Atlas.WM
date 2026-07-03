"""Tests for the stable self-predictive objectives (v4 B7, finding H1)."""

import numpy as np
import torch
import torch.nn as nn
import yaml

from atlas_wm.training.objectives import ema_update, make_ema_target, vicreg_regularizer


class TestEMATarget:
    def test_target_starts_identical_and_frozen(self):
        online = nn.Linear(4, 4)
        target = make_ema_target(online)
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            assert torch.equal(p_o, p_t)
            assert not p_t.requires_grad

    def test_update_moves_target_toward_online(self):
        online = nn.Linear(4, 4)
        target = make_ema_target(online)
        with torch.no_grad():
            for p in online.parameters():
                p.add_(1.0)
        before = [p.clone() for p in target.parameters()]
        ema_update(online, target, tau=0.9)
        for p_t, p_b, p_o in zip(target.parameters(), before, online.parameters()):
            expected = 0.9 * p_b + 0.1 * p_o
            torch.testing.assert_close(p_t, expected)

    def test_tau_one_freezes_target(self):
        online = nn.Linear(4, 4)
        target = make_ema_target(online)
        with torch.no_grad():
            for p in online.parameters():
                p.mul_(3.0)
        before = [p.clone() for p in target.parameters()]
        ema_update(online, target, tau=1.0)
        for p_t, p_b in zip(target.parameters(), before):
            torch.testing.assert_close(p_t, p_b)


class TestVICRegRegularizer:
    def test_collapsed_batch_is_penalized(self):
        z = torch.zeros(32, 8)  # fully collapsed
        var_loss, cov_loss = vicreg_regularizer(z)
        assert var_loss.item() > 0.999  # std ≈ 0 → (almost) full hinge (sqrt eps)
        assert cov_loss.item() == 0.0

    def test_healthy_batch_is_not_penalized(self):
        g = torch.Generator().manual_seed(0)
        z = torch.randn(4096, 8, generator=g) * 1.5  # dispersed, independent dims
        var_loss, cov_loss = vicreg_regularizer(z)
        assert var_loss.item() == 0.0
        assert cov_loss.item() < 0.05

    def test_redundant_dims_are_penalized(self):
        g = torch.Generator().manual_seed(0)
        base = torch.randn(512, 1, generator=g)
        z = base.repeat(1, 8)  # perfectly correlated dims
        _, cov_loss = vicreg_regularizer(z)
        assert cov_loss.item() > 0.5

    def test_scale_runaway_is_penalized(self):
        # The covariance term grows ~scale^4 — it replaces the L2 crutch's
        # anti-runaway role for any latent with residual correlation.
        g = torch.Generator().manual_seed(0)
        base = torch.randn(512, 8, generator=g)
        mixed = base @ (torch.eye(8) + 0.2)  # mild correlation
        _, cov_small = vicreg_regularizer(mixed)
        _, cov_big = vicreg_regularizer(mixed * 10)
        assert cov_big.item() > 100 * cov_small.item()

    def test_degenerate_batch_returns_zero(self):
        z = torch.randn(1, 8)
        var_loss, cov_loss = vicreg_regularizer(z)
        assert var_loss.item() == 0.0 and cov_loss.item() == 0.0


class TestTrainWithObjectives:
    def _config(self, tmp_path, objective):
        import numpy as np

        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
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
                "objective": objective,
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
        path = tmp_path / f"config_{objective}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    def test_all_objectives_train_and_record_metadata(self, tmp_path):
        import argparse

        from train import train

        from atlas_wm.checkpointing.io import load_checkpoint

        for objective in ("ema", "vicreg", "legacy"):
            config = self._config(tmp_path, objective)
            ckpt = str(tmp_path / f"{objective}.safetensors")
            result = train(
                argparse.Namespace(
                    config=str(config),
                    max_steps=None,
                    no_checkpoint=False,
                    output_checkpoint=ckpt,
                    seed=None,
                )
            )
            assert np.isfinite(result["best_val_loss"]), f"{objective} diverged"
            assert len(result["history"]["val_next_mse"]) == 1
            _, meta = load_checkpoint(
                ckpt,
                expected_model_class="ContinuousEncoder+StructuredDynamics",
                strict_env=False,
                allow_unsigned=True,
            )
            assert meta["objective"] == objective
