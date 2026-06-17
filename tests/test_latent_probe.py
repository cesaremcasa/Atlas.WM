"""Latent probe tests (Block 12).

Validates the ridge-regression probing utility used to test AD-2: that variable
physics is identifiable from z_static_slow. Tests cover the math primitives and
the end-to-end probe against a controllable fake encoder.
"""

import numpy as np
import torch
import torch.nn as nn

from atlas_wm.eval.latent_probe import (
    ProbeResult,
    fit_ridge,
    predict_ridge,
    probe_latent,
    r2_score,
)

RNG = np.random.default_rng(0)


class TestR2Score:
    def test_perfect_prediction_is_one(self):
        y = RNG.normal(size=(100, 3))
        r2 = r2_score(y, y.copy())
        np.testing.assert_allclose(r2, np.ones(3), atol=1e-9)

    def test_mean_prediction_is_zero(self):
        y = RNG.normal(size=(100, 2))
        y_pred = np.broadcast_to(y.mean(axis=0), y.shape)
        r2 = r2_score(y, y_pred)
        np.testing.assert_allclose(r2, np.zeros(2), atol=1e-9)

    def test_constant_target_returns_zero(self):
        y = np.full((50, 1), 3.0)
        r2 = r2_score(y, y + RNG.normal(size=y.shape))
        np.testing.assert_allclose(r2, np.zeros(1), atol=1e-9)


class TestRidge:
    def test_recovers_linear_relationship(self):
        x = RNG.normal(size=(500, 4))
        true_w = RNG.normal(size=(4, 2))
        y = x @ true_w + 0.5
        w = fit_ridge(x, y, alpha=1e-6)
        y_pred = predict_ridge(x, w)
        r2 = r2_score(y, y_pred)
        assert (r2 > 0.99).all(), f"Ridge failed to fit linear data: R²={r2}"

    def test_bias_is_last_row(self):
        x = RNG.normal(size=(200, 3))
        y = np.full((200, 1), 7.0)  # pure offset, zero slope
        w = fit_ridge(x, y, alpha=1.0)
        assert w.shape == (4, 1)
        assert abs(w[-1, 0] - 7.0) < 0.1


class _LinearSlowEncoder(nn.Module):
    """Fake encoder whose z_static_slow is a fixed linear map of the input.

    z_static_immutable is constant (zeros) — it carries no information, mirroring
    the AD-2 passthrough that must not encode variable physics.
    """

    def __init__(self, input_dim: int, d_slow: int):
        super().__init__()
        self.register_buffer("w", torch.randn(input_dim, d_slow))
        self.d_slow = d_slow

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "z_static_slow": x @ self.w,
            "z_static_immutable": torch.zeros(x.shape[0], 4),
        }


class TestProbeLatent:
    def test_identifiable_latent_scores_high(self):
        torch.manual_seed(0)
        encoder = _LinearSlowEncoder(input_dim=6, d_slow=8)
        obs = RNG.normal(size=(800, 6)).astype(np.float32)
        # Targets are a linear function of z_static_slow → decodable.
        with torch.no_grad():
            slow = encoder(torch.tensor(obs))["z_static_slow"].numpy()
        targets = slow @ RNG.normal(size=(8, 3))

        result = probe_latent(
            encoder,
            obs,
            targets,
            latent_key="z_static_slow",
            target_names=["gravity", "friction_agent", "friction_box"],
            alpha=1e-4,
        )
        assert isinstance(result, ProbeResult)
        assert result.r2_per_target.shape == (3,)
        assert result.r2_mean > 0.95, f"Identifiable latent scored low: {result.r2_mean}"

    def test_uninformative_latent_scores_near_zero(self):
        torch.manual_seed(1)
        encoder = _LinearSlowEncoder(input_dim=6, d_slow=8)
        obs = RNG.normal(size=(800, 6)).astype(np.float32)
        with torch.no_grad():
            slow = encoder(torch.tensor(obs))["z_static_slow"].numpy()
        targets = slow @ RNG.normal(size=(8, 3))

        # z_static_immutable is constant zeros — cannot predict the targets.
        result = probe_latent(
            encoder,
            obs,
            targets,
            latent_key="z_static_immutable",
            target_names=["gravity", "friction_agent", "friction_box"],
            alpha=1.0,
        )
        assert result.r2_mean < 0.1, f"Constant latent should not predict: {result.r2_mean}"

    def test_result_str_is_readable(self):
        encoder = _LinearSlowEncoder(input_dim=6, d_slow=8)
        obs = RNG.normal(size=(100, 6)).astype(np.float32)
        targets = RNG.normal(size=(100, 3))
        result = probe_latent(encoder, obs, targets, latent_key="z_static_slow")
        assert "z_static_slow" in str(result)
        assert "R²" in str(result)
