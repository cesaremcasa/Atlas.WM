"""Tests for AD-3 identifiability: ActionInvarianceCritic and env hash."""

import torch
import torch.optim as optim

from atlas_wm.checkpointing.env_hash import compute_env_hash
from atlas_wm.models.identifiability import (
    ActionInvarianceCritic,
    critic_loss,
    encoder_adversarial_loss,
)

D_IMM = 8
ACTION_DIM = 8
BATCH = 16


class TestActionInvarianceCritic:
    def test_output_shape(self):
        c = ActionInvarianceCritic(d_immutable=D_IMM, action_dim=ACTION_DIM)
        z = torch.randn(BATCH, D_IMM)
        out = c(z)
        assert out.shape == (BATCH, ACTION_DIM)

    def test_critic_loss_is_scalar(self):
        c = ActionInvarianceCritic(d_immutable=D_IMM, action_dim=ACTION_DIM)
        z = torch.randn(BATCH, D_IMM)
        action = torch.randn(BATCH, ACTION_DIM)
        loss = critic_loss(c, z, action)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_adversarial_loss_is_negative_critic(self):
        c = ActionInvarianceCritic(d_immutable=D_IMM, action_dim=ACTION_DIM)
        z = torch.randn(BATCH, D_IMM)
        action = torch.randn(BATCH, ACTION_DIM)
        c_loss = critic_loss(c, z, action)
        adv_loss = encoder_adversarial_loss(c, z, action)
        assert torch.allclose(c_loss, -adv_loss)

    def test_critic_trains_to_predict_action(self):
        """Critic loss decreases after several gradient steps."""
        c = ActionInvarianceCritic(d_immutable=D_IMM, action_dim=ACTION_DIM)
        opt = optim.Adam(c.parameters(), lr=1e-2)
        z = torch.randn(BATCH, D_IMM)
        action = torch.randn(BATCH, ACTION_DIM)

        initial_loss = critic_loss(c, z.detach(), action).item()
        for _ in range(20):
            loss = critic_loss(c, z.detach(), action)
            opt.zero_grad()
            loss.backward()
            opt.step()
        final_loss = critic_loss(c, z.detach(), action).item()

        assert final_loss < initial_loss, "Critic loss did not decrease during training."

    def test_detached_z_no_grad_to_z(self):
        """critic_loss with detached z produces no gradient to z."""
        c = ActionInvarianceCritic(d_immutable=D_IMM, action_dim=ACTION_DIM)
        z = torch.randn(BATCH, D_IMM, requires_grad=True)
        action = torch.randn(BATCH, ACTION_DIM)
        loss = critic_loss(c, z.detach(), action)
        loss.backward()
        assert z.grad is None

    def test_adversarial_loss_grads_flow_to_z(self):
        """encoder_adversarial_loss produces gradient to z (no detach)."""
        c = ActionInvarianceCritic(d_immutable=D_IMM, action_dim=ACTION_DIM)
        z = torch.randn(BATCH, D_IMM, requires_grad=True)
        action = torch.randn(BATCH, ACTION_DIM)
        loss = encoder_adversarial_loss(c, z, action)
        loss.backward()
        assert z.grad is not None


class TestEnvHash:
    def test_returns_16_hex_chars(self):
        h = compute_env_hash()
        assert len(h) == 16 or h == "unknown-no-lockfile"

    def test_deterministic_for_same_file(self, tmp_path):
        lock = tmp_path / "requirements.lock"
        lock.write_text("torch==2.1.2\nnumpy==1.26.4\n")
        h1 = compute_env_hash(str(lock))
        h2 = compute_env_hash(str(lock))
        assert h1 == h2

    def test_different_files_different_hashes(self, tmp_path):
        lock_a = tmp_path / "a.lock"
        lock_b = tmp_path / "b.lock"
        lock_a.write_text("torch==2.1.2\n")
        lock_b.write_text("torch==2.2.0\n")
        assert compute_env_hash(str(lock_a)) != compute_env_hash(str(lock_b))

    def test_missing_file_fallback(self, tmp_path):
        try:
            h = compute_env_hash(str(tmp_path / "nonexistent.lock"))
            assert h == "unknown-no-lockfile"
        except FileNotFoundError:
            pass  # also acceptable behavior
