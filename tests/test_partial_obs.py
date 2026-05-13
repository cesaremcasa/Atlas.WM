"""Tests for partial observability transform (Block 11)."""

import pytest
import torch

from atlas_wm.data.partial_obs import PartialObsWrapper, nearest_k_obs
from atlas_wm.models.entity_encoder import EntityEncoder

BATCH = 4
N_OBJ = 5
ENTITY_DIM = 2


def _make_x(batch: int = BATCH, n_obj: int = N_OBJ, dim: int = ENTITY_DIM) -> torch.Tensor:
    return torch.randn(batch, n_obj, dim)


class TestNearestKObs:
    def test_output_shape(self):
        x = _make_x()
        out = nearest_k_obs(x, k=3)
        assert out.shape == (BATCH, 3, ENTITY_DIM)

    def test_k_equals_n_obj_returns_full(self):
        x = _make_x()
        out = nearest_k_obs(x, k=N_OBJ)
        assert torch.equal(out, x)

    def test_k_greater_than_n_obj_returns_full(self):
        x = _make_x()
        out = nearest_k_obs(x, k=N_OBJ + 3)
        assert torch.equal(out, x)

    def test_agent_is_always_at_index_0(self):
        x = _make_x()
        out = nearest_k_obs(x, k=3)
        assert torch.equal(out[:, 0, :], x[:, 0, :])

    def test_k1_returns_only_agent(self):
        x = _make_x()
        out = nearest_k_obs(x, k=1)
        assert out.shape == (BATCH, 1, ENTITY_DIM)
        assert torch.equal(out[:, 0, :], x[:, 0, :])

    def test_invalid_k_raises(self):
        x = _make_x()
        with pytest.raises(ValueError, match="k must be"):
            nearest_k_obs(x, k=0)

    def test_nearest_objects_are_closer_than_excluded(self):
        """k=2 keeps the nearest non-agent; the excluded one is farther."""
        x = torch.zeros(1, 4, 2)
        # Agent at origin, objects at distances 1, 3, 10
        x[0, 0] = torch.tensor([0.0, 0.0])
        x[0, 1] = torch.tensor([1.0, 0.0])  # nearest
        x[0, 2] = torch.tensor([3.0, 0.0])
        x[0, 3] = torch.tensor([10.0, 0.0])  # farthest
        out = nearest_k_obs(x, k=2)
        assert out.shape == (1, 2, 2)
        assert torch.equal(out[0, 1], x[0, 1]), "Nearest object should be kept."

    def test_selects_k_minus_1_nearest(self):
        """k=3: agent + 2 nearest non-agent objects."""
        x = torch.zeros(1, 5, 2)
        x[0, 0] = torch.tensor([0.0, 0.0])  # agent
        x[0, 1] = torch.tensor([2.0, 0.0])
        x[0, 2] = torch.tensor([1.0, 0.0])  # nearest
        x[0, 3] = torch.tensor([5.0, 0.0])
        x[0, 4] = torch.tensor([3.0, 0.0])
        out = nearest_k_obs(x, k=3)
        # Agent (dist=0), obj2 (dist=1), obj1 (dist=2) should be selected
        kept_positions = {tuple(out[0, i].tolist()) for i in range(1, 3)}
        assert (1.0, 0.0) in kept_positions, "Nearest object (1,0) must be in k=3 output"
        assert (2.0, 0.0) in kept_positions, "Second-nearest (2,0) must be in k=3 output"


class TestPartialObsWrapper:
    def test_wrapper_output_shape(self):
        wrap = PartialObsWrapper(k=3)
        x = _make_x()
        out = wrap(x)
        assert out.shape == (BATCH, 3, ENTITY_DIM)

    def test_wrapper_preserves_agent(self):
        wrap = PartialObsWrapper(k=2)
        x = _make_x()
        out = wrap(x)
        assert torch.equal(out[:, 0, :], x[:, 0, :])

    def test_wrapper_with_entity_encoder(self):
        """Partial obs → EntityEncoder must produce correct output shapes."""
        wrap = PartialObsWrapper(k=3)
        enc = EntityEncoder(entity_dim=ENTITY_DIM, d_static=16, d_dynamic=32, d_controllable=16)
        enc.eval()
        x = _make_x(batch=BATCH, n_obj=8)
        x_partial = wrap(x)
        with torch.no_grad():
            out = enc(x_partial)
        assert out["z_full"].shape == (BATCH, 16 + 32 + 16)
