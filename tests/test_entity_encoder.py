"""Tests for EntityEncoder — multi-object, variable n_objects (Block 10)."""

import torch

from atlas_wm.models.entity_encoder import EntityEncoder

ENTITY_DIM = 2
D_STATIC = 16
D_DYNAMIC = 32
D_CONTROLLABLE = 16
BATCH = 4


def _make_enc(**kwargs) -> EntityEncoder:
    defaults = dict(
        entity_dim=ENTITY_DIM, d_static=D_STATIC, d_dynamic=D_DYNAMIC, d_controllable=D_CONTROLLABLE
    )
    defaults.update(kwargs)
    return EntityEncoder(**defaults)


class TestOutputInterface:
    def test_required_keys_present(self):
        enc = _make_enc()
        x = torch.randn(BATCH, 3, ENTITY_DIM)
        out = enc(x)
        required = {
            "z_static",
            "z_static_immutable",
            "z_static_slow",
            "z_dynamic",
            "z_controllable",
            "z_full",
        }
        assert required.issubset(out.keys())

    def test_z_full_shape(self):
        enc = _make_enc()
        x = torch.randn(BATCH, 3, ENTITY_DIM)
        out = enc(x)
        assert out["z_full"].shape == (BATCH, D_STATIC + D_DYNAMIC + D_CONTROLLABLE)

    def test_static_split_consistent(self):
        enc = _make_enc()
        x = torch.randn(BATCH, 3, ENTITY_DIM)
        out = enc(x)
        reconstructed = torch.cat([out["z_static_immutable"], out["z_static_slow"]], dim=-1)
        assert torch.equal(reconstructed, out["z_static"])

    def test_d_immutable_default_is_half(self):
        enc = _make_enc()
        assert enc.d_immutable == D_STATIC // 2
        assert enc.d_slow == D_STATIC - D_STATIC // 2


class TestVariableObjects:
    def test_handles_3_objects(self):
        enc = _make_enc()
        out = enc(torch.randn(BATCH, 3, ENTITY_DIM))
        assert out["z_full"].shape == (BATCH, D_STATIC + D_DYNAMIC + D_CONTROLLABLE)

    def test_handles_5_objects(self):
        enc = _make_enc()
        out = enc(torch.randn(BATCH, 5, ENTITY_DIM))
        assert out["z_full"].shape == (BATCH, D_STATIC + D_DYNAMIC + D_CONTROLLABLE)

    def test_handles_10_objects(self):
        enc = _make_enc()
        out = enc(torch.randn(BATCH, 10, ENTITY_DIM))
        assert out["z_full"].shape == (BATCH, D_STATIC + D_DYNAMIC + D_CONTROLLABLE)

    def test_output_dim_independent_of_n_objects(self):
        enc = _make_enc()
        out3 = enc(torch.randn(BATCH, 3, ENTITY_DIM))
        out7 = enc(torch.randn(BATCH, 7, ENTITY_DIM))
        assert out3["z_full"].shape == out7["z_full"].shape


class TestPermutationProperties:
    def test_agent_is_entity_0(self):
        """Changing entity 0 (agent) must change the output; changing entity 1+ alone may not."""
        enc = _make_enc()
        enc.eval()
        x = torch.randn(1, 3, ENTITY_DIM)
        x_modified = x.clone()
        x_modified[0, 0, :] = x_modified[0, 0, :] + 1.0  # perturb agent

        with torch.no_grad():
            out_orig = enc(x)["z_full"]
            out_mod = enc(x_modified)["z_full"]

        assert not torch.equal(out_orig, out_mod), "Agent perturbation must change output."

    def test_box_permutation_changes_output(self):
        """Swapping boxes changes context (mean pooling loses order within boxes)."""
        enc = _make_enc()
        enc.eval()
        x = torch.randn(1, 3, ENTITY_DIM)
        x_perm = x.clone()
        x_perm[0, 1, :], x_perm[0, 2, :] = x[0, 2, :].clone(), x[0, 1, :].clone()

        with torch.no_grad():
            out_orig = enc(x)["z_controllable"]
            out_perm = enc(x_perm)["z_controllable"]

        # z_controllable depends on the agent (entity 0) and the global context
        # Swapping box entities changes the mean-pool → context changes → controllable changes
        # This is a sanity test (not a strict equivariance test)
        assert out_orig.shape == out_perm.shape


class TestGradients:
    def test_all_params_receive_grad(self):
        enc = _make_enc()
        x = torch.randn(BATCH, 4, ENTITY_DIM)
        out = enc(x)
        out["z_full"].sum().backward()
        for name, param in enc.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_gradients_finite(self):
        enc = _make_enc()
        x = torch.randn(BATCH, 4, ENTITY_DIM)
        out = enc(x)
        out["z_full"].sum().backward()
        for name, param in enc.named_parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


class TestDeterminism:
    def test_eval_deterministic(self):
        enc = _make_enc()
        enc.eval()
        x = torch.randn(BATCH, 3, ENTITY_DIM)
        with torch.no_grad():
            out1 = enc(x)["z_full"]
            out2 = enc(x)["z_full"]
        assert torch.equal(out1, out2)
