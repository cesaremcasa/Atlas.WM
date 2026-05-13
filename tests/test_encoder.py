"""Unit tests for ContinuousEncoder (AD-2 hybrid static output)."""

import torch

from atlas_wm.models.continuous_encoder import ContinuousEncoder

INPUT_DIM = 6
D_STATIC = 16
D_DYNAMIC = 32
D_CONTROLLABLE = 16
BATCH = 8


def _make_encoder(**kwargs) -> ContinuousEncoder:
    defaults = dict(
        input_dim=INPUT_DIM, d_static=D_STATIC, d_dynamic=D_DYNAMIC, d_controllable=D_CONTROLLABLE
    )
    defaults.update(kwargs)
    return ContinuousEncoder(**defaults)


class TestOutputKeys:
    def test_all_required_keys_present(self):
        enc = _make_encoder()
        out = enc(torch.randn(BATCH, INPUT_DIM))
        required = {
            "z_static",
            "z_static_immutable",
            "z_static_slow",
            "z_dynamic",
            "z_controllable",
            "z_full",
        }
        assert required.issubset(out.keys()), f"Missing keys: {required - out.keys()}"

    def test_no_unexpected_nones(self):
        enc = _make_encoder()
        out = enc(torch.randn(BATCH, INPUT_DIM))
        for k, v in out.items():
            assert v is not None, f"Key {k!r} is None"


class TestOutputShapes:
    def test_z_static_shape(self):
        enc = _make_encoder()
        out = enc(torch.randn(BATCH, INPUT_DIM))
        assert out["z_static"].shape == (BATCH, D_STATIC)

    def test_z_static_immutable_shape(self):
        enc = _make_encoder()
        out = enc(torch.randn(BATCH, INPUT_DIM))
        assert out["z_static_immutable"].shape == (BATCH, enc.d_immutable)

    def test_z_static_slow_shape(self):
        enc = _make_encoder()
        out = enc(torch.randn(BATCH, INPUT_DIM))
        assert out["z_static_slow"].shape == (BATCH, enc.d_slow)

    def test_z_dynamic_shape(self):
        enc = _make_encoder()
        out = enc(torch.randn(BATCH, INPUT_DIM))
        assert out["z_dynamic"].shape == (BATCH, D_DYNAMIC)

    def test_z_controllable_shape(self):
        enc = _make_encoder()
        out = enc(torch.randn(BATCH, INPUT_DIM))
        assert out["z_controllable"].shape == (BATCH, D_CONTROLLABLE)

    def test_z_full_shape(self):
        enc = _make_encoder()
        out = enc(torch.randn(BATCH, INPUT_DIM))
        assert out["z_full"].shape == (BATCH, D_STATIC + D_DYNAMIC + D_CONTROLLABLE)

    def test_immutable_plus_slow_equals_static(self):
        enc = _make_encoder()
        assert enc.d_immutable + enc.d_slow == D_STATIC


class TestStaticSplit:
    def test_immutable_is_first_slice_of_static(self):
        enc = _make_encoder()
        x = torch.randn(BATCH, INPUT_DIM)
        out = enc(x)
        expected_imm = out["z_static"][:, : enc.d_immutable]
        assert torch.equal(out["z_static_immutable"], expected_imm)

    def test_slow_is_second_slice_of_static(self):
        enc = _make_encoder()
        x = torch.randn(BATCH, INPUT_DIM)
        out = enc(x)
        expected_slow = out["z_static"][:, enc.d_immutable :]
        assert torch.equal(out["z_static_slow"], expected_slow)

    def test_concat_immutable_slow_equals_static(self):
        enc = _make_encoder()
        x = torch.randn(BATCH, INPUT_DIM)
        out = enc(x)
        reconstructed = torch.cat([out["z_static_immutable"], out["z_static_slow"]], dim=-1)
        assert torch.equal(reconstructed, out["z_static"])


class TestCustomDimensions:
    def test_custom_d_immutable(self):
        enc = _make_encoder(d_immutable=4)
        assert enc.d_immutable == 4
        assert enc.d_slow == D_STATIC - 4
        out = enc(torch.randn(BATCH, INPUT_DIM))
        assert out["z_static_immutable"].shape == (BATCH, 4)
        assert out["z_static_slow"].shape == (BATCH, D_STATIC - 4)

    def test_d_immutable_default_is_half_d_static(self):
        enc = _make_encoder()
        assert enc.d_immutable == D_STATIC // 2

    def test_different_batch_sizes(self):
        enc = _make_encoder()
        for bs in [1, 4, 32]:
            out = enc(torch.randn(bs, INPUT_DIM))
            assert out["z_full"].shape == (bs, D_STATIC + D_DYNAMIC + D_CONTROLLABLE)


class TestGradients:
    def test_gradients_flow_through_all_heads(self):
        enc = _make_encoder()
        x = torch.randn(BATCH, INPUT_DIM)
        out = enc(x)
        out["z_full"].sum().backward()
        for name, param in enc.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_heads_are_independent(self):
        """dynamic_head and controllable_head have no overlapping parameters."""
        enc = _make_encoder()
        x = torch.randn(BATCH, INPUT_DIM)

        # Backprop through z_dynamic: dynamic_head grads, controllable_head no grads
        out = enc(x)
        out["z_dynamic"].sum().backward()
        assert enc.dynamic_head[0].weight.grad is not None, "dynamic_head has no grad"
        assert enc.controllable_head[0].weight.grad is None, (
            "controllable_head should not have grad"
        )

        enc.zero_grad()

        # Backprop through z_controllable: controllable_head grads, dynamic_head no grads
        out = enc(x)
        out["z_controllable"].sum().backward()
        assert enc.controllable_head[0].weight.grad is not None, "controllable_head has no grad"
        assert enc.dynamic_head[0].weight.grad is None, "dynamic_head should not have grad"


class TestDeterminism:
    def test_eval_mode_deterministic(self):
        enc = _make_encoder()
        enc.eval()
        x = torch.randn(BATCH, INPUT_DIM)
        with torch.no_grad():
            out1 = enc(x)
            out2 = enc(x)
        assert torch.equal(out1["z_full"], out2["z_full"])

    def test_same_input_same_output(self):
        enc = _make_encoder()
        enc.eval()
        x = torch.randn(BATCH, INPUT_DIM)
        with torch.no_grad():
            out1 = enc(x)
            out2 = enc(x)
        for key in ["z_static", "z_dynamic", "z_controllable", "z_full"]:
            assert torch.equal(out1[key], out2[key])
