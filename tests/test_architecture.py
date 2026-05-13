import torch

from atlas_wm.models.continuous_encoder import ContinuousEncoder


def test_encoder_shapes():
    """Encoder output keys and shapes match declared dimensions."""
    model = ContinuousEncoder(input_dim=6, d_static=16, d_dynamic=32, d_controllable=16)
    x = torch.randn(4, 6)
    out = model(x)

    assert out["z_static"].shape == (4, 16), "z_static shape mismatch"
    assert out["z_dynamic"].shape == (4, 32), "z_dynamic shape mismatch"
    assert out["z_controllable"].shape == (4, 16), "z_controllable shape mismatch"
    assert out["z_full"].shape == (4, 64), "z_full shape mismatch"
