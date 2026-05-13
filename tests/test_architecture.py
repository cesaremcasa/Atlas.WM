import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.structured_encoder import StructuredEncoder


def test_encoder_shapes():
    """Test that encoder outputs correct latent dimensions."""
    model = StructuredEncoder(d_static=32, d_dynamic=64, d_controllable=32)
    dummy_input = torch.randn(4, 3, 10, 10)

    output = model(dummy_input)

    assert output['z_static'].shape == (4, 32), "Static shape mismatch"
    assert output['z_dynamic'].shape == (4, 64), "Dynamic shape mismatch"
    assert output['z_controllable'].shape == (4, 32), "Controllable shape mismatch"
    assert output['z_full'].shape == (4, 128), "Full shape mismatch"

    print("✅ test_encoder_shapes PASSED")

if __name__ == "__main__":
    test_encoder_shapes()
