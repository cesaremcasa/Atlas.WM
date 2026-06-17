"""ONNX export tests (Block 13).

Verifies the encoder and dynamics export to valid ONNX graphs, that the two
graphs compose (encoder output feeds dynamics input), and that ONNX inference
matches PyTorch within tolerance.

Skipped automatically when the optional ``onnx`` / ``onnxruntime`` dependencies
are absent (they are not part of the pinned CI toolchain), so the suite stays
green without them.
"""

import numpy as np
import pytest
import torch

from atlas_wm.export.onnx_export import (
    DynamicsONNX,
    EncoderONNX,
    d_full,
    export_dynamics,
    export_encoder,
)
from atlas_wm.models.continuous_encoder import ContinuousEncoder
from atlas_wm.models.structured_dynamics import StructuredDynamics

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

D_STATIC, D_DYNAMIC, D_CONTROLLABLE, ACTION_DIM = 16, 32, 16, 8
WIDTH = d_full(D_STATIC, D_DYNAMIC, D_CONTROLLABLE)


def _encoder() -> ContinuousEncoder:
    torch.manual_seed(0)
    enc = ContinuousEncoder(
        input_dim=6, d_static=D_STATIC, d_dynamic=D_DYNAMIC, d_controllable=D_CONTROLLABLE
    )
    enc.eval()
    return enc


def _dynamics() -> StructuredDynamics:
    torch.manual_seed(1)
    dyn = StructuredDynamics(
        d_static=D_STATIC,
        d_dynamic=D_DYNAMIC,
        d_controllable=D_CONTROLLABLE,
        action_dim=ACTION_DIM,
    )
    dyn.eval()
    return dyn


def test_z_full_width_is_64():
    assert WIDTH == 64


def test_encoder_onnx_matches_torch(tmp_path):
    enc = _encoder()
    path = str(tmp_path / "encoder.onnx")
    export_encoder(enc, path, input_dim=6)

    obs = torch.randn(5, 6)
    with torch.no_grad():
        torch_out = EncoderONNX(enc)(obs).numpy()

    sess = ort.InferenceSession(path)
    onnx_out = sess.run(["z_full"], {"obs": obs.numpy()})[0]

    np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-4, atol=1e-5)


def test_dynamics_onnx_matches_torch(tmp_path):
    dyn = _dynamics()
    path = str(tmp_path / "dynamics.onnx")
    export_dynamics(dyn, path, action_dim=ACTION_DIM)

    z_full = torch.randn(5, WIDTH)
    action = torch.zeros(5, ACTION_DIM)
    action[:, 0] = 1.0
    with torch.no_grad():
        torch_out = DynamicsONNX(dyn)(z_full, action).numpy()

    sess = ort.InferenceSession(path)
    onnx_out = sess.run(["z_full_next"], {"z_full": z_full.numpy(), "action": action.numpy()})[0]

    np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-4, atol=1e-5)


def test_graphs_compose_into_rollout(tmp_path):
    """Encoder output z_full must be a valid dynamics input (shapes line up)."""
    enc, dyn = _encoder(), _dynamics()
    enc_path = str(tmp_path / "encoder.onnx")
    dyn_path = str(tmp_path / "dynamics.onnx")
    export_encoder(enc, enc_path, input_dim=6)
    export_dynamics(dyn, dyn_path, action_dim=ACTION_DIM)

    obs = np.random.randn(3, 6).astype(np.float32)
    action = np.zeros((3, ACTION_DIM), dtype=np.float32)
    action[:, 2] = 1.0

    enc_sess = ort.InferenceSession(enc_path)
    dyn_sess = ort.InferenceSession(dyn_path)

    z_full = enc_sess.run(["z_full"], {"obs": obs})[0]
    assert z_full.shape == (3, WIDTH)
    z_next = dyn_sess.run(["z_full_next"], {"z_full": z_full, "action": action})[0]
    assert z_next.shape == (3, WIDTH)


def test_dynamic_batch_axis(tmp_path):
    """Exported encoder accepts a batch size different from the export-time dummy."""
    enc = _encoder()
    path = str(tmp_path / "encoder.onnx")
    export_encoder(enc, path, input_dim=6)
    sess = ort.InferenceSession(path)
    for batch in (1, 7, 64):
        out = sess.run(["z_full"], {"obs": np.random.randn(batch, 6).astype(np.float32)})[0]
        assert out.shape == (batch, WIDTH)


def test_immutable_passthrough_preserved_in_onnx(tmp_path):
    """AD-2 passthrough must survive export: z_static_immutable unchanged by dynamics."""
    dyn = _dynamics()
    path = str(tmp_path / "dynamics.onnx")
    export_dynamics(dyn, path, action_dim=ACTION_DIM)

    z_full = np.random.randn(4, WIDTH).astype(np.float32)
    action = np.zeros((4, ACTION_DIM), dtype=np.float32)
    action[:, 1] = 1.0
    sess = ort.InferenceSession(path)
    z_next = sess.run(["z_full_next"], {"z_full": z_full, "action": action})[0]

    d_immutable = D_STATIC // 2
    np.testing.assert_allclose(
        z_full[:, :d_immutable], z_next[:, :d_immutable], rtol=1e-5, atol=1e-6
    )
