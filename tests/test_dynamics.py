"""Tests for StructuredDynamics (AD-2 Hybrid Static Decomposition)."""

import torch

from atlas_wm.models.structured_dynamics import StructuredDynamics

D_STATIC = 16
D_DYNAMIC = 32
D_CONTROLLABLE = 16
ACTION_DIM = 8
BATCH = 4


def _make_dynamics(**kwargs) -> StructuredDynamics:
    defaults = dict(
        d_static=D_STATIC, d_dynamic=D_DYNAMIC, d_controllable=D_CONTROLLABLE, action_dim=ACTION_DIM
    )
    defaults.update(kwargs)
    return StructuredDynamics(**defaults)


def _make_z(batch: int = BATCH) -> dict[str, torch.Tensor]:
    dyn = _make_dynamics()
    return {
        "z_static_immutable": torch.randn(batch, dyn.d_immutable),
        "z_static_slow": torch.randn(batch, dyn.d_slow),
        "z_dynamic": torch.randn(batch, D_DYNAMIC),
        "z_controllable": torch.randn(batch, D_CONTROLLABLE),
    }


def test_dynamics_output_shapes():
    """All output tensors have expected shapes."""
    dyn = _make_dynamics()
    z = _make_z()
    action = torch.randn(BATCH, ACTION_DIM)
    out = dyn(z, action)

    assert out["z_static_immutable"].shape == (BATCH, dyn.d_immutable)
    assert out["z_static_slow"].shape == (BATCH, dyn.d_slow)
    assert out["z_static"].shape == (BATCH, D_STATIC)
    assert out["z_dynamic"].shape == (BATCH, D_DYNAMIC)
    assert out["z_controllable"].shape == (BATCH, D_CONTROLLABLE)
    assert out["z_full"].shape == (BATCH, D_STATIC + D_DYNAMIC + D_CONTROLLABLE)
    assert out["delta_slow"].shape == (BATCH, dyn.d_slow)


def test_static_immutable_passthrough():
    """CRITICAL: z_static_immutable must be bit-for-bit identical across a step.

    This is an architectural guarantee, not an optimization target.
    """
    dyn = _make_dynamics()
    dyn.eval()
    z = _make_z()
    action = torch.randn(BATCH, ACTION_DIM)

    with torch.no_grad():
        out = dyn(z, action)

    assert torch.equal(out["z_static_immutable"], z["z_static_immutable"]), (
        "z_static_immutable changed — hard passthrough invariant violated."
    )


def test_static_slow_can_drift():
    """z_static_slow is allowed to drift (residual update, not identity)."""
    dyn = _make_dynamics()
    dyn.eval()
    z = _make_z()
    action = torch.randn(BATCH, ACTION_DIM)

    with torch.no_grad():
        out = dyn(z, action)

    assert not torch.equal(out["z_static_slow"], z["z_static_slow"]), (
        "z_static_slow is frozen — static_slow_net appears to be a no-op."
    )


def test_delta_slow_is_residual():
    """delta_slow == z_static_slow_next - z_static_slow_input."""
    dyn = _make_dynamics()
    z = _make_z()
    action = torch.randn(BATCH, ACTION_DIM)

    with torch.no_grad():
        out = dyn(z, action)

    expected = out["z_static_slow"] - z["z_static_slow"]
    assert torch.allclose(out["delta_slow"], expected), "delta_slow is not the residual delta."


def test_immutable_passthrough_gradient_is_identity():
    """Gradient through z_static_immutable is identity (no learned transformation).

    The architectural guarantee is NOT gradient isolation — it's that no learnable
    network transforms z_static_immutable. The gradient of any z_full-based loss
    wrt z_static_immutable must be a pure identity (grad == ones for .sum() loss).
    """
    dyn = _make_dynamics()
    dyn.train()

    z_imm = torch.randn(BATCH, dyn.d_immutable, requires_grad=True)
    z = {
        "z_static_immutable": z_imm,
        "z_static_slow": torch.randn(BATCH, dyn.d_slow, requires_grad=True),
        "z_dynamic": torch.randn(BATCH, D_DYNAMIC, requires_grad=True),
        "z_controllable": torch.randn(BATCH, D_CONTROLLABLE, requires_grad=True),
    }
    action = torch.randn(BATCH, ACTION_DIM)
    out = dyn(z, action)
    out["z_full"].sum().backward()

    # Identity passthrough: grad of sum loss wrt z_imm is all-ones (no network distortion)
    assert z_imm.grad is not None
    assert torch.allclose(z_imm.grad, torch.ones_like(z_imm)), (
        "Gradient through z_static_immutable is not identity — a network is transforming it."
    )


def test_immutable_no_learnable_transform():
    """static_slow_net input size == d_slow, not d_immutable (structural guarantee)."""
    dyn = _make_dynamics()
    first_layer = dyn.static_slow_net[0]
    assert first_layer.in_features == dyn.d_slow, (
        f"static_slow_net operates on d_slow={dyn.d_slow} dims, "
        f"but in_features={first_layer.in_features}"
    )


def test_legacy_z_static_fallback():
    """When z_static_immutable/slow absent, falls back to splitting z_static."""
    dyn = _make_dynamics()
    z_static = torch.randn(BATCH, D_STATIC)
    z = {
        "z_static": z_static,
        "z_dynamic": torch.randn(BATCH, D_DYNAMIC),
        "z_controllable": torch.randn(BATCH, D_CONTROLLABLE),
    }
    action = torch.randn(BATCH, ACTION_DIM)

    with torch.no_grad():
        out = dyn(z, action)

    assert torch.equal(out["z_static_immutable"], z_static[:, : dyn.d_immutable])
    assert out["z_full"].shape == (BATCH, D_STATIC + D_DYNAMIC + D_CONTROLLABLE)


def test_residual_dynamic_evolution():
    """Dynamic component changes across a step (autonomous residual update)."""
    dyn = _make_dynamics()
    dyn.eval()
    z = _make_z()
    action = torch.zeros(BATCH, ACTION_DIM)

    with torch.no_grad():
        out = dyn(z, action)

    assert not torch.equal(out["z_dynamic"], z["z_dynamic"]), (
        "z_dynamic unchanged with zero action — dynamic_net appears to be a no-op."
    )
