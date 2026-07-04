"""Dissipative symplectic dynamics head tests (v4 B13)."""

import torch

from atlas_wm.models.structured_dynamics import StructuredDynamics


def _z(batch=4, d_static=16, d_dynamic=32, d_controllable=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "z_static": torch.randn(batch, d_static, generator=g),
        "z_dynamic": torch.randn(batch, d_dynamic, generator=g),
        "z_controllable": torch.randn(batch, d_controllable, generator=g),
    }


class TestHamiltonianHead:
    def test_forward_shapes_and_passthrough(self):
        dyn = StructuredDynamics(dynamics_head="hamiltonian")
        z = _z()
        out = dyn(z, torch.zeros(4, 8))
        assert out["z_full"].shape == (4, 64)
        torch.testing.assert_close(out["z_static_immutable"], z["z_static"][:, :8])

    def test_damping_contracts_momentum_without_force(self):
        # With the force field zeroed, ||p'|| <= ||p|| strictly (sigmoid < 1):
        # the head is dissipative by construction, like friction.
        dyn = StructuredDynamics(dynamics_head="hamiltonian")
        with torch.no_grad():
            for prm in dyn.force_net.parameters():
                prm.zero_()
        z = _z(seed=3)
        out = dyn(z, torch.zeros(4, 8))
        p, p_next = z["z_dynamic"][:, 16:], out["z_dynamic"][:, 16:]
        assert (p_next.norm(dim=-1) < p.norm(dim=-1)).all()

    def test_damping_depends_on_slow_latent(self):
        dyn = StructuredDynamics(dynamics_head="hamiltonian")
        z1, z2 = _z(seed=5), _z(seed=5)
        z2["z_static"] = z2["z_static"].clone()
        z2["z_static"][:, 8:] += 2.0  # perturb only the slow slice
        o1, o2 = dyn(z1, torch.zeros(4, 8)), dyn(z2, torch.zeros(4, 8))
        assert not torch.allclose(o1["z_dynamic"], o2["z_dynamic"])

    def test_invalid_config_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="dynamics_head"):
            StructuredDynamics(dynamics_head="nope")
        with pytest.raises(ValueError, match="even d_dynamic"):
            StructuredDynamics(d_dynamic=31, dynamics_head="hamiltonian")
