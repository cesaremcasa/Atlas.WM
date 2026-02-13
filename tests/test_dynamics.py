import torch
import sys
import os

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.structured_dynamics import StructuredDynamics

def test_dynamics_shapes():
    """Validate output shapes for all latent components."""
    print("Testing shapes...")
    dynamics = StructuredDynamics(d_static=32, d_dynamic=64, d_controllable=32, action_dim=4)
    
    z_dict = {
        'z_static': torch.randn(4, 32),
        'z_dynamic': torch.randn(4, 64),
        'z_controllable': torch.randn(4, 32),
    }
    action = torch.randn(4, 4)
    
    z_next = dynamics(z_dict, action)
    
    assert z_next['z_static'].shape == (4, 32), "Static dimension mismatch"
    assert z_next['z_dynamic'].shape == (4, 64), "Dynamic dimension mismatch"
    assert z_next['z_controllable'].shape == (4, 32), "Controllable dimension mismatch"
    assert z_next['z_full'].shape == (4, 128), "Full latent dimension mismatch"
    
    print("PASS: test_dynamics_shapes")

def test_static_immutability():
    """
    CRITICAL TEST: Static component MUST be bit-for-bit identical.
    This is an architectural guarantee, not an optimization target.
    """
    print("Testing static immutability...")
    dynamics = StructuredDynamics()
    dynamics.eval()
    
    z_dict = {
        'z_static': torch.randn(4, 32),
        'z_dynamic': torch.randn(4, 64),
        'z_controllable': torch.randn(4, 32),
    }
    action = torch.randn(4, 4)
    
    with torch.no_grad():
        z_next = dynamics(z_dict, action)
    
    # HARD CHECK: Not just "close", but IDENTICAL
    assert torch.equal(z_next['z_static'], z_dict['z_static']), \
        "Static component changed! Architectural constraint violated."
    
    print("PASS: test_static_immutability (bit-perfect match)")

def test_static_no_gradient_flow():
    """
    Verify that gradients CANNOT flow to z_static during dynamics update.
    """
    print("Testing static gradient isolation...")
    dynamics = StructuredDynamics()
    dynamics.train()
    
    z_dict = {
        'z_static': torch.randn(4, 32, requires_grad=True),
        'z_dynamic': torch.randn(4, 64, requires_grad=True),
        'z_controllable': torch.randn(4, 32, requires_grad=True),
    }
    action = torch.randn(4, 4)
    
    # Forward pass
    z_next = dynamics(z_dict, action)
    
    # Try to backprop through full output
    loss = z_next['z_full'].sum()
    loss.backward()
    
    # Static should have NO gradient (detached)
    # Note: In PyTorch, comparing to None is safer than checking for zeros
    # because detached tensors might not even have a grad attribute populated
    # depending on the graph construction, though here it returns a new tensor.
    assert z_dict['z_static'].grad is None, \
        "Gradients leaked to z_static! Detach failed."
    
    # Dynamic and controllable SHOULD have gradients
    assert z_dict['z_dynamic'].grad is not None, "Dynamic not learning"
    assert z_dict['z_controllable'].grad is not None, "Controllable not learning"
    
    print("PASS: test_static_no_gradient_flow")

def test_residual_updates():
    """Verify that dynamic and controllable components use residual connections."""
    print("Testing residual connections...")
    dynamics = StructuredDynamics()
    dynamics.eval()
    
    z_dict = {
        'z_static': torch.randn(1, 32),
        'z_dynamic': torch.randn(1, 64),
        'z_controllable': torch.randn(1, 32),
    }
    
    # Zero action to test pure dynamic evolution
    action = torch.zeros(1, 4)
    
    with torch.no_grad():
        z_next = dynamics(z_dict, action)
    
    # With zero action, control_net should output small values (near init)
    # but dynamic_net will transform input. 
    # This test mostly ensures the model runs without error in zero-action case.
    # A deeper test would mock the networks to return specific values.
    
    # Simple sanity: output is not identical to input (it evolved)
    # Unless the nets are initialized to zero, which they aren't.
    assert not torch.equal(z_next['z_dynamic'], z_dict['z_dynamic']), \
        "Dynamic should have changed"
    
    print("PASS: test_residual_updates")

if __name__ == "__main__":
    test_dynamics_shapes()
    test_static_immutability()
    test_static_no_gradient_flow()
    test_residual_updates()
    print("\nALL DYNAMICS TESTS PASSED")
