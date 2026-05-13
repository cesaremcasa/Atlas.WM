import torch
import torch.nn as nn

class StructuredDynamics(nn.Module):
    """
    Predicts next latent state with structured transitions.
    
    CHANGE: Switching to SOFT constraint for static component.
    Hard constraint caused artificial 0.0000 loss.
    """
    def __init__(self, d_static=16, d_dynamic=32, d_controllable=16, action_dim=8):
        super().__init__()
        
        self.d_static = d_static
        self.d_dynamic = d_dynamic
        self.d_controllable = d_controllable
        
        # Static: Small residual network (allows tiny drift)
        self.static_net = nn.Sequential(
            nn.Linear(d_static, 32),
            nn.ReLU(),
            nn.Linear(32, d_static)
        )
        
        # Dynamic: Autonomous evolution
        self.dynamic_net = nn.Sequential(
            nn.Linear(d_dynamic, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, d_dynamic),
        )
        
        # Controllable: Action-conditioned
        self.control_net = nn.Sequential(
            nn.Linear(d_controllable + action_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, d_controllable),
        )
    
    def forward(self, z_dict, action):
        """
        Args:
            z_dict: Dict with keys ['z_static', 'z_dynamic', 'z_controllable']
            action: Tensor [B, action_dim]
        """
        z_static = z_dict['z_static']
        z_dynamic = z_dict['z_dynamic']
        z_controllable = z_dict['z_controllable']
        
        # Soft Constraint: Static can change slightly (residual)
        delta_static = self.static_net(z_static) 
        z_static_next = z_static + delta_static
        
        # Dynamic evolves
        delta_dynamic = self.dynamic_net(z_dynamic)
        z_dynamic_next = z_dynamic + delta_dynamic
        
        # Controllable responds to action
        control_input = torch.cat([z_controllable, action], dim=-1)
        delta_controllable = self.control_net(control_input)
        z_controllable_next = z_controllable + delta_controllable
        
        return {
            'z_static': z_static_next,
            'z_dynamic': z_dynamic_next,
            'z_controllable': z_controllable_next,
            'z_full': torch.cat([
                z_static_next, 
                z_dynamic_next, 
                z_controllable_next
            ], dim=-1)
        }
