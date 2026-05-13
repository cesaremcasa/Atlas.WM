import torch
import torch.nn as nn

class StructuredEncoder(nn.Module):
    """
    Encoder for Continuous State Space (6D Vector).
    """
    def __init__(self, input_dim=6, d_static=16, d_dynamic=32, d_controllable=16):
        super().__init__()
        
        # Input is flat vector [x, y, x, y, x, y]
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.static_head = nn.Linear(64, d_static)
        self.dynamic_head = nn.Linear(64, d_dynamic)
        self.controllable_head = nn.Linear(64, d_controllable)
    
    def forward(self, observation):
        features = self.backbone(observation)
        
        z_static = self.static_head(features)
        z_dynamic = self.dynamic_head(features)
        z_controllable = self.controllable_head(features)
        
        return {
            'z_static': z_static,
            'z_dynamic': z_dynamic,
            'z_controllable': z_controllable,
            'z_full': torch.cat([z_static, z_dynamic, z_controllable], dim=-1)
        }
