import torch
import torch.nn as nn

class StructuredEncoderSmall(nn.Module):
    """
    SMALLER encoder to prevent instant memorization.
    Forces model to learn compressed representations.
    """
    def __init__(self, input_channels=3, d_static=16, d_dynamic=32, d_controllable=16):
        super().__init__()
        
        self.d_static = d_static
        self.d_dynamic = d_dynamic
        self.d_controllable = d_controllable
        
        # SMALLER backbone
        self.conv_backbone = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, 1, 1),  # Reduced from 32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),              # Reduced from 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),              # Reduced from 128
            nn.ReLU(),
        )
        
        self._static_head = None
        self._dynamic_head = None
        self._controllable_head = None
    
    def _init_heads(self, feature_dim):
        if self._static_head is None:
            # SMALLER heads (64 hidden instead of 256)
            self._static_head = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.d_static)
            )
            
            self._dynamic_head = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.d_dynamic)
            )
            
            self._controllable_head = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.d_controllable)
            )
    
    def forward(self, x):
        features = self.conv_backbone(x)
        features = features.flatten(1)
        
        if self._static_head is None:
            self._init_heads(features.shape[1])
            print(f"Initialized heads with input dim: {features.shape[1]}")
        
        z_static = self._static_head(features)
        z_dynamic = self._dynamic_head(features)
        z_controllable = self._controllable_head(features)
        
        return {
            'z_static': z_static,
            'z_dynamic': z_dynamic,
            'z_controllable': z_controllable,
            'z_full': torch.cat([z_static, z_dynamic, z_controllable], dim=-1)
        }
