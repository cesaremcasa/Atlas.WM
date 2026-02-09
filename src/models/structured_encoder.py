import torch
import torch.nn as nn

class StructuredEncoder(nn.Module):
    """
    Maps observations to structured latent space.
    Three separate heads ensure interpretable decomposition.
    Uses Lazy Linear Initialization to handle unknown feature dimensions automatically.
    """
    def __init__(self, 
                 input_channels=3,
                 d_static=32,
                 d_dynamic=64, 
                 d_controllable=32):
        super().__init__()
        
        # Shared feature extractor (Conv Backbone)
        self.conv_backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),      
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),     
            nn.ReLU(),
        )
        
        # We will initialize the Linear heads lazily in the first forward pass
        self.static_head = None
        self.dynamic_head = None
        self.controllable_head = None
        
        # Store dimensions to build heads later
        self.d_static = d_static
        self.d_dynamic = d_dynamic
        self.d_controllable = d_controllable
    
    def _init_heads(self, feature_dim, device):
        """Helper to initialize linear heads after seeing first input."""
        if self.static_head is None:
            self.static_head = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.d_static),
            ).to(device)
            
            self.dynamic_head = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.d_dynamic),
            ).to(device)
            
            self.controllable_head = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.d_controllable),
            ).to(device)
            
            print(f"Initialized Linear heads with input dim: {feature_dim}")

    def forward(self, observation):
        """
        Args:
            observation: [B, C, H, W]
        Returns:
            dict: {
                'z_static': [B, d_static],
                'z_dynamic': [B, d_dynamic],
                'z_controllable': [B, d_controllable],
                'z_full': [B, d_static + d_dynamic + d_controllable]
            }
        """
        features = self.conv_backbone(observation)
        
        # Flatten
        features = features.flatten(1) 
        feature_dim = features.shape[1]
        device = observation.device
        
        # Initialize heads on first pass
        self._init_heads(feature_dim, device)

        # Extract structured sub-spaces
        z_static = self.static_head(features)
        z_dynamic = self.dynamic_head(features)
        z_controllable = self.controllable_head(features)
        
        # Concatenate for dynamics model
        z_full = torch.cat([z_static, z_dynamic, z_controllable], dim=-1)
        
        return {
            'z_static': z_static,
            'z_dynamic': z_dynamic,
            'z_controllable': z_controllable,
            'z_full': z_full
        }

if __name__ == "__main__":
    # Quick test
    model = StructuredEncoder(input_channels=3, d_static=32, d_dynamic=64, d_controllable=32)
    dummy_input = torch.randn(4, 3, 10, 10) # Batch of 4
    
    output = model(dummy_input)
    
    print("Structured Encoder Test:")
    print(f"  Input Shape: {dummy_input.shape}")
    print(f"  Z_static: {output['z_static'].shape}")
    print(f"  Z_dynamic: {output['z_dynamic'].shape}")
    print(f"  Z_controllable: {output['z_controllable'].shape}")
    print(f"  Z_full: {output['z_full'].shape}")
