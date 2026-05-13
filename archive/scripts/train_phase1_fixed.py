import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from src.models.structured_encoder import StructuredEncoder
from src.models.structured_dynamics import StructuredDynamics
from src.data.atlas_dataset import ATLASDataset

def train():
    # Load config
    with open('configs/experiment.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset = ATLASDataset('data/processed', split='train')
    val_dataset = ATLASDataset('data/processed', split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize models
    encoder = StructuredEncoder(
        input_channels=config['model']['input_channels'],
        d_static=config['model']['d_static'],
        d_dynamic=config['model']['d_dynamic'],
        d_controllable=config['model']['d_controllable']
    ).to(device)
    
    dynamics = StructuredDynamics(
        d_static=config['model']['d_static'],
        d_dynamic=config['model']['d_dynamic'],
        d_controllable=config['model']['d_controllable'],
        action_dim=4
    ).to(device)
    
    # Optimizer
    params = list(encoder.parameters()) + list(dynamics.parameters())
    optimizer = optim.Adam(params, lr=config['training']['learning_rate'])
    
    print("Starting training...")
    
    for epoch in range(config['training']['num_epochs']):
        # Training
        encoder.train()
        dynamics.train()
        train_loss = 0.0
        
        for batch in train_loader:
            obs = batch['obs'].to(device)
            action = batch['action'].to(device)
            next_obs = batch['next_obs'].to(device)
            
            # Forward pass
            z_t = encoder(obs)
            z_t1_pred = dynamics(z_t, action)
            
            with torch.no_grad():
                z_t1_true = encoder(next_obs)
            
            # Compute loss
            loss = nn.functional.mse_loss(z_t1_pred['z_full'], z_t1_true['z_full'])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        encoder.eval()
        dynamics.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['obs'].to(device)
                action = batch['action'].to(device)
                next_obs = batch['next_obs'].to(device)
                
                z_t = encoder(obs)
                z_t1_pred = dynamics(z_t, action)
                z_t1_true = encoder(next_obs)
                
                loss = nn.functional.mse_loss(z_t1_pred['z_full'], z_t1_true['z_full'])
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save checkpoint with CORRECT keys
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),  # FIXED KEY
                'dynamics': dynamics.state_dict(),  # FIXED KEY
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, f'checkpoints/phase1_epoch_{epoch+1}.pt')
            print(f"Saved checkpoint to checkpoints/phase1_epoch_{epoch+1}.pt")
    
    print("Training complete.")

if __name__ == '__main__':
    train()
