import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import sys
sys.path.insert(0, '.')
from src.models.continuous_encoder import ContinuousEncoder
from src.models.structured_dynamics import StructuredDynamics
from src.data.continuous_dataset import ContinuousDataset

def train():
    with open('configs/experiment.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = ContinuousDataset('data/processed', split='train')
    val_dataset = ContinuousDataset('data/processed', split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    encoder = ContinuousEncoder(input_dim=6, d_static=16, d_dynamic=32, d_controllable=16).to(device)
    dynamics = StructuredDynamics(d_static=16, d_dynamic=32, d_controllable=16, action_dim=8).to(device)
    
    params = list(encoder.parameters()) + list(dynamics.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    
    print("Starting training...")
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(50):
        encoder.train()
        dynamics.train()
        train_loss = 0.0
        
        for batch in train_loader:
            obs = batch['obs'].to(device)
            action = batch['action'].to(device)
            next_obs = batch['next_obs'].to(device)
            
            z_t = encoder(obs)
            z_t1_pred = dynamics(z_t, action)
            
            with torch.no_grad():
                z_t1_true = encoder(next_obs)
            
            # Loss with regularization
            pred_loss = nn.functional.mse_loss(z_t1_pred['z_full'], z_t1_true['z_full'])
            
            z_var = z_t1_pred['z_full'].var(dim=0).mean()
            var_penalty = torch.clamp(1.0 - z_var, min=0)
            
            loss = pred_loss + 0.01 * var_penalty
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
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
        
        print(f"Epoch {epoch+1}/50 | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'dynamics': dynamics.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, 'checkpoints/best_model.pt')
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print("Training complete.")

if __name__ == '__main__':
    train()
