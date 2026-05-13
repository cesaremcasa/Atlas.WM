import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.atlas_dataset import ATLASDataset
from src.models.structured_encoder import StructuredEncoder
from src.models.structured_dynamics import StructuredDynamics

def compute_loss(z_t, z_t1_pred, z_t1_true):
    pred_loss = F.mse_loss(z_t1_pred['z_full'], z_t1_true['z_full'])
    static_pen = F.mse_loss(z_t1_pred['z_static'], z_t['z_static'])
    return pred_loss + 5.0 * static_pen

def train_epoch(encoder, dynamics, loader, opt, device):
    encoder.train(); dynamics.train()
    total = 0; n=0
    for batch in loader:
        obs = batch['obs'].to(device); act = batch['action'].to(device)
        n_obs = batch['next_obs'].to(device)
        
        # No noise on coordinates (precision matters)
        z_t = encoder(obs); z_t1 = encoder(n_obs); z_p = dynamics(z_t, act)
        loss = compute_loss(z_t, z_p, z_t1)
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(dynamics.parameters()), 1.0)
        opt.step()
        total += loss.item(); n+=1
    return total/n

def validate(encoder, dynamics, loader, device):
    encoder.eval(); dynamics.eval()
    total = 0; n=0
    with torch.no_grad():
        for batch in loader:
            obs = batch['obs'].to(device); act = batch['action'].to(device)
            n_obs = batch['next_obs'].to(device)
            z_t = encoder(obs); z_t1 = encoder(n_obs); z_p = dynamics(z_t, act)
            loss = compute_loss(z_t, z_p, z_t1)
            total += loss.item(); n+=1
    return total/n

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = ATLASDataset('data/processed', 'train')
    val = ATLASDataset('data/processed', 'val')
    train_l = DataLoader(train, batch_size=256, shuffle=True)
    val_l = DataLoader(val, batch_size=256)
    
    # Continuous Vector Input
    enc = StructuredEncoder(input_dim=6, d_static=16, d_dynamic=32, d_controllable=16).to(device)
    dyn = StructuredDynamics(16, 32, 16, action_dim=8).to(device)
    opt = optim.Adam(list(enc.parameters())+list(dyn.parameters()), lr=0.001)
    
    print(f"Train: {len(train)} | Val: {len(val)}")
    for e in range(20):
        t = train_epoch(enc, dyn, train_l, opt, device)
        v = validate(enc, dyn, val_l, device)
        print(f"Epoch {e+1} | Train: {t:.4e} | Val: {v:.4e}")
        
        if (e+1)%5==0:
            torch.save({'enc':enc.state_dict(), 'dyn':dyn.state_dict()}, f"checkpoints/phase1_cont_ep{e+1}.pt")

if __name__ == "__main__":
    main()
