import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.atlas_dataset import ATLASDataset
from src.models.structured_encoder import StructuredEncoder
from src.models.structured_dynamics import StructuredDynamics

print("Loading data...")
train_dataset = ATLASDataset('data/processed', split='train')
val_dataset = ATLASDataset('data/processed', split='val')

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

print("Initializing models...")
encoder = StructuredEncoder(input_channels=4, d_static=32, d_dynamic=64, d_controllable=32)
dynamics = StructuredDynamics(d_static=32, d_dynamic=64, d_controllable=32, action_dim=8)

encoder.eval()
dynamics.eval()

# Check ONE training batch
print("\n--- TRAINING BATCH CHECK ---")
batch = next(iter(train_loader))
obs = batch['obs']
action = batch['action']
next_obs = batch['next_obs']

z_t = encoder(obs)
z_t1_true = encoder(next_obs)
z_t1_pred = dynamics(z_t, action)

pred_loss = F.mse_loss(z_t1_pred['z_full'], z_t1_true['z_full'])
static_pen = F.mse_loss(z_t1_pred['z_static'], z_t['z_static'])

print(f"Raw Train Pred Loss: {pred_loss.item():.6f}")
print(f"Raw Train Static Penalty: {static_pen.item():.6f}")
print(f"Raw Total Train Loss: {(pred_loss + 5.0*static_pen).item():.6f}")

# Check ONE validation batch
print("\n--- VALIDATION BATCH CHECK ---")
batch = next(iter(val_loader))
obs = batch['obs']
action = batch['action']
next_obs = batch['next_obs']

z_t = encoder(obs)
z_t1_true = encoder(next_obs)
z_t1_pred = dynamics(z_t, action)

pred_loss = F.mse_loss(z_t1_pred['z_full'], z_t1_true['z_full'])
static_pen = F.mse_loss(z_t1_pred['z_static'], z_t['z_static'])

print(f"Raw Val Pred Loss: {pred_loss.item():.6f}")
print(f"Raw Val Static Penalty: {static_pen.item():.6f}")
print(f"Raw Total Val Loss: {(pred_loss + 5.0*static_pen).item():.6f}")

# Check if Latents are Zero
print("\n--- LATENT SANITY CHECK ---")
print(f"Z_t Mean: {z_t['z_full'].mean().item():.6f}")
print(f"Z_t Std: {z_t['z_full'].std().item():.6f}")
print(f"Z_t1_pred Mean: {z_t1_pred['z_full'].mean().item():.6f}")
print(f"Z_t1_pred Std: {z_t1_pred['z_full'].std().item():.6f}")
