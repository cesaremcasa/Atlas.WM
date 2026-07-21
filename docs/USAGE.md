# Atlas.WM - Usage Guide (v4.0)

## Install

```bash
git clone https://github.com/cesaremcasa/Atlas.WM.git && cd Atlas.WM
uv sync --extra dev          # or: pip install -e ".[dev]"
```

## The full pipeline

```bash
# 1. Generate data (choose environment and policy)
python scripts/generate_data.py --randomize-physics --process-noise-std 0.05 \
    --episode-reset-prob 0.02 --seed 42 --num-samples 50000            # gridworld, random
python scripts/generate_data.py --policy active ...                    # + info-seeking policy (B11)
python scripts/generate_data.py --env mujoco --randomize-physics ...   # MuJoCo tier (B14)

# 2. Split (episode-grouped, fingerprinted - re-splits automatically when raw changes)
python scripts/split_data.py [--raw-dir data/raw --processed-dir data/processed]

# 3. Train the world model (VICReg + prediction grounding + K-step rollouts)
python scripts/train.py --config configs/experiments/v3_variable_physics.yaml --seed 42

# 4. Evaluate: open-loop rollout MSE by horizon + AD-2 passthrough check
python scripts/evaluate.py --checkpoint checkpoints/best_model.safetensors --horizon 10

# 5. Train the physics belief encoder (engineered features, distributional head)
python scripts/train_physics_belief.py --config configs/experiments/v3_variable_physics.yaml \
    --window-k 40 --output checkpoints/physics_belief.safetensors

# 6. Probe identifiability (episode-grouped splits)
python scripts/probe_physics.py --checkpoint checkpoints/best_model.safetensors \
    --belief-checkpoint checkpoints/physics_belief.safetensors

# 7. Belief-condition the world model (RMA phase 2)
python scripts/precompute_belief.py --belief-checkpoint checkpoints/physics_belief.safetensors
# then set training.use_belief: true and retrain (step 3)

# 8. Reproduce the friction_agent identifiability evidence
python scripts/oracle_friction_agent.py --episodes 400 --process-noise-std 0.05
```

## Key config switches (`configs/base.yaml`)

| Key | Values | Meaning |
|---|---|---|
| `model.frame_stack` | 1 / **2** | 2 makes velocity observable (B6) |
| `model.dynamics_head` | **residual** / hamiltonian | symplectic-dissipative head (B13; OOD-robust, underfits in-dist) |
| `training.objective` | **vicreg** / ema / legacy | stable self-predictive recipes (B7) |
| `training.rollout_k` | 1 / **4** | K-step self-fed rollout training (B8) |
| `training.use_belief` | **false** / true | condition z_slow on causal beliefs (B12) |
| `training.lambda_imm_*` | **0.0** | immutable anchor (B9); enable ≥0.1 when episode identity is observable |
| `training.seed` | **42** | full training reproducibility (B3) |

## Checkpoints & security

Safetensors-only, embedded metadata (dims, seed, objective, git_sha).
Sign a checkpoint dir: `python scripts/sign_checkpoint.py` (HMAC-SHA256,
`ATLAS_SIGNING_KEY`). Production loads: `load_checkpoint(...,
require_signature=True)` - fail-closed (B17). Export:
`python scripts/export_onnx.py --checkpoint ... --out-dir export/`.

## Testing

```bash
pytest            # 195 tests: unit, physics contracts, canaries, security, regression locks
make ci-local     # lint + type + test + security gates
python scripts/chaos_physics.py   # randomized physics tripwire
```
