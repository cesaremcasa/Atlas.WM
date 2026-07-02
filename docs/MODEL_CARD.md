# Model Card — Atlas.WM v3.1

A structured world model that decomposes its latent space into interpretable
components with **architectural** (not merely learned) guarantees.

## Model details

- **Name / version:** Atlas.WM 3.1.0
- **Type:** Latent-space world model (encoder + structured one-step dynamics)
- **Author:** Cesar Augusto
- **License:** MIT
- **Frameworks:** PyTorch ≥ 2.1; export to ONNX (opset 17)
- **Decision record:** AD-1 … AD-8 (see `docs/Atlas-WM-v3-Architecture-Plan.md`)

### Components

| Module | Signature | Role |
|--------|-----------|------|
| `ContinuousEncoder` | `obs[B,6] → z_full[B,64]` | Encodes observations into the structured latent |
| `StructuredDynamics` | `(z, action[B,8]) → z'` | One-step latent transition |
| `ActionInvarianceCritic` | `z_static_immutable → action` | Adversary enforcing identifiability (AD-3) |
| `EntityEncoder` | `entities → z` | Permutation-equivariant variant for variable object counts |
| `PhysicsBeliefEncoder` | `obs_window[B,K,6] → z_static_slow[B,8]` | GRU over K consecutive obs → physics belief (Block 14) |
| `PhysicsHead` | `z_static_slow[B,8] → physics_hat[B,n]` | Supervised head; predicts the recoverable physics subset `{gravity, friction_box}` (Block 14) |

### Latent layout (`z_full`, width 64)

```
[ z_static (16) | z_dynamic (32) | z_controllable (16) ]
   = [ z_static_immutable (8) | z_static_slow (8) ] | ...
```

| Sub-space | Width | Guarantee |
|-----------|-------|-----------|
| `z_static_immutable` | 8 | Hard passthrough in dynamics — bit-identical across time (AD-2) |
| `z_static_slow` | 8 | Soft residual; drift penalized by `lambda_slow_drift` |
| `z_dynamic` | 32 | Autonomous residual evolution |
| `z_controllable` | 16 | Action-conditioned residual |

## Intended use

- **In scope:** research on disentangled/identifiable world models; studying
  domain randomization and latent identifiability of physical parameters;
  short-horizon latent rollout on `CruelGridworld`-style continuous dynamics.
- **Out of scope:** safety-critical control, real-world robotics deployment, or
  any setting requiring calibrated uncertainty. This is a research artifact.

## Training data

- **Environment:** `CruelGridworld` — 3 objects (1 agent + 2 boxes) on a 20×20
  grid with non-linear gravity, friction, wall/obstacle bounces; 8 discrete
  actions; 6-D continuous observation.
- **Dataset:** ~50k random-exploration transitions (`generate_data.py`), split
  80/10/10, observations normalized to `[0, 1]`.
- **Variable-physics variant (Block 12):** gravity ∈ [2,8], friction_agent ∈
  [0.90,0.99], friction_box ∈ [0.95,0.995] resampled per episode, with optional
  Gaussian process noise; ground-truth parameters recorded for probing.

## Evaluation

- **Physics contract tests** assert gravity, friction, Newton's third law,
  bounds, and action effects (tripwire against silent regressions).
- **Determinism canary** (AD-7): same seed → bit-identical encoder/dynamics
  outputs; `z_static_immutable` constant across a rollout.
- **Rollout drift** stays finite and bounded over open-loop horizons.
- **Latent probe** (Block 12): ridge R² of physics decoded from `z_static_slow`
  vs. the immutable passthrough — direct evidence the decomposition routes
  variable physics where intended. (R² is near zero on an untrained encoder, as
  expected; train before interpreting.)
- **PhysicsBeliefEncoder probe** (Block 14): ridge R² of physics decoded from
  the GRU's output over a K-step window. Targets the **recoverable subset**
  `{gravity, friction_box}` only; single-step MLP gives R²≈0 by design.
  See *Physics identifiability* below for the empirical ceiling.
- **ONNX parity:** exported graphs match PyTorch within `rtol=1e-4`, and the
  immutable passthrough is preserved in the exported `dynamics.onnx`.

## Reproducibility & provenance

- Toolchain pinned with hashes in `requirements.lock` (AD-8); CI installs from it.
- Checkpoints stored only as `safetensors` (AD-4) with embedded metadata
  (`git_sha`, `config_hash`, `env_hash`, schema version) and optional
  HMAC-SHA256 signing (AD-4, key via `ATLAS_SIGNING_KEY`).
- Environment content-addressing via `env_hash` (AD-6) flags cross-environment
  loads.

## Physics identifiability (Block 14 — empirical finding)

Not all episode-level physics parameters are recoverable from observation
windows under the current environment and random-exploration data regime:

- **`friction_agent` is not identifiable.** The agent receives a fixed force
  impulse every step, which re-randomizes its velocity and masks the friction
  decay. An *oracle* probe — a large MLP given privileged hand-crafted dynamics
  features (per-step speeds, accelerations, velocity-decay ratios, inter-object
  distances) — still scores R² < 0 for `friction_agent`. It is therefore
  excluded from the identification target.
- **`gravity` and `friction_box` are weakly recoverable.** The oracle ceiling is
  ≈0.15–0.45 R². Gravity is an intermittent inter-object attraction (`G/dist²`,
  active only when objects are 1–10 apart); boxes are moved only by that
  coupling plus friction, so `friction_box` is the most observable parameter.
- **The GRU belief encoder under-performs the oracle ceiling** (mean R² ≈ 0–0.15
  on the recoverable subset, with longer ~50-step episodes and `window_k=20`).
  The raw-sequence → physics mapping is hard to learn from limited windows at
  this signal-to-noise ratio. A promising future direction is feeding the GRU
  engineered dynamics features (as the oracle uses) rather than raw positions.

This is a data-regime limitation, not a modeling bug: the belief-encoder
architecture (RMA/VariBAD-style recurrent distillation) is standard and correct.

## Limitations & ethical considerations

- Trained and evaluated only on a synthetic toy environment; no transfer claims.
- Identifiability guarantees on `z_static_immutable` are architectural
  (passthrough) and adversarial (critic); `z_static_slow` identifiability is
  empirical and depends on training quality — verify with the latent probe.
- Episode-level physics identification is limited by the data regime — see
  *Physics identifiability* above; only `{gravity, friction_box}` are targeted.
- No personal or sensitive data is involved.

## How to export

```bash
pip install 'atlas-wm[export]'
python scripts/export_onnx.py \
    --checkpoint checkpoints/best_model.safetensors --out-dir export/
# -> export/encoder.onnx, export/dynamics.onnx (compose for rollout)
```
