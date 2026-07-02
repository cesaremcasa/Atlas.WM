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

## Physics identifiability (Block 14 finding — RETRACTED in v4)

> **Retraction (2026-07-02).** The Block-14 conclusion that `friction_agent`
> is "not identifiable" is **wrong**, and the numbers below it are invalid.
> Two defects were found in the v4 red-team review (`docs/v4.0-ROADMAP.md`):
>
> 1. **The oracle was broken and unreproducible.** The MLP oracle probe that
>    scored R² < 0 was never committed to the repo, and MSE-based fits are
>    destroyed by heavy-tailed wall/obstacle-bounce outliers. A robust
>    closed-form estimator — median of per-step velocity-decay ratios on
>    position-only observations (`scripts/oracle_friction_agent.py`) —
>    recovers `friction_agent` with **R² = 0.85, MAE = 0.004** over 400
>    random-policy episodes on the corrected environment (R² = 0.98 with a
>    stricter gravity gate). The agent is the *most* identifiable parameter,
>    not the least: it is the only object under continuous known excitation.
> 2. **The environment silently destroyed the box-physics signal.** Boxes did
>    not collide with walls and drifted out of the grid (observed range
>    [−62, +77] on a [0, 20] grid), leaving the 1–10-unit gravity interaction
>    band after a few steps. The reported "oracle ceiling of 0.15–0.45 R²"
>    for `gravity`/`friction_box` measured this simulation bug (fixed, v4 B1).
>
> Consequences: `friction_agent` returns to the identification target set,
> and the belief-encoder results (R² ≈ 0–0.15) are void — they were further
> confounded by a 20× data-scale mismatch between the training and probe
> pipelines (fixed, v4 B2). A full re-baseline on the corrected environment
> is roadmap block **B5**; this section will be rewritten with those numbers.
> Until then, no v3.x identifiability claim stands.

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
