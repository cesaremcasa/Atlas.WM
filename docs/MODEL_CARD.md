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
  the GRU's output over a K-step window, targeting all three parameters
  (the v3.x `friction_agent` exclusion was retracted — see below); the
  single-step MLP gives R²≈0 by design.
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
> pipelines (fixed, v4 B2). No v3.x identifiability claim stands. The v4
> re-baseline below replaces them.

## Physics identifiability — v4.0 re-baseline (B5)

Measured on the corrected environment (boxes contained, B1), consistent data
pipeline (B2), seeded training (B3), and **episode-grouped probe splits** (the
v3.x sequential splits leaked physics labels across overlapping windows).
Dataset: 50k random-policy transitions, ~1,000 episodes of ~50 steps,
per-episode physics randomization, process noise σ = 0.05.

| Probe | gravity | friction_agent | friction_box |
|---|---|---|---|
| Closed-form oracle (median-of-ratios, same data) | — | **+0.865** | — |
| Single-step encoder, `z_static_slow` | −0.02 | −0.30 | −0.03 |
| Single-step encoder, `z_static_immutable` | −0.03 | −0.28 | −0.06 |
| GRU belief encoder (window 20, obs+Δobs+action) | −0.05 | −0.49 | −0.10 |

Two conclusions, both different from the retracted v3.x narrative:

1. **Identifiability is not data-limited.** The closed-form estimator
   recovers `friction_agent` with R² = 0.865 (MAE 0.006, 557 episodes) from
   the *same* noisy on-disk dataset — the information is present and
   extractable; the per-episode median averages the process noise out. The
   v3.x framing ("a data-regime limitation, not a modeling bug") had it
   backwards. (On a noise-free variant of the dataset the pattern is
   unchanged: oracle ≈ 0.88, learned encoders still negative.)
2. **The raw-sequence GRU recipe is the bottleneck.** Its training loss
   decreases while cross-episode validation R² stays negative for all three
   parameters: it memorizes training episodes instead of learning the
   dynamics → physics mapping. The gap between +0.865 (engineered per-step
   decay features + robust aggregation) and −0.49 (raw windows) on identical
   data is the project's clearest signal for what to build next: feed the
   belief encoder the oracle's dynamics features and train with an
   episode-contrastive objective (roadmap **B10**), then compare
   active-exploration data collection against random policies (**B11**).

Single-step probes hover at ≈ 0 as expected (physics is invisible in one
frame); mildly negative values reflect ridge variance at ~63 held-out
episodes. Reproduce:
`generate_data.py --randomize-physics --process-noise-std 0.05
--episode-reset-prob 0.02 --seed 42 --num-samples 50000`, `split_data.py`,
both trainers, then `probe_physics.py` (probe table) and the
`oracle_friction_agent.estimate_friction_agent` per-episode sweep over
`data/raw` (oracle row).

## Frame stacking (v4 B6)

`frame_stack: 2` is now the default model input: the previous same-episode
frame is concatenated to each observation (12-D input), making velocity
observable — a single position frame leaves one-step prediction ill-posed
(finding M2). Measured on the noisy re-baseline dataset (val split,
observation-space next-frame MSE, identical seeds):

| Next-frame predictor | 1 frame | 2 frames |
|---|---|---|
| Linear ridge (information ceiling proxy) | 0.000856 | **0.000274** |
| Trained world model (current v3-era loss) | 0.000921 | 0.001750 |

Two findings:

1. **The velocity information is real**: a linear model improves 3.1× with
   the stacked input.
2. **The current training recipe cannot exploit it** — the trained world
   model with 1 frame merely matches its linear ceiling, and with 2 frames
   lands 6× *below* it (a plain ridge beats the full
   encoder→dynamics→decoder stack). This is the same pattern as the
   oracle-vs-GRU identifiability gap: information present, L2-anchored
   self-predictive recipe unable to use it. The comparison is re-run in B7
   after the objective is replaced (EMA-target / VICReg).

## Stable objective + prediction grounding (v4 B7)

The v3.x objective (online-detached self-predictive target + L2 scale
anchor + a variance hinge on the wrong tensor) is replaced. Two candidates
were ablated, and a **prediction-grounding term** was added — nothing in the
old loss optimized the ``decoder∘dynamics`` composition that inference uses:

| Recipe (obs-space next-frame MSE, val) | 1 frame | 2 frames |
|---|---|---|
| legacy (L2 anchor) | 0.000921 | 0.001750 |
| EMA target + grounding | — | 0.001122 |
| **VICReg + grounding (new default)** | 0.001055 | **0.000996** |
| Linear ridge ceiling | 0.000856 | 0.000274 |

Findings:

1. **VICReg wins the ablation**: healthy latent dispersion (mean per-dim
   std ≈ 0.99) with no scale anchor, while the EMA-target encoder partially
   collapses here (std ≈ 0.09). ``objective: vicreg`` is the new default;
   ``ema`` remains available.
2. **Frame stacking now helps** (0.000996 vs 0.001055) — under the legacy
   recipe it was 90% *worse*. The B6 direction is vindicated.
3. **43% error reduction** vs the legacy 2-frame recipe, with
   ``lambda_latent_l2`` retired entirely.
4. **Open gap**: 3.6× above the 2-frame linear ceiling. The one-step
   information is demonstrably linearly extractable; remaining suspects are
   training length/schedule and the structured-latent routing bottleneck.
   Tracked for B8+ alongside the multi-step rollout loss.

Checkpoint selection and early stopping now use the objective-agnostic
observation-space next-frame MSE (latent losses are not comparable across
objectives).

## Multi-step rollout training (v4 B8)

Training now runs K-step self-fed rollouts (`rollout_k: 4` default): each
predicted latent feeds the next dynamics step, with per-step latent
supervision and prediction grounding. One-step teacher forcing never
exposed the model to its own compounding error — the open-loop regime a
world model is actually used in. `scripts/evaluate.py` is now a real
evaluator: open-loop obs-space MSE by horizon plus the AD-2 passthrough
check.

Open-loop rollout on the noisy re-baseline val split (VICReg, 2 frames):

| Horizon | one-step trained (K=1) | rollout-trained (K=4) |
|---|---|---|
| 1 | **0.000993** | 0.001157 |
| 4 | 0.010577 | **0.010267** |
| 10 | 0.038915 | **0.035682** |

The classic trade-off, mildly: one-step wins at h=1 (16%), rollout
training wins from h≥3 (8% at h=10). Gains are modest — note that the
per-step process noise (σ = 0.05) is irreducible and accumulates like a
random walk over horizons, so part of the long-horizon error floor cannot
be modeled away by any deterministic predictor.

**AD-2 verified in the open loop**: `z_static_immutable` max drift over a
10-step self-fed rollout is exactly 0.0 for both models — the passthrough
guarantee now has an end-to-end measurement, not just a single-step
tautology test.

## Immutable anchor & critic retirement (v4 B9)

The adversarial `ActionInvarianceCritic` is **retired from training**
(module kept, deprecated): with random-policy data the action is sampled
independently of the observation, so `I(z_imm(obs); action) = 0` for *any*
encoder — the adversarial game reduced to an arms race around noise
(finding C4). Action routing is architectural: actions enter only
`control_net` in `StructuredDynamics`.

In its place, the **immutable anchor** implements the intervention loss the
v3.x plan prescribed and never shipped (finding C3): z_imm must be
invariant within a same-episode window (MSE to the first position) and
variant across episodes (VICReg over per-episode batch means — kills the
collapsed-constant solution that made the passthrough guarantee vacuous).

Honest measurements:

- **Mechanism validated**: on synthetic data with observable episode
  identity, the across/within episode variance ratio exceeds the 10×
  regression floor by orders of magnitude (`tests/test_imm_anchor.py`).
- **This environment carries no signal for it**: a single 2-frame input
  exposes no episode invariants (walls are unobservable; physics requires
  temporal context), so the ratio stays ≈3 with or without the anchor while
  h=1 prediction degrades ~25% (0.001165 control vs 0.001454 anchored; the
  control run also confirms the B9 refactor is clean vs B8's 0.001157).
- **Default: disabled on this env** (`lambda_imm_* = 0.0`), enable ≥ 0.1
  when the input exposes episode invariants — belief-encoder integration
  (B12) or richer environments (B14+). The DoD item "z_imm provably
  informative" lands there, not here; leaving the anchor on today would be
  paying real prediction error for provably absent information.

## Belief encoder v2 — engineered dynamics features (v4 B10)

The belief encoder now consumes **engineered per-step dynamics features**
(`atlas_wm/data/dynamics_features.py`, 27 dims: the oracle's gated
velocity-decay ratio + its running median, excitation, distances, box
acceleration projections onto attractor directions, 1/d² regressors,
aligned actions) instead of raw obs windows, with a **heteroscedastic
head** (μ, logσ; Gaussian NLL) and a **half-window InfoNCE** episode
contrastive. First positive learned physics identification in the
project (supervised head, val, window 40; raw-GRU baseline from B5):

| Parameter | raw-GRU (B5) | belief v2 |
|---|---|---|
| gravity | −0.05 | **+0.43** |
| friction_agent | −0.49 | **+0.22** |
| friction_box | −0.10 | **+0.09** |

What the debugging established (each stage measured):

1. Without the oracle's **gravity gate** on ratio validity, biased "valid"
   steps poison the GRU's mean-style aggregation (medians resist outliers;
   GRUs don't). Gravity flipped positive first because its features are
   action-free.
2. An **action-alignment off-by-one** paired each decay ratio with the
   next step's action; the unit test initially validated the bug because
   its fixture used the same shift. Fixed and verified digit-for-digit
   against the oracle on identical frames.
3. The remaining gap to the oracle's 0.865 is **evidence length, not
   modeling**: friction_agent lives in [0.90, 0.99] (std 0.025) while the
   optimal estimator's error over an 18-step window is std 0.043 — SNR < 1,
   so no learner can score positive there. 38-step windows (window_k 40,
   now the default) bring the error to the signal's scale; the oracle's
   0.865 uses full ~50-step episodes. Ridge probes at window 40 rest on
   only 42 val episodes and are statistically fragile (±0.5 swings) — the
   supervised-head val R² above is the stable metric.

Direct consequence for **B11**: active exploration raises the per-step
information rate (more valid steps: excited, boxes distant, no bounces),
which shortens the window needed for a given belief quality — measure
valid-steps/window under active vs random policies alongside R².

## Active exploration for system identification (v4 B11)

`InfoSeekingPolicy` (`atlas_wm/data/exploration.py`) collects data by
greedily maximizing the B10 information-rate proxies (ASID-style, with the
Fisher objective replaced by the measured proxies): a MEASURE phase runs an
oscillating **rosette dash** in open space (speed ≈ 2 → per-step ratio
error halves; golden-angle axis rotation), alternating with a STIR phase
that drives the nearest box through the gravity band. Available via
`generate_data.py --policy active`.

**Identical belief model + training, only the data collection changes**
(supervised val R², window 40):

| Parameter | random data | active data |
|---|---|---|
| gravity | 0.43 | **0.67** |
| friction_agent | 0.22 | **0.29** |
| friction_box | 0.09 | 0.07 |
| mean | 0.246 | **0.341 (+39%)** |

Episode-level closed-form estimator: active R² 0.956 / MAE 0.0036 vs
random 0.914 / 0.0054.

A finding worth publishing on its own: the first policy version dashed
along a **fixed line**, and when that line crossed an unobservable
obstacle the agent hit it repeatedly — corrupting the ratio median of the
whole episode (episode-level R² collapsed to 0.23 with near-unchanged MAE:
a heavy-tail signature). Random walks don't repeat their own mistakes;
deterministic information-seeking policies do. The rosette rotation turns
such collisions into sparse outliers the median absorbs. Active system
identification needs *anti-fragility to unobservables*, not just
information greed.

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
