# Changelog

All notable changes to Atlas.WM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Release readiness (v4.0.1 pending)

- Package metadata and import-time version are aligned at `4.0.1`; the
  lock-derived SBOM and reproducible wheel/sdist staging now carry the same
  version.
- CI and scheduled canaries use verified action SHAs, locked uv environments,
  and deterministic checkout gates.

- The release staging path builds with hash-constrained setuptools/wheel
  dependencies, emits SHA256 checksums, and excludes datasets, checkpoints,
  papers, host paths, and environment-specific files from artifacts.

### Added (v4.1 - physics identification on the MuJoCo tier)

- **COAST/PUSH active policy for MuJoCo** (`CoastPushPolicy` + no-op action
  8 in `MujocoPointMass`): random-policy MuJoCo data identifies nothing
  (R² ≈ 0, 332 eps) - friction needs free sliding, mass needs contacts;
  the policy manufactures both events.
- **`atlas_wm.data.mujoco_features`** (16 dims): box-coast deceleration →
  μ·g (Coulomb decelerates linearly, not exponentially) and ALIGNED-contact
  impulse response → mass (head-on contacts only, cos > 0.8 - the raw
  speed-ratio confounded geometry with mass at R² −0.27); running medians
  per the B10 pattern. Belief trainer is env-aware via `obs_scale.json`.
- **Result** (1048 episodes, window 60): learned belief **friction +0.28,
  mass +0.11** - both positive on real contact physics; structural finding:
  gravity alone is unidentifiable here (only μ·g enters box dynamics) and
  is excluded BY STRUCTURE, with the measurement to back it. Measured
  iteration ledger in the part-1/2/3 commit messages, including the
  agent-rides-frictionless-slide-joints discovery.

## [4.0.0] - 2026-07-04

The correction-and-retraction rebuild: every v3.x headline claim was
re-verified, two were retracted, and the training recipe, data pipeline,
evaluation methodology and environment tier were rebuilt with honest,
reproducible measurements (18-block roadmap in `docs/v4.0-ROADMAP.md`,
executed as PRs #25–#43).

### Retracted (v4 B4)

- **The Block-14 identifiability finding.** `friction_agent` IS identifiable
  under random-policy data: a robust median-of-ratios oracle
  (`scripts/oracle_friction_agent.py`, now committed) scores **R² = 0.85,
  MAE = 0.004** over 400 episodes on the corrected environment. The original
  "R² < 0" oracle was never committed and its MSE objective is destroyed by
  bounce outliers. The weak gravity/friction_box ceiling (0.15–0.45 R²)
  measured the box-containment bug fixed in B1. See the retraction notice in
  `docs/MODEL_CARD.md`; full re-baseline lands with B5.

### Security (v4 B17 - fail-closed checkpoint verification)

- **`load_checkpoint(require_signature=True)`** (finding H4): missing
  manifest, unset/malformed/short `ATLAS_SIGNING_KEY`, a file **not
  listed** in the manifest, or any manifest integrity failure now raise
  `SignatureMismatch` instead of warning - an attacker who can replace a
  checkpoint could previously just delete the manifest. Legacy fail-open
  behavior unchanged when the flag is off; 4 new tests. The torch.export/
  dynamo ONNX switch and OpenSSF model-signing are deferred to v4.1
  (exporter swap at release time adds risk; the current exporter is
  parity-tested).

### Changed (v4 B16 - uv lockfile)

- **`uv.lock` is the canonical lockfile** (cross-platform, hashed;
  `uv sync --locked` in CI and `make install`). `requirements.lock` is
  retained for one release as a fallback and for the pip-audit gate.
  Trackio/tyro adoption (roadmap B16 stretch) deferred: swapping config
  and tracking layers mid-release adds risk without changing any result;
  tracked for v4.1.

### Removed (v4 B15 - EntityEncoder retired)

- **`EntityEncoder` moved to `archive/`**: it was dead code - never wired
  to any environment ("n_objects ∈ [3, 10]" was never exercised) and
  permutation-*invariant* (mean-pool), not equivariant as documented.
  Real multi-object support belongs to the v5 line on the MuJoCo tier
  (slot/particle direction per the roadmap), not to a vestigial module
  whose claims the code never backed.

### Added (v4 B14 - MuJoCo environment tier)

- **`MujocoPointMass`** (`--env mujoco`): actuated ball + two passive boxes,
  real MuJoCo contact/friction, per-episode randomized friction/mass/gravity
  in `info`, same interface contract as CruelGridworld (8 discrete force
  directions, 6-D obs, seeded) - the whole v4 pipeline runs unchanged.
- **Per-dataset observation scale** (`obs_scale.json` written by
  generate_data, copied by split_data, resolved by the datasets): MuJoCo
  observations live in [−1.2, 1.2], not [0, 20].
- First MuJoCo baseline (VICReg + grounding + rollout K=4): h=1 MSE
  0.001089 vs linear ceiling 0.000465 - a **2.3× gap vs 3.6× on the
  gridworld**, consistent with the process-noise analysis; AD-2 passthrough
  drift exactly 0.0 over 8-step rollouts.

### Added (v4 B13 - dissipative symplectic dynamics head)

- **`model.dynamics_head: hamiltonian`**: z_dynamic splits into (q, p);
  dissipative symplectic-Euler step with learned force field and per-dim
  damping conditioned on z_static_slow (friction analog; contraction
  tested). Honest A/B verdict: **relative OOD robustness is real**
  (gravity 8–12, outside train range: +12% degradation vs the residual
  head's +69% at h=1) **but absolute error is worse everywhere** (3.5× in
  distribution). Ships as a non-default option; echoes the literature that
  gains come from second-order structure, not symplecticity per se -
  revisit with tuned capacity on MuJoCo (B14+).

### Added (v4 B12 - belief conditioning, RMA phase 2)

- **`scripts/precompute_belief.py`** runs the trained belief GRU causally
  per episode (`PhysicsBeliefEncoder.forward_sequence`, prefix-property
  tested) and stores per-row beliefs; `training.use_belief` substitutes
  them for the encoder's `z_static_slow` in both the start latent and the
  rollout targets - the dynamics' slow residual learns to track the
  belief. The RMA phase-2 integration the v3.x design promised.
- A/B on active data (identical seeds): h=1 next-frame MSE 0.001080 →
  **0.001017 (−5.8%)**; h=4 0.011232 → 0.011059 (−1.5%). Honest caveat:
  the process-noise random walk dominates long-horizon error on this env,
  capping what known physics can buy; the conditioning's value should
  grow in lower-noise / richer settings (B14+).

### Added (v4 B11 - active exploration for system identification)

- **`InfoSeekingPolicy`** + `generate_data.py --policy active`: rosette-dash
  MEASURE phase (2× per-step ratio precision) alternating with box-STIR
  (gravity signal). Same belief model/training, active vs random data:
  mean val R² **0.246 → 0.341 (+39%)**; gravity 0.43 → **0.67**;
  friction_agent 0.22 → 0.29. Episode-level estimator: R² 0.956 / MAE
  0.0036 (active) vs 0.914 / 0.0054 (random).
- Key negative finding documented: a fixed-line dash repeatedly struck
  unobservable obstacles and poisoned episode medians (heavy-tail R²
  collapse) - deterministic info-seeking needs anti-fragility to
  unobservables; the golden-angle rosette provides it. Regression test
  locks the information advantage (mean estimator error < 75% of random).

### Added (v4 B10 - belief encoder v2)

- **Engineered dynamics features** (`atlas_wm.data.dynamics_features`, 27
  dims: gated decay ratio + running median - the oracle's sufficient
  statistic - excitation, distances, box-acceleration projections, 1/d²,
  aligned actions) replace raw obs windows as the belief GRU's input;
  **heteroscedastic physics head** (μ, logσ, Gaussian NLL) and
  **half-window InfoNCE** episode contrastive added; `window_k` default 40
  (evidence-length analysis in MODEL_CARD).
- **First positive learned physics identification**: supervised val R²
  gravity **+0.43**, friction_agent **+0.22**, friction_box **+0.09**
  (raw-GRU baseline: −0.05 / −0.49 / −0.10). Remaining gap to the 0.865
  oracle is evidence length (SNR < 1 at short windows), not modeling -
  the direct setup for B11 active exploration.

### Changed (v4 B9 - immutable anchor, critic retired)

- **Adversarial critic retired from training** (finding C4: with
  random-policy data, I(z_imm; action) = 0 for any encoder - the game was
  an arms race around noise). Module kept with a deprecation note; action
  routing is architectural (actions only enter `control_net`).
- **Immutable anchor** - the intervention loss v3.x promised and never
  shipped (C3): within-episode invariance + cross-episode VICReg on z_imm
  episode means. Mechanism regression-tested with a 10× variance-floor test
  on identity-observable synthetic data. On CruelGridworld a single input
  carries no episode identity (ratio ≈3 with or without; ~25% h=1 cost), so
  the anchor ships **disabled by default** here - to be enabled with B12
  belief integration / B14+ richer envs. A control run confirms the
  refactor itself is clean (h=1: 0.001165 vs B8's 0.001157).

### Added (v4 B8 - multi-step rollout training + real evaluation)

- **`training.rollout_k`** (default 4): K-step self-fed rollout training -
  predicted latents feed the next dynamics step with per-step latent
  supervision + grounding; windows come from `EpisodeATLASDataset`.
  Validation tracks h=1 and h=K separately.
- **`scripts/evaluate.py` is now a real evaluator** (finding M5): open-loop
  obs-space MSE by horizon and the AD-2 immutable-passthrough drift check
  (measured: exactly 0.0 over 10-step rollouts).
- Result: the classic horizon trade-off - one-step model wins at h=1
  (0.000993 vs 0.001157), rollout-trained wins from h≥3 (h=10: 0.0357 vs
  0.0389). Process noise sets an irreducible random-walk floor at long
  horizons.

### Added (v4 B7 - stable objective + prediction grounding)

- **`training.objective`**: `vicreg` (default; variance hinge + covariance
  penalty on the encoder output - the tensor that can collapse) or `ema`
  (EMA-target encoder, BYOL/TD-MPC2 lineage) replace the v3.x recipe;
  `legacy` retained for comparison. `lambda_latent_l2` and the mis-placed
  variance penalty are retired.
- **Prediction grounding** (`lambda_next_recon`): the loss now optimizes
  `decoder(dynamics(z))` against the actual next observation - the
  inference path was previously never trained directly.
- **Objective-agnostic model selection**: checkpoints and early stopping
  now use observation-space next-frame MSE.
- Result on the noisy re-baseline data (2 frames): 0.001750 → **0.000996**
  (43% ↓); frame stacking now helps (it hurt under the legacy recipe); the
  remaining 3.6× gap to the linear ceiling is tracked for B8+.

### Added (v4 B6 - frame stacking)

- **`frame_stack: 2` is the new default model input**: `ATLASDataset`
  concatenates the previous same-episode frame (velocity becomes observable,
  finding M2); `train.py` plumbs it through encoder/decoder widths and
  checkpoint metadata; `probe_physics.py` feeds the encoder the same view it
  was trained on. Honest acceptance result: a linear ridge improves 3.1×
  with stacked input (0.000856 → 0.000274 next-frame MSE) while the trained
  world model under the current v3-era loss gets *worse* (0.000921 →
  0.001750) - the recipe, not the information, is the bottleneck; the
  comparison is re-run after B7 replaces the objective.

### Changed (v4 B5 - re-baseline)

- **New identifiability baseline** on the corrected environment with
  episode-grouped probe splits (see the re-baseline section in
  `docs/MODEL_CARD.md`): the closed-form oracle recovers `friction_agent`
  with **R² = 0.865** (MAE 0.006) from the same noisy (σ = 0.05) dataset
  where the raw-sequence GRU belief encoder scores **negative R² on all
  three parameters** - the bottleneck is the training recipe, not the data.
  `friction_agent` is back in the target set everywhere (`PHYSICS_KEYS`,
  probe defaults, configs).
- Probe splits are now grouped by episode (`latent_probe._split_indices`):
  sequential row splits leaked physics labels across overlapping windows and
  inflated v3.x probe R².

### Fixed (v4 phase 0)

- **Environment (B1)**: boxes now collide with walls and obstacles like the
  agent; previously they exited the grid (range [−62, +77] on [0, 20]),
  violating the observation space and erasing the gravity signal after a few
  steps. 100-seed containment regression test added.
- **Data pipeline (B2)**: observation normalization moved in memory into the
  datasets (split files on disk are never modified; legacy in-place-normalized
  dirs are rejected); the `.split` sentinel stores a SHA-256 of the raw arrays
  so regenerated data triggers an automatic re-split; episode-windowing
  off-by-one fixed (`cumsum[i-k]` → `cumsum[i-k+1]`).
- **Reproducibility (B3)**: both trainers seed python/numpy/torch and their
  DataLoaders (`--seed` / `training.seed`); a real training canary asserts
  bit-identical loss traces; `d_static_immutable`/`d_static_slow` are actually
  plumbed into the models and inferred at ONNX export (previously the split
  silently stayed at `d_static // 2`); critic weights are checkpointed.
- **CI (B4)**: `ci.yml` re-enabled; `chaos-physics.yml` now runs a real,
  committed `scripts/chaos_physics.py` (containment, finiteness, determinism,
  dissipation over randomized episodes); `train-canary.yml` points at tests
  that exist; `make security` no longer flags the migration script.

### Fixed

- **Main-encoder divergence (root cause)**: added a latent-magnitude penalty
  `lambda_latent_l2` (default 0.01) on `‖z_full‖`. The self-predictive `pred_loss`
  (target = the encoder's own detached output over `next_obs`) has a degenerate
  direction: the encoder can inflate its representation scale without bound while
  the dynamics tracks it, so after a few stable epochs the loss explodes ~10×/epoch.
  Reconstruction alone does not anchor it at the full 50k scale. The L2 penalty
  removes the runaway direction - validated end-to-end on `scripts/train.py`
  (stable, monotonic convergence). NOTE: an earlier hypothesis blamed the
  adversarial loss; that was disproven - disabling the adversarial term entirely
  still diverged identically. The divergence merely *coincided* with the warmup
  boundary.
- **Adversarial loss hardening** (defense-in-depth, not the divergence cause):
  bounded `encoder_adversarial_loss`; its `-mse(pred, action)` objective was
  unbounded below. The fooling reward is capped at 0.5.
- **Data pipeline**: `split_data.py` now splits by shuffled episode ID (not
  transition index) to prevent physics distribution shift between train/val/test.
  (Its `.normalized`-sentinel handling was later superseded by the B2 in-memory
  normalization above - the sentinel is now a rejected legacy marker.)

### Changed

- ~~**Physics identification scope** (Block 14)~~ **[SUPERSEDED - see
  *Retracted (v4 B4)* above.]** This entry claimed `friction_agent` is not
  identifiable and rescoped the targets to `{gravity, friction_box}`; the
  claim was retracted and the full target set restored in B5. Retained only
  for the still-valid data recipe: the belief pipeline generates ~50-step
  episodes (`--episode-reset-prob 0.02`) with `window_k=20`.

## [3.1.0] - 2026-06-24

### Added

- **PhysicsBeliefEncoder** (Block 14): GRU-based encoder over K consecutive
  same-episode observations → `z_static_slow`. Follows the RMA/VariBAD pattern:
  accumulates temporal evidence to identify episode-level physical constants
  (gravity, friction) that are invisible from a single position-only snapshot.
- **PhysicsHead**: supervised auxiliary linear head `z_static_slow → physics_hat`
  for training the belief encoder with ground-truth physics labels.
- **EpisodeATLASDataset**: windowed dataset returning K-step same-episode windows
  using a vectorized cumsum boundary-detection algorithm; requires
  `episode_ids.npy` produced by `generate_data.py --randomize-physics`.
- **`scripts/train_physics_belief.py`**: training script for the GRU belief
  encoder; saves checkpoint as `.safetensors` (AD-4) with full metadata.
- **`scripts/probe_physics.py`** extended: `--belief-checkpoint` flag runs the
  PhysicsBeliefEncoder probe alongside the single-step baseline.

### Fixed

- `export_onnx.py`: infer `input_dim`, `d_static`, `d_dynamic`, `d_controllable`,
  and `action_dim` from state dict weight shapes - no longer hardcoded; prevents
  shape mismatch on non-default checkpoints.
- `probe_physics.py`: `gru_input_dim` fallback now computes `obs_dim + action_dim`
  when the key is absent from checkpoint metadata (previously defaulted to
  `obs_dim` only, causing GRU shape errors for obs+action checkpoints).
- mypy `no-any-return` errors in `physics_belief.py` and `episode_dataset.py`.

---

## [3.0.0] - 2026-06-17

The v3.0 line rebuilds Atlas.WM as an installable, reproducible, security-hardened
package around the **hybrid static latent decomposition** (AD-2) and a set of
locked architectural decisions (AD-1 … AD-8). Delivered as 13 sequential blocks.

### Added

- **Hybrid static decomposition** (Block 5, AD-2): the latent splits into
  `z_static_immutable` (hard architectural passthrough - bit-identical across
  time), `z_static_slow` (soft residual with a drift penalty), `z_dynamic`
  (autonomous evolution), and `z_controllable` (action-conditioned).
- **Identifiability** (Block 6, AD-3): an `ActionInvarianceCritic` trained
  adversarially to keep action information out of `z_static_immutable`, plus an
  intervention loss and content-addressed environment hashing (AD-6).
- **EntityEncoder** (Block 10): permutation-equivariant, mean-pooled encoder
  supporting a variable number of objects.
- **Partial observability** (Block 11): nearest-K object masking wrapper.
- **Variable physics & process noise** (Block 12): per-episode domain
  randomization of gravity/friction and seeded Gaussian process noise in
  `CruelGridworld`, with a closed-form **ridge latent probe**
  (`atlas_wm.eval.latent_probe`) that validates physics is decodable from
  `z_static_slow` and not from the immutable passthrough.
- **ONNX export** (Block 13): `atlas_wm.export.onnx_export` and
  `scripts/export_onnx.py` emit composable `encoder.onnx` (`obs → z_full`) and
  `dynamics.onnx` (`z_full, action → z_full_next`) graphs with dynamic batch
  axes. Available via the optional `export` extra.
- **Model card** (`docs/MODEL_CARD.md`) and this changelog (Block 13).
- Test suite grown to 100+ tests across unit, integration, physics contract,
  security, latent-probe, and ONNX-parity categories.

### Changed

- **Packaging** (Block 1): code restructured into the installable
  `src/atlas_wm/` package; legacy v2.0 artifacts quarantined under `archive/`.
- **Reproducible CI** (Blocks 2 & 13): GitHub Actions installs the fully
  hash-pinned toolchain from `requirements.lock` (AD-8) instead of unpinned
  latest tools, with pip caching; the Test job is a hard gate.
- **Checkpoint I/O** (Block 3, AD-4): all reads/writes go through
  `safetensors`; pickle-based formats are rejected.

### Security

- **HMAC-SHA256 checkpoint signing** (Block 4): `manifest.sig` integrity
  verification keyed by `ATLAS_SIGNING_KEY`.
- Pickle / `torch.save` / `torch.load` forbidden in production code paths,
  enforced by a CI tripwire (the migration script is the sole exception).
- SBOM (`sbom.json`, CycloneDX) and `pip-audit` over the lockfile in CI.

### Determinism

- **Determinism canary & rollout-drift** integration tests (Block 9, AD-7);
  physics contract tests with a chaos tripwire (Block 8). Default environment
  behavior is byte-identical across the Block 12 changes (variable physics and
  noise consume no extra RNG draws unless explicitly enabled).

### Notes

- The optional `export` dependencies (`onnx`, `onnxruntime`) are intentionally
  **not** in `requirements.lock`; the ONNX parity tests skip when they are
  absent, keeping the pinned CI toolchain lean. Regenerate the lock with
  `uv pip compile pyproject.toml --extra dev --extra export --generate-hashes`
  if you want them pinned.

## [2.0.0] - 2025

Continuous-physics world model with structured latents on `CruelGridworld`.
See `docs/v2.0-COMPLETION-REPORT.md` and `docs/v2.0-TECHNICAL-POSTMORTEM.md`.

[4.0.0]: https://github.com/cesaremcasa/Atlas.WM/releases/tag/v4.0.0
[3.1.0]: https://github.com/cesaremcasa/Atlas.WM/releases/tag/v3.1.0
[3.0.0]: https://github.com/cesaremcasa/Atlas.WM/releases/tag/v3.0.0
[2.0.0]: https://github.com/cesaremcasa/Atlas.WM/releases/tag/v2.0.0
