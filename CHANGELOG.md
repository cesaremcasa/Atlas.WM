# Changelog

All notable changes to Atlas.WM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Retracted (v4 B4)

- **The Block-14 identifiability finding.** `friction_agent` IS identifiable
  under random-policy data: a robust median-of-ratios oracle
  (`scripts/oracle_friction_agent.py`, now committed) scores **R² = 0.85,
  MAE = 0.004** over 400 episodes on the corrected environment. The original
  "R² < 0" oracle was never committed and its MSE objective is destroyed by
  bounce outliers. The weak gravity/friction_box ceiling (0.15–0.45 R²)
  measured the box-containment bug fixed in B1. See the retraction notice in
  `docs/MODEL_CARD.md`; full re-baseline lands with B5.

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
  removes the runaway direction — validated end-to-end on `scripts/train.py`
  (stable, monotonic convergence). NOTE: an earlier hypothesis blamed the
  adversarial loss; that was disproven — disabling the adversarial term entirely
  still diverged identically. The divergence merely *coincided* with the warmup
  boundary.
- **Adversarial loss hardening** (defense-in-depth, not the divergence cause):
  bounded `encoder_adversarial_loss`; its `-mse(pred, action)` objective was
  unbounded below. The fooling reward is capped at 0.5.
- **Data pipeline**: `split_data.py` now splits by shuffled episode ID (not
  transition index) to prevent physics distribution shift between train/val/test,
  and clears the `.normalized` sentinel on re-split so `train.py` re-normalizes
  freshly split data.

### Changed

- **Physics identification scope** (Block 14): the PhysicsBeliefEncoder and probe
  now target the recoverable subset `{gravity, friction_box}` only. `friction_agent`
  is excluded — validated as not identifiable under the current env + random-policy
  regime (oracle probe on privileged features scores R² < 0). The belief pipeline
  now generates ~50-step episodes (`--episode-reset-prob 0.02`) with `window_k=20`
  to give the GRU enough evidence. See the new *Physics identifiability* section in
  `docs/MODEL_CARD.md` for the empirical ceiling and rationale.

## [3.1.0] — 2026-06-24

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
  and `action_dim` from state dict weight shapes — no longer hardcoded; prevents
  shape mismatch on non-default checkpoints.
- `probe_physics.py`: `gru_input_dim` fallback now computes `obs_dim + action_dim`
  when the key is absent from checkpoint metadata (previously defaulted to
  `obs_dim` only, causing GRU shape errors for obs+action checkpoints).
- mypy `no-any-return` errors in `physics_belief.py` and `episode_dataset.py`.

---

## [3.0.0] — 2026-06-17

The v3.0 line rebuilds Atlas.WM as an installable, reproducible, security-hardened
package around the **hybrid static latent decomposition** (AD-2) and a set of
locked architectural decisions (AD-1 … AD-8). Delivered as 13 sequential blocks.

### Added

- **Hybrid static decomposition** (Block 5, AD-2): the latent splits into
  `z_static_immutable` (hard architectural passthrough — bit-identical across
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

## [2.0.0] — 2025

Continuous-physics world model with structured latents on `CruelGridworld`.
See `docs/v2.0-COMPLETION-REPORT.md` and `docs/v2.0-TECHNICAL-POSTMORTEM.md`.

[3.1.0]: https://github.com/cesaremcasa/Atlas.WM/releases/tag/v3.1.0
[3.0.0]: https://github.com/cesaremcasa/Atlas.WM/releases/tag/v3.0.0
[2.0.0]: https://github.com/cesaremcasa/Atlas.WM/releases/tag/v2.0.0
