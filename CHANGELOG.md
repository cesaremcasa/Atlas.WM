# Changelog

All notable changes to Atlas.WM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] — unreleased

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

[3.0.0]: https://github.com/cesaremcasa/Atlas.WM/releases/tag/v3.0.0
[2.0.0]: https://github.com/cesaremcasa/Atlas.WM/releases/tag/v2.0.0
