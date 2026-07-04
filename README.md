# Atlas.WM: Structured World Model Framework

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-v4.0.0%20released-green.svg)](CHANGELOG.md)

A **small, structured, verifiable, CPU-trainable world model** for physics
identification research — the auditable counterpoint to billion-parameter
video world models. Latent space decomposed into interpretable components
with *architectural* guarantees; every claim in this repository is backed by
a committed script, a reproducible number, and a regression test — including
two public retractions of v3.x findings that did not survive re-verification.

**Author:** Cesar Augusto · **v4.0.0** (2026-07-04, PRs #25–#44)

## What it can prove

| Claim | Number |
|---|---|
| Physics is identifiable from position-only random-policy data | oracle friction R² **0.865** |
| A learned belief identifies physics, given engineered dynamics features | gravity R² **+0.67** |
| Data collection matters as much as the model (active vs random) | **+39%** belief quality |
| The immutable-latent guarantee holds in open-loop rollouts | drift **exactly 0.0** |
| The pipeline generalizes to real MuJoCo contact physics unchanged | **2.3×** gap to linear ceiling |

Full ledger with evidence pointers, retractions, named findings and honest
limits: [`docs/RESULTS.md`](docs/RESULTS.md).

## Architecture

`z_full` (64) = `[ z_static_immutable (8) | z_static_slow (8) | z_dynamic (32) | z_controllable (16) ]`

| Component | Enforcement |
|---|---|
| `z_static_immutable` | hard passthrough in dynamics (AD-2) + optional cross-episode anchor (B9) |
| `z_static_slow` | drift-penalized residual; optionally conditioned on a **causal physics belief** (GRU over engineered dynamics features, B10–B12) |
| `z_dynamic` | residual MLP or **dissipative symplectic (q,p) head** (B13) |
| `z_controllable` | action-conditioned (actions enter *only* here — architectural routing) |

Training: VICReg-regularized self-predictive objective + prediction
grounding + K-step self-fed rollouts (B7–B8). Environments: `CruelGridworld`
(toy, 2D nonlinear gravity) and `MujocoPointMass` (real contact physics),
with random or information-seeking data collection (B11).

## Quick start

```bash
uv sync --extra dev
python scripts/generate_data.py --randomize-physics --process-noise-std 0.05 \
    --episode-reset-prob 0.02 --seed 42
python scripts/split_data.py
python scripts/train.py --config configs/experiments/v3_variable_physics.yaml
python scripts/evaluate.py --checkpoint checkpoints/best_model.safetensors
```

All workflows (belief training, probing, active exploration, belief
conditioning, signing, ONNX export): [`docs/USAGE.md`](docs/USAGE.md).

## Repository layout

```
src/atlas_wm/        installable package (models, environments, data, training, checkpointing, eval, export)
scripts/             pipeline entry points (generate, split, train, evaluate, probe, sign, export, oracle)
configs/             base.yaml + experiments/
tests/               195 tests: unit, physics contracts, canaries, security, regression locks
docs/                RESULTS.md · USAGE.md · MODEL_CARD.md · v4.0-ROADMAP.md · historical postmortems
archive/             quarantined legacy code (do not import)
```

## Documentation

- [`docs/RESULTS.md`](docs/RESULTS.md) — the results ledger: claims, evidence, retractions, limits
- [`docs/USAGE.md`](docs/USAGE.md) — every workflow and config switch
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — model card with full experimental history
- [`docs/v4.0-ROADMAP.md`](docs/v4.0-ROADMAP.md) — the red-team findings and 18-block rebuild plan
- [`CHANGELOG.md`](CHANGELOG.md) — block-by-block ledger

## License

MIT — see [LICENSE](LICENSE).
