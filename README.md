# Atlas.WM: Structured World Model Framework

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-v3.0%20in%20progress-blue.svg)](docs/Atlas-WM-v3-Architecture-Plan.md)

**Status:** v2.0 architecture validated | v3.0 upgrade in progress (Block 1/13)
**Author:** Cesar Augusto

A structured world model that decomposes the latent space into interpretable semantic components and enforces physical constraints through architectural guarantees rather than optimization targets.

---

## Architecture

The latent space is split into four components:

| Component | Description | Enforcement |
|-----------|-------------|-------------|
| `z_static_immutable` | Scene invariants that never change | Hard passthrough (AD-2) |
| `z_static_slow` | Per-episode physics (gravity, friction) | Soft residual + drift penalty |
| `z_dynamic` | Time-varying state (positions, velocities) | Residual update |
| `z_controllable` | Action-sensitive components | Action-conditioned |

See `docs/Atlas-WM-v3-Architecture-Plan.md` for the full architecture and 8 locked design decisions.

---

## Repository Layout

```
Atlas.WM/
├── src/atlas_wm/          # installable package
│   ├── models/            # ContinuousEncoder, StructuredDynamics, heads, losses
│   ├── environments/      # cruel_gridworld.py (single canonical env)
│   ├── data/              # ATLASDataset
│   ├── training/          # train, eval, rollout
│   ├── checkpointing/     # safetensors I/O, signing, env hash
│   └── utils/
├── scripts/               # train.py, generate_data.py, split_data.py
├── configs/
│   ├── base.yaml
│   └── experiments/       # v2_baseline.yaml, v3_hybrid_static.yaml
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── physics/
│   └── security/
├── docs/                  # architecture plan, ADRs, model card
├── archive/               # quarantined v2.0 artifacts (do not import)
└── checkpoints/           # signed .safetensors only
```

---

## Quick Start

```bash
git clone https://github.com/cesaremcasa/Atlas.WM.git
cd Atlas.WM

python -m venv .venv && source .venv/bin/activate

pip install -e .

# Generate training data (50k samples)
python scripts/generate_data.py

# Split into train/val/test
python scripts/split_data.py

# Train
python scripts/train.py
```

---

## v3.0 Upgrade Progress

| Block | Theme | Status |
|-------|-------|--------|
| 1 — Hygiene & Quarantine | Repo cleanup, `src/atlas_wm/` package | ✅ In progress |
| 2 — CI & Lockfile | GitHub Actions, `requirements.lock`, SBOM | ⏳ Pending |
| 3 — Safetensors I/O | Remove `torch.save/load`, safetensors migration | ⏳ Pending |
| 4 — Checkpoint Signing | HMAC-SHA256 manifest | ⏳ Pending |
| 5 — Hybrid Static | `z_static_immutable` + `z_static_slow` | ⏳ Pending |
| 6 — Identifiability | Intervention loss, action-invariance critic | ⏳ Pending |
| 7 — Encoder Tests | Full unit test coverage for `ContinuousEncoder` | ⏳ Pending |
| 8 — Physics Tests | Environment contract tests + chaos tripwire | ⏳ Pending |
| 9 — Determinism Canary | Seeded training reproducibility | ⏳ Pending |
| 10 — Multi-Object | Entity encoder, n_objects ∈ [3, 10] | ⏳ Pending |
| 11 — Partial Observability | Nearest-K wrapper, recurrent belief | ⏳ Pending |
| 12 — Variable Physics | Domain randomization, latent probing | ⏳ Pending |
| 13 — Release v3.0 | Model card, ONNX export, tagged release | ⏳ Pending |

---

## License

MIT — see [LICENSE](LICENSE).
