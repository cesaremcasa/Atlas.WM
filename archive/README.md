# Archive — Quarantined v2.0 Artifacts

**Baseline commit:** `7dd1f37867804bbc436d89021b001833aa0b3001` (v2.0 tag)
**Quarantined by:** Block 1 — Repository Hygiene & Quarantine
**Policy:** Files in this directory are frozen. They are NOT on the Python import path.
Do not import from `archive/` in production code. Do not modify these files.

## Rationale

During the v2.0 → v3.0 upgrade (see `docs/Atlas-WM-v3-Architecture-Plan.md`),
several files were identified as legacy artifacts per AD-1:

> "Legacy artifacts (StructuredEncoder, structured_encoder_small,
> cruel_gridworld_backup, cruel_gridworld_fixed, ATLASDataset, redundant
> training scripts) are quarantined or removed."

They are preserved here for historical reference and postmortem analysis only.

---

## Quarantined Files

### `archive/encoders/`

| File | Original Path | Reason |
|------|--------------|--------|
| `structured_encoder.py` | `src/models/structured_encoder.py` | Legacy encoder; superseded by `ContinuousEncoder`. All callers (train_phase1*, debug_loss) also archived. |
| `structured_encoder_small.py` | `src/models/structured_encoder_small.py` | Unused CNN encoder with lazy-init bug (heads initialized in `forward()` instead of `__init__()`); never called in any training run. |

### `archive/environments/`

| File | Original Path | Reason |
|------|--------------|--------|
| `cruel_gridworld_backup.py` | `src/environments/cruel_gridworld_backup.py` | v1.0 discrete gridworld (pre-gravity, pre-continuous state). Zero references in v2.0 codebase. Preserved for historical context. |
| `cruel_gridworld_fixed.py` | `src/environments/cruel_gridworld_fixed.py` | Undocumented intermediate environment variant. Zero references. Superseded by `cruel_gridworld.py` (v2.0 continuous physics). |

### `archive/scripts/`

| File | Original Path | Reason |
|------|--------------|--------|
| `train_continuous.py` | `scripts/train_continuous.py` | Less robust than `train_continuous_fixed.py`; lacks LR scheduler, NaN detection, and aggressive gradient clipping. |
| `train_phase1.py` | `scripts/train_phase1.py` | Uses legacy `ATLASDataset` + `StructuredEncoder`; superseded by `train_continuous_fixed.py`. |
| `train_phase1_fixed.py` | `scripts/train_phase1_fixed.py` | Config-driven variant of train_phase1; still uses legacy dataset and encoder stack. |
| `debug_loss.py` | `scripts/debug_loss.py` | Debugging utility using legacy `ATLASDataset` + `StructuredEncoder`. No longer valid after encoder migration. |
| `generate_data_v1.py` | `scripts/generate_data.py` | Only generates 2,000 episodes (vs 50k canonical). Superseded by `generate_diverse_data.py`. |
| `generate_data_fixed.py` | `scripts/generate_data_fixed.py` | Intermediate data generation script; lacks diversity reporting. Superseded. |
| `atlas_dataset_legacy.py` | `src/data/atlas_dataset.py` | Legacy `ATLASDataset` class with verbose loading. Superseded by unified `src/atlas_wm/data/dataset.py`. |
