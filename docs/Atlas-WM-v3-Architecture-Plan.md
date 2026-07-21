---
project: Atlas.WM
document_type: Architecture & Execution Plan
target_version: v3.0
baseline_version: v2.0 (commit 7dd1f37)
status: Approved for execution
audience: Claude Code (autonomous execution) + human reviewer
language_policy: All code, comments, commits, tests, logs, and reports in English. Documentation in English.
execution_model: Sequential block execution. Each block must reach DoD before next block starts. One PR per block, squash-merged to main.
---

# Atlas.WM - v3.0 Architecture & Execution Plan

## How to use this document

This document is consumed sequentially by Claude Code. Each block is a self-contained unit of work with:

- **Goal** - what the block achieves.
- **Scope** - what is in / out.
- **Architectural decisions** - what is fixed.
- **Tasks** - ordered, atomic.
- **Definition of Done (DoD)** - verifiable acceptance criteria.
- **Smoke test** - commands that must pass on the runner.
- **Commit & merge protocol** - branch name, commit messages, PR title.

**Execution rule:** do not start block N+1 until block N's DoD is green and PR is merged to main. If DoD fails, fix in the same branch - do not move on.

**Output language rule:** every artifact Claude Code writes (code, comments, logs, commit messages, test names, /tmp/ reports) is English. No exceptions.

---

## Locked Architectural Decisions

These decisions are not open for re-litigation during execution. Any deviation requires explicit human approval.

**AD-1 - v2.0 is the foundation, not a throwaway prototype**
ContinuousEncoder, cruel_gridworld.py, and the 50k transition dataset are kept. v3.0 extends v2.0; it does not rewrite it. Legacy artifacts (StructuredEncoder, structured_encoder_small, cruel_gridworld_backup, cruel_gridworld_fixed, ATLASDataset, redundant training scripts) are quarantined or removed.

**AD-2 - Hybrid static decomposition**
The static component is split into two sub-spaces:
- `z_static_immutable` - hard structural passthrough. The dynamics module returns this slice byte-identical to its input. Bit-equality is an architectural invariant, not a learned property.
- `z_static_slow` - soft-constrained residual update. Allowed to change, but with explicit drift penalty in the loss and a tracked rollout-drift metric. This is where v3.0 "variable physics" (gravity, friction) live.

This is structurally enforced via output decomposition (KKT-hardnet style for the immutable slice; regularized residual for the slow slice).

**AD-3 - Identifiability is a Phase 1 concern, not a Phase 5 concern**
A passthrough constraint without an identifiability objective allows the encoder to collapse `z_static_immutable` to a degenerate (e.g. zero) representation. Therefore, from the moment passthrough is introduced, an intervention loss is co-trained:
- Null action: `z_dynamic` and `z_controllable` must not move beyond noise floor when `a == no-op`.
- Non-null action: only `z_controllable` is permitted to change significantly between steps under direct action.
- Action-invariance: `z_static_immutable` must satisfy `MI(z_static_immutable; a) ≈ 0` (estimated via a critic).

**AD-4 - Checkpoint format: safetensors only**
`torch.save` and `torch.load` are removed from the codebase. All checkpoint I/O uses `safetensors.torch.save_file` / `load_file`. Metadata (git SHA, env hash, config hash, train timestamp, model class, ATLAS schema version) is embedded in the safetensors header. Every published checkpoint ships with an HMAC-SHA256 signature manifest. CI rejects PRs that introduce `torch.load` or `pickle` in production code paths.

**AD-5 - Single source of truth for hyperparameters**
One training entrypoint: `scripts/train.py`. Hyperparameters live in `configs/experiments/*.yaml`. Hardcoded values in scripts are bugs. A CI check parses the active config and asserts that the training run uses the declared values.

**AD-6 - Environment is content-addressed**
Every checkpoint records a hash of the environment's physical parameters (gravity, friction, action_space size, observation shape, object count). Loading a checkpoint against a mismatched env hash fails by default. Override requires explicit `--allow-env-mismatch` flag and emits a loud warning.

**AD-7 - Determinism is a tested invariant**
A canary test runs end-to-end training for N=100 steps under a fixed seed and asserts that final loss matches a recorded reference to within tight tolerance. Non-determinism in critical paths (data ordering, env reset, weight init) is treated as a bug, not a quirk.

**AD-8 - Security posture**
- Production code paths (train, eval, inference) never import pickle directly.
- Dependencies are pinned with hashes in `requirements.lock`.
- SBOM (CycloneDX) is generated in CI and committed.
- `bandit` and `pip-audit` run in CI; high-severity findings block merge.
- A `SECURITY.md` documents the threat model and supported reporting channels.

---

## Repository Layout (target state at end of execution)

```
Atlas.WM/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # lint, type, test, security on PR
│       ├── train-canary.yml          # weekly determinism check
│       └── chaos-physics.yml         # weekly physics tripwire
├── configs/
│   ├── base.yaml                     # defaults
│   └── experiments/
│       ├── v2_baseline.yaml          # frozen reference
│       ├── v3_hybrid_static.yaml     # current active
│       └── v3_multi_object.yaml      # Block 10
├── docs/
│   ├── ARCHITECTURE.md               # this document, mirrored
│   ├── SECURITY.md                   # threat model
│   ├── CHANGELOG.md
│   ├── MODEL_CARD.md
│   └── adr/                          # architecture decision records
│       └── 001-hybrid-static-decomposition.md
├── src/atlas_wm/
│   ├── __init__.py
│   ├── models/
│   │   ├── continuous_encoder.py
│   │   ├── structured_dynamics.py    # hybrid passthrough + residual
│   │   ├── heads.py                  # static_immutable, static_slow, dynamic, controllable
│   │   └── losses.py                 # reconstruction + intervention + drift
│   ├── environments/
│   │   └── cruel_gridworld.py        # single canonical env
│   ├── data/
│   │   ├── dataset.py                # single ATLASDataset
│   │   └── generation.py
│   ├── training/
│   │   ├── train.py                  # single entrypoint
│   │   ├── eval.py
│   │   └── rollout.py                # multi-step evaluation
│   ├── checkpointing/
│   │   ├── io.py                     # safetensors only
│   │   ├── signing.py                # HMAC manifest
│   │   └── env_hash.py
│   └── utils/
│       ├── seeding.py
│       └── logging.py
├── scripts/
│   ├── train.py                      # thin wrapper around training.train
│   ├── generate_data.py
│   ├── evaluate.py
│   └── verify_checkpoint.py
├── tests/
│   ├── unit/
│   │   ├── test_encoder.py
│   │   ├── test_dynamics_invariants.py
│   │   ├── test_losses.py
│   │   ├── test_checkpoint_io.py
│   │   └── test_env_hash.py
│   ├── integration/
│   │   ├── test_train_loop.py
│   │   ├── test_rollout_drift.py
│   │   └── test_determinism_canary.py
│   ├── physics/
│   │   ├── test_gravity.py
│   │   ├── test_collisions.py
│   │   ├── test_bounds.py
│   │   └── test_action_space_contract.py
│   └── security/
│       ├── test_no_pickle_in_prod.py
│       └── test_load_rejects_unsigned.py
├── archive/                          # frozen, not in import path
│   ├── README.md
│   ├── encoders/
│   │   ├── structured_encoder.py
│   │   └── structured_encoder_small.py
│   ├── environments/
│   │   ├── cruel_gridworld_fixed.py
│   │   └── cruel_gridworld_backup.py
│   └── scripts/
│       └── (the 7 redundant train_* scripts)
├── checkpoints/
│   ├── README.md                     # signing policy
│   ├── manifest.sig                  # HMAC-SHA256 of all .safetensors
│   └── *.safetensors                 # only signed files allowed
├── pyproject.toml                    # replaces requirements.txt
├── requirements.lock                 # hashed pins
├── sbom.json
├── README.md
├── SECURITY.md
├── CONTRIBUTING.md
├── LICENSE
└── .gitignore
```

---

## Execution Blocks

Total: 13 blocks organized in 5 phases. Each block is a single PR.

| Phase | Blocks | Theme |
|-------|--------|-------|
| 0 - Hygiene | 1, 2 | Repo layout, CI skeleton |
| 1 - Security & supply chain | 3, 4 | Safetensors, signing, lockfile, SBOM |
| 2 - Foundation correctness | 5, 6, 7 | Hybrid decomposition, identifiability, encoder migration |
| 3 - Test coverage | 8, 9 | Physics tests, determinism canary, rollout drift |
| 4 - v3.0 features | 10, 11, 12 | Multi-object, partial obs, variable physics |
| 5 - Production readiness | 13 | Model card, ONNX export, release |

---

## BLOCK 1 - Repository Hygiene & Quarantine

**Goal:** Clean up the v2.0 mess so that future blocks have a single source of truth for code, configs, and scripts. No semantic changes to model or training - pure structural cleanup.

**Scope In:**
- Move legacy encoders, environments, datasets, and redundant scripts to `archive/`.
- Create the target directory layout (empty placeholders OK).
- Establish `pyproject.toml`, package `src/atlas_wm/` properly importable.
- Add `.gitignore` rules for `checkpoints/*.pt`, `data/processed/`, `wandb/`, `.venv/`.

**Scope Out:**
- Any model or training logic change.
- Any new tests beyond an import-smoke test.

**Tasks:**
1. Create `archive/` with `README.md` explaining quarantine rationale and original commit SHA.
2. Move `src/models/structured_encoder.py` → `archive/encoders/`.
3. Move `src/models/structured_encoder_small.py` → `archive/encoders/`.
4. Move `src/environments/cruel_gridworld_fixed.py` → `archive/environments/`.
5. Move `src/environments/cruel_gridworld_backup.py` → `archive/environments/`.
6. Move `src/data/atlas_dataset.py` content into a single unified `src/atlas_wm/data/dataset.py` (start by aliasing whichever was used by best training run; ContinuousDataset is the canonical).
7. Move `scripts/train_phase1*.py`, `scripts/train_continuous*.py` (except the best one, renamed to `scripts/train.py`) → `archive/scripts/`.
8. Reorganize source under `src/atlas_wm/` package layout per "Repository Layout" above.
9. Add `pyproject.toml` declaring the package. Pin Python >=3.11.
10. Update all imports in remaining code to use `from atlas_wm.X import Y`.
11. Add `.gitignore` entries: `*.pt`, `checkpoints/*.safetensors` (signed files only via manifest), `data/processed/`, `wandb/`, `.venv/`, `__pycache__/`, `.pytest_cache/`.
12. Update `README.md` with the new layout.

**Definition of Done:**
- [ ] `pip install -e .` succeeds.
- [ ] `python -c "import atlas_wm; from atlas_wm.models.continuous_encoder import ContinuousEncoder; from atlas_wm.environments.cruel_gridworld import CruelGridworld"` returns 0.
- [ ] `archive/` exists with a `README.md` listing every quarantined file and its rationale.
- [ ] No file outside `archive/` imports anything from `archive/`.
- [ ] `git ls-files src/atlas_wm/environments/` lists exactly one environment: `cruel_gridworld.py`.
- [ ] `git ls-files scripts/ | grep '^scripts/train' | wc -l` returns 1.
- [ ] `git ls-files src/atlas_wm/data/` lists exactly one dataset module.

**Smoke Test:**
```bash
set -e
pip install -e . > /tmp/install.log 2>&1
python -c "
import atlas_wm
from atlas_wm.models.continuous_encoder import ContinuousEncoder
from atlas_wm.environments.cruel_gridworld import CruelGridworld
from atlas_wm.data.dataset import ATLASDataset
print('IMPORTS OK')
" > /tmp/smoke_block1.log
test -f archive/README.md
test ! -f src/models/structured_encoder.py
test ! -f src/environments/cruel_gridworld_backup.py
echo "BLOCK 1 SMOKE OK"
```

**Branch / Commit / Merge:**
- Branch: `block/01-hygiene-quarantine`
- Commit prefix: `chore(block-01):`
- PR title: `Block 1 - Repository hygiene and legacy quarantine`
- Merge: squash to main.

---

## BLOCK 2 - CI Skeleton & Lockfile

**Goal:** Stand up GitHub Actions so that every PR from Block 3 onward runs lint, type, test, and security checks. Replace `requirements.txt` with a hashed lockfile and pin Python and torch versions strictly.

**Scope In:**
- `.github/workflows/ci.yml` running on PR and push to main.
- `pyproject.toml` with dev dependencies (pytest, pytest-cov, ruff, mypy, bandit, pip-audit).
- `requirements.lock` generated via `uv pip compile --generate-hashes`.
- SBOM generation via `cyclonedx-py` committed at `sbom.json`.

**Scope Out:**
- Any new tests beyond what exists today (those come in later blocks). CI runs whatever tests already exist; expect some to fail - that's tracked.

**Tasks:**
1. Install uv in CI runner.
2. Generate `requirements.lock` with hashes from `pyproject.toml`.
3. Write `.github/workflows/ci.yml` with jobs: lint, type, test, security.
   - lint: `ruff check . + ruff format --check .`
   - type: `mypy src/atlas_wm`
   - test: `pytest -q --cov=atlas_wm --cov-report=xml`
   - security: `bandit -r src/ + pip-audit -r requirements.lock`
4. Generate `sbom.json` via `cyclonedx-py environment -o sbom.json` and commit.
5. Add `.pre-commit-config.yaml` for local developer parity (ruff, mypy, bandit).
6. Document `make ci-local` target that runs the same checks locally.

**Definition of Done:**
- [ ] CI runs green on the Block 2 PR for lint and security jobs.
- [ ] test job runs (may have failures from pre-existing v2.0 bugs - those are catalogued, not silenced).
- [ ] `pip-audit -r requirements.lock` reports zero CRITICAL or HIGH findings.
- [ ] `sbom.json` is present and parses as valid CycloneDX.
- [ ] `pre-commit run --all-files` exits 0 on a clean clone.
- [ ] `requirements.lock` exists and contains hash lines (`--hash=sha256:...`).

**Smoke Test:**
```bash
set -e
uv pip sync requirements.lock > /tmp/sync.log 2>&1
ruff check src/ tests/ > /tmp/lint.log 2>&1
bandit -r src/ -ll > /tmp/bandit.log 2>&1
pip-audit -r requirements.lock --severity high > /tmp/audit.log 2>&1
python -c "import json; json.load(open('sbom.json'))" && echo "SBOM JSON valid"
echo "BLOCK 2 SMOKE OK"
```

**Branch / Commit / Merge:**
- Branch: `block/02-ci-lockfile`
- Commit prefix: `ci(block-02):`
- PR title: `Block 2 - CI skeleton, lockfile, SBOM`
- Merge: squash to main.

---

## BLOCK 3 - Safetensors Migration & Checkpoint I/O

**Goal:** Remove every `torch.save`/`torch.load` call from production code paths. Replace with safetensors. Embed metadata in the safetensors header.

**Scope In:**
- New module `src/atlas_wm/checkpointing/io.py` with `save_checkpoint()` and `load_checkpoint()`.
- Migration of `scripts/train.py` and `scripts/evaluate.py` to use the new I/O.
- A one-shot migration utility `scripts/migrate_pt_to_safetensors.py`.
- CI rule: PRs introducing `torch.load(` or `pickle.load(` outside `archive/` fail.

**Scope Out:**
- Signing (Block 4).
- Env hash content - placeholder string for now (Block 6 fills it).

**Tasks:**
1. Add `safetensors` to `pyproject.toml` dependencies.
2. Implement `src/atlas_wm/checkpointing/io.py`.
3. Migrate `scripts/train.py` to call `save_checkpoint(...)` and remove `torch.save`.
4. Migrate `scripts/evaluate.py` to call `load_checkpoint(...)` and remove `torch.load`.
5. Write `scripts/migrate_pt_to_safetensors.py`.
6. Add CI grep rule in `.github/workflows/ci.yml`.
7. Move `checkpoints/best_model.pt` to `archive/` after migration.

**Branch / Commit / Merge:**
- Branch: `block/03-safetensors-io`
- Commit prefix: `feat(block-03):`
- PR title: `Block 3 - Safetensors checkpoint I/O, remove pickle from prod paths`

---

## BLOCK 4 - Checkpoint Signing & Verification

**Goal:** Every checkpoint shipped in `checkpoints/` is signed with HMAC-SHA256.

**Branch / Commit / Merge:**
- Branch: `block/04-checkpoint-signing`
- Commit prefix: `feat(block-04):`
- PR title: `Block 4 - HMAC checkpoint signing and verification`

---

## BLOCK 5 - Hybrid Static Decomposition

**Goal:** Refactor `structured_dynamics.py` to enforce AD-2: split static into `z_static_immutable` (architectural passthrough) and `z_static_slow` (residual with drift penalty).

**Branch / Commit / Merge:**
- Branch: `block/05-hybrid-static`
- Commit prefix: `feat(block-05):`
- PR title: `Block 5 - Hybrid static decomposition (immutable passthrough + slow residual)`

---

## BLOCK 6 - Identifiability Loss

**Goal:** Implement AD-3: enforce that `z_static_immutable` carries information that does not depend on the action.

**Branch / Commit / Merge:**
- Branch: `block/06-identifiability`
- Commit prefix: `feat(block-06):`
- PR title: `Block 6 - Intervention loss, action-invariance critic, env hash`

---

## BLOCK 7 - Encoder Tests & Cleanup

**Goal:** Replace the v2.0 test that targeted the wrong encoder. Cover ContinuousEncoder with proper unit tests.

**Branch / Commit / Merge:**
- Branch: `block/07-encoder-tests`
- Commit prefix: `test(block-07):`
- PR title: `Block 7 - ContinuousEncoder test coverage, remove legacy encoder references`

---

## BLOCK 8 - Physics Tests for the Environment

**Goal:** Bring environment test coverage from 0% to fully specified physics contract.

**Branch / Commit / Merge:**
- Branch: `block/08-physics-tests`
- Commit prefix: `test(block-08):`
- PR title: `Block 8 - Physics test suite and chaos tripwire`

---

## BLOCK 9 - Determinism Canary & Rollout Drift Evaluation

**Goal:** Add two integration tests that catch regressions invisible to unit tests.

**Branch / Commit / Merge:**
- Branch: `block/09-determinism-rollout`
- Commit prefix: `test(block-09):`
- PR title: `Block 9 - Determinism canary and rollout drift evaluation`

---

## BLOCK 10 - Multi-Object Generalization

**Goal:** Replace fixed 6D input with entity-centric encoding supporting n_objects ∈ [3, 10].

**Branch / Commit / Merge:**
- Branch: `block/10-multi-object`
- Commit prefix: `feat(block-10):`
- PR title: `Block 10 - Entity encoder for variable n_objects`

---

## BLOCK 11 - Partial Observability

**Goal:** Train against observations that include only the nearest-K objects.

**Branch / Commit / Merge:**
- Branch: `block/11-partial-observability`
- Commit prefix: `feat(block-11):`
- PR title: `Block 11 - Partial observability and recurrent belief state`

---

## BLOCK 12 - Variable Physics & Process Noise

**Goal:** Sample gravity and friction per episode. Add Gaussian process noise.

**Branch / Commit / Merge:**
- Branch: `block/12-variable-physics`
- Commit prefix: `feat(block-12):`
- PR title: `Block 12 - Variable physics, process noise, latent probing`

---

## BLOCK 13 - Production Readiness & Release v3.0

**Goal:** Model card, ONNX export, CHANGELOG, tagged release v3.0.0.

**Branch / Commit / Merge:**
- Branch: `block/13-release-v3`
- Commit prefix: `release(block-13):`
- PR title: `Block 13 - Production readiness, model card, ONNX export, v3.0.0 release`
- Merge: squash to main, then tag `v3.0.0`.
