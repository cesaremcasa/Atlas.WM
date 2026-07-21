# Atlas.WM v4.0 - Results Ledger

Every claim in this project is backed by a committed script, a reproducible
number, and (where applicable) a regression test. This ledger consolidates
them. Dates: v4.0 rebuild executed 2026-07-02 → 2026-07-04 (PRs #25–#44).

## Headline results

| # | Claim | Evidence | Where |
|---|-------|----------|-------|
| 1 | Physics is identifiable from position-only observations under a random policy | Closed-form median-of-ratios estimator: **friction_agent R² = 0.865, MAE 0.006** on the noisy (σ=0.05) on-disk dataset | `scripts/oracle_friction_agent.py`, guarded by `tests/test_oracle_friction.py` |
| 2 | A learned model identifies physics - given the right features | Belief v2 (engineered features, window 40): **gravity +0.43, friction_agent +0.22, friction_box +0.09** (raw-GRU baseline: −0.05/−0.49/−0.10) | `atlas_wm/data/dynamics_features.py`, `scripts/train_physics_belief.py` |
| 3 | Data collection matters as much as the model | Identical model+training, active vs random data: mean belief R² **0.246 → 0.341 (+39%)**; gravity **0.67** | `atlas_wm/data/exploration.py`, regression-locked in `tests/test_active_exploration.py` |
| 4 | The AD-2 architectural guarantee is real | `z_static_immutable` max drift over open-loop self-fed rollouts: **exactly 0.0**, incl. on MuJoCo | `scripts/evaluate.py`, `tests/test_rollout_training.py` |
| 5 | The self-predictive objective is stable without scale crutches | VICReg recipe: z-std 0.99, next-frame MSE **0.001750 → 0.000996 (−43%)**; frame stacking flipped from harmful to helpful | `atlas_wm/training/objectives.py`, `tests/test_objectives.py` |
| 6 | Rollout training changes rollout behavior | One-step wins h=1 (0.000993 vs 0.001157); rollout-trained wins h≥3 (h=10: 0.0357 vs 0.0389) | `training.rollout_k`, `scripts/evaluate.py` |
| 7 | Physical structure generalizes where MLPs interpolate | Symplectic-dissipative head: OOD gravity degradation **+12% vs +69%** (residual) - but absolute error 3.5× worse in-dist (honest negative) | `model.dynamics_head`, `tests/test_hamiltonian_head.py` |
| 8 | The pipeline is environment-general | Unchanged pipeline on real MuJoCo contact physics: h=1 MSE 0.001089 vs linear ceiling 0.000465 (**2.3× gap vs 3.6×** on the gridworld) | `atlas_wm/environments/mujoco_pointmass.py` |

| 9 | The identification thesis transfers to real contact physics (v4.1) | COAST/PUSH policy + physics-informed features on MuJoCo: learned belief **friction +0.28, mass +0.11** (random + generic stats: R² ≈ 0); gravity alone structurally unidentifiable (only μ·g enters box dynamics) | `atlas_wm/data/mujoco_features.py`, `CoastPushPolicy` |

## Retractions (v3.x claims that did not survive re-verification)

1. **"friction_agent is not identifiable"** - wrong. The original oracle was
   never committed and its MSE objective is destroyed by bounce outliers;
   the environment also silently voided the box-physics signal (boxes exited
   the grid, fixed in B1). See MODEL_CARD "RETRACTED" section.
2. **"Belief R² 0–0.15 is a data-regime ceiling"** - wrong. Numbers were
   confounded by a 20× data-scale mismatch between pipelines (fixed in B2)
   and probe label leakage from overlapping windows (fixed in B5).

## Named findings

- **Anti-fragility to unobservables** (B11): a deterministic info-seeking
  policy that dashed along a fixed line repeatedly struck unobservable
  obstacles and poisoned episode medians (heavy-tail R² collapse with
  near-unchanged MAE). Random walks don't repeat their mistakes;
  deterministic policies do. Golden-angle rosette rotation fixes it.
- **Evidence-length bound** (B10): friction_agent's target std (0.025) is
  below the optimal estimator's window-level error at 18 steps (0.043) -
  SNR < 1 means no learner can score positive there; 38-step windows reach
  signal scale. Belief quality is bounded by evidence, and active
  exploration raises the per-step information rate.
- **Prediction grounding** (B7): nothing in a latent-matching +
  current-frame-reconstruction loss trains the `decoder∘dynamics`
  composition inference actually uses; adding it cut ~20% error on top of
  the stable objective.

## Known limits (also in MODEL_CARD)

Toy/point-mass environments only; no real-robot or visual claims; heuristic
(not learned) exploration policy; process-noise random walk floors
long-horizon error; the symplectic head underfits at current capacity.
