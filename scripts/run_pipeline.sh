#!/usr/bin/env bash
# Atlas.WM full training pipeline — runs steps in order.
#
# Usage:
#   ./scripts/run_pipeline.sh                                    # fixed physics
#   ./scripts/run_pipeline.sh --variable-physics                 # domain randomization
#   ./scripts/run_pipeline.sh --variable-physics --probe         # + latent probe after training
#   ./scripts/run_pipeline.sh --variable-physics --belief        # + train PhysicsBeliefEncoder
#   ./scripts/run_pipeline.sh --variable-physics --belief --probe  # full pipeline
#   ./scripts/run_pipeline.sh --config configs/experiments/v3_hybrid_static.yaml
#
# Steps:
#   1. generate_data.py          — collect 50k transitions (+ episode_ids) from CruelGridworld
#   2. split_data.py             — split 80/10/10 into data/processed/
#   3. train.py                  — train encoder + dynamics, save best checkpoint
#   3b. train_physics_belief.py  — (optional) train GRU belief encoder for physics identification
#   4. probe_physics.py          — (optional) ridge R² probe on z_static_slow
#
# All steps are idempotent: re-running skips completed steps unless --force is passed.

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────────────────────
VARIABLE_PHYSICS=false
RUN_PROBE=false
RUN_BELIEF=false
FORCE=false
CONFIG="configs/base.yaml"
SEED=42
CHECKPOINT="checkpoints/best_model.safetensors"
BELIEF_CHECKPOINT="checkpoints/physics_belief.pt"

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variable-physics) VARIABLE_PHYSICS=true; CONFIG="configs/experiments/v3_variable_physics.yaml" ;;
    --probe)            RUN_PROBE=true ;;
    --belief)           RUN_BELIEF=true ;;
    --force)            FORCE=true ;;
    --config)           CONFIG="$2"; shift ;;
    --checkpoint)       CHECKPOINT="$2"; shift ;;
    --belief-checkpoint) BELIEF_CHECKPOINT="$2"; shift ;;
    --seed)             SEED="$2"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

# ── helpers ───────────────────────────────────────────────────────────────────
step() { echo; echo "══════════════════════════════════════"; echo "  $1"; echo "══════════════════════════════════════"; }
force_flag() { $FORCE && echo "--force" || echo ""; }

# ── step 1: generate data ────────────────────────────────────────────────────
step "1/4 — Generate data"
GEN_ARGS="--seed $SEED"
if $VARIABLE_PHYSICS; then
  GEN_ARGS="$GEN_ARGS --randomize-physics --process-noise-std 0.05"
fi
# shellcheck disable=SC2086
python scripts/generate_data.py $GEN_ARGS

# ── step 2: split ─────────────────────────────────────────────────────────────
step "2/4 — Split data"
# shellcheck disable=SC2046
python scripts/split_data.py $(force_flag)

# ── step 3: train ─────────────────────────────────────────────────────────────
step "3/4 — Train"
python scripts/train.py --config "$CONFIG" --output-checkpoint "$CHECKPOINT"

# ── step 3b: train PhysicsBeliefEncoder (optional, variable-physics only) ─────
if $RUN_BELIEF; then
  if ! $VARIABLE_PHYSICS; then
    echo "WARNING: --belief requires --variable-physics (no physics labels in fixed-physics dataset)"
  else
    step "3b — Train PhysicsBeliefEncoder (GRU for physics identification)"
    python scripts/train_physics_belief.py \
      --config "$CONFIG" \
      --output "$BELIEF_CHECKPOINT"
  fi
else
  step "3b — PhysicsBeliefEncoder (skipped — pass --belief to enable)"
fi

# ── step 4: probe (optional) ──────────────────────────────────────────────────
if $RUN_PROBE; then
  step "4/4 — Latent probe"
  python scripts/probe_physics.py \
    --checkpoint "$CHECKPOINT" \
    --data-dir data/processed \
    --split val
else
  step "4/4 — Latent probe (skipped — pass --probe to enable)"
fi

echo
echo "Pipeline complete."
echo "  Checkpoint       : $CHECKPOINT"
$RUN_BELIEF && echo "  Belief encoder   : $BELIEF_CHECKPOINT" || true
$RUN_PROBE  && echo "  Probe            : see output above" || true
