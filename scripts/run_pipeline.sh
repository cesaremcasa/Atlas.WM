#!/usr/bin/env bash
# Atlas.WM full training pipeline — runs steps in order.
#
# Usage:
#   ./scripts/run_pipeline.sh                                    # fixed physics
#   ./scripts/run_pipeline.sh --variable-physics                 # domain randomization
#   ./scripts/run_pipeline.sh --variable-physics --probe         # + latent probe after training
#   ./scripts/run_pipeline.sh --config configs/experiments/v3_hybrid_static.yaml
#
# Steps:
#   1. generate_data.py  — collect 50k transitions from CruelGridworld
#   2. split_data.py     — split 80/10/10 into data/processed/
#   3. train.py          — train encoder + dynamics, save best checkpoint
#   4. probe_physics.py  — (optional) ridge R² probe on z_static_slow
#
# All steps are idempotent: re-running skips completed steps unless --force is passed.

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────────────────────
VARIABLE_PHYSICS=false
RUN_PROBE=false
FORCE=false
CONFIG="configs/base.yaml"
SEED=42
CHECKPOINT="checkpoints/best_model.safetensors"

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variable-physics) VARIABLE_PHYSICS=true; CONFIG="configs/experiments/v3_variable_physics.yaml" ;;
    --probe)            RUN_PROBE=true ;;
    --force)            FORCE=true ;;
    --config)           CONFIG="$2"; shift ;;
    --checkpoint)       CHECKPOINT="$2"; shift ;;
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
echo "  Checkpoint : $CHECKPOINT"
$RUN_PROBE && echo "  Probe      : see output above" || true
