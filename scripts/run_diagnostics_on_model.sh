#!/usr/bin/env bash
# =============================================================================
# run_diagnostics_on_model.sh
#
# Run ONLY the diagnostics (oversmoothing / attention / layer / rollout-curve) on
# an ALREADY-TRAINED model (no training, no eval/plots). Intended for models trained
# before this pipeline existed.
#
# Usage (env-driven):
#   CONFIG=configs/weather_dual_resolution_l3_hidden128.yaml \
#   CONFIG_NAME=raw_l3_hidden128 \
#   EXP_DIR=experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128 \
#   CHECKPOINT=experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128/best_ckpt.tar \
#   bash scripts/run_diagnostics_on_model.sh
#
# Writes <EXP_DIR>/diagnostics_full_eval/. Add DRY_RUN=1 to preview.
#
# NOTE: the diagnostics rollout curve reports S2..S10 as NaN whenever the valid
# loader only provides one target step (load_only_current_rollout /
# target_rollout_steps=1). That is a MISSING-TARGET artifact of the probe, not
# model divergence -- the real 10-day RMSE/ACC comes from run_eval_on_model.sh.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

exec env \
  RUN_TRAIN=0 RUN_EVAL=0 RUN_PLOT=0 RUN_QUALITATIVE=0 RUN_DIAGNOSTICS=1 \
  bash "${SCRIPT_DIR}/run_full_pipeline.sh"
