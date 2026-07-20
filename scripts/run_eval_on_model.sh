#!/usr/bin/env bash
# =============================================================================
# run_eval_on_model.sh
#
# Run ONLY the 10-day RMSE/ACC evaluation on an ALREADY-TRAINED model (no training,
# no plots/diagnostics). Intended for models trained before this pipeline existed:
# point CONFIG/CONFIG_NAME at the model's config section, EXP_DIR at its experiment
# directory, and CHECKPOINT at its weights.
#
# Usage (env-driven):
#   CONFIG=configs/weather_dual_resolution_l3_hidden128.yaml \
#   CONFIG_NAME=raw_l3_hidden128 \
#   EXP_DIR=experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128 \
#   CHECKPOINT=experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128/best_ckpt.tar \
#   bash scripts/run_eval_on_model.sh
#
# Writes <EXP_DIR>/evaluation_test_weekly52/S10/{rollout_rmse.csv,rollout_acc.csv}
# plus per-variable plots and a summary. All EVAL_VARIABLES / SELECTION / STRIDE /
# N_INITIAL_CONDITIONS / ROLLOUT_STEPS / DEVICE / RESOLUTION_MODE overrides from
# run_full_pipeline.sh apply. Add DRY_RUN=1 to preview the command.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Only the EVAL stage; all others off. CHECKPOINT/EXP_DIR are read by the pipeline.
exec env \
  RUN_TRAIN=0 RUN_EVAL=1 RUN_PLOT=0 RUN_QUALITATIVE=0 RUN_DIAGNOSTICS=0 \
  bash "${SCRIPT_DIR}/run_full_pipeline.sh"
