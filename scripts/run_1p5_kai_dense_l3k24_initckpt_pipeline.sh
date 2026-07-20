#!/usr/bin/env bash
# =============================================================================
# run_1p5_kai_dense_l3k24_initckpt_pipeline.sh — BACK-COMPAT SHIM (P2)
#
# 1.5-degree (121x240, poles) L3 two-phase pipeline on the kai_1p5 dataset:
# S1 100ep base -> S2..S10 curriculum warm start -> weekly test eval (2020).
# The L3 graph (graph_1p5_121x240_k8_l3k24_hybrid_row_aware_L3_v4.pt) and the
# delta stats auto-build on first run. Mechanics now live in run_pipeline.sh.
#
# Usage (unchanged):
#   nohup env GPU=2 bash scripts/run_1p5_kai_dense_l3k24_initckpt_pipeline.sh \
#     > runs/1p5_kai_pipeline.out 2>&1 &
# Flags: FORCE_RETRAIN_S1=1 / FORCE_RETRAIN_CURR=1 rerun a stage; RUN_EVAL=0 skips eval.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [[ "${RUN_EVAL:-1}" != "1" ]]; then STAGES="train"; else STAGES="train,clim,eval"; fi

exec bash "${SCRIPT_DIR}/run_pipeline.sh" \
  --resolution 1p5 --gpu "${GPU:-2}" \
  --s1-config configs/experiments/config_1p5_l3_hidden128_base_S1_100epoch_dense_l3k24.yaml \
  --s1-name   s1only_1p5_l3_hidden128_base_100epoch_dense_l3k24 \
  --curr-config configs/experiments/config_1p5_l3_hidden128_dense_l3k24_curriculum_S2toS10_3ep_initckpt.yaml \
  --curr-name   dense_l3k24_1p5_curriculum_S2toS10_3ep_initckpt \
  --clim data/stats/kai_1p5_train_dayofyear_climatology.nc \
  --stages "${STAGES}"
