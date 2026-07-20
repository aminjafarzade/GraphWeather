#!/usr/bin/env bash
# =============================================================================
# run_1p5_kai_l4_initckpt_pipeline.sh — BACK-COMPAT SHIM (P2 consolidation)
#
# The 1.5-degree (121x240, poles) L4 two-phase pipeline mechanics now live in
# scripts/run_pipeline.sh. This shim keeps the 1p5-specific knowledge — the
# VARIANT -> (S1 config, curriculum config) matrix and the shared graph / clim
# paths — and delegates train/eval to the unified launcher.
#
# Usage (unchanged):
#   nohup env GPU=0 VARIANT=h160 bash scripts/run_1p5_kai_l4_initckpt_pipeline.sh \
#     > runs/1p5_kai_l4_h160_pipeline.out 2>&1 &
# Flags: VARIANT=h128|h160|h128_150ep|h160_150ep (default h128);
#        FORCE_RETRAIN_S1=1 / FORCE_RETRAIN_CURR=1 rerun a stage;
#        RUN_EVAL=0 skips eval; PREBUILD_ONLY=1 builds the shared graph and exits;
#        PYTHON=<path> overrides the interpreter; DRY_RUN=1 previews commands.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

VARIANT="${VARIANT:-h128}"
case "${VARIANT}" in
  h128)
    S1_CONFIG="configs/experiments/config_1p5_l4_hidden128_base_S1_100epoch_dense_l4k24.yaml"
    S1_NAME="s1only_1p5_l4_hidden128_base_100epoch_dense_l4k24"
    CURR_CONFIG="configs/experiments/config_1p5_l4_hidden128_dense_l4k24_curriculum_S2toS10_3ep_initckpt.yaml"
    CURR_NAME="dense_l4k24_1p5_hidden128_curriculum_S2toS10_3ep_initckpt" ;;
  h160)
    S1_CONFIG="configs/experiments/config_1p5_l4_hidden160_base_S1_100epoch_dense_l4k24.yaml"
    S1_NAME="s1only_1p5_l4_hidden160_base_100epoch_dense_l4k24"
    CURR_CONFIG="configs/experiments/config_1p5_l4_hidden160_dense_l4k24_curriculum_S2toS10_3ep_initckpt.yaml"
    CURR_NAME="dense_l4k24_1p5_hidden160_curriculum_S2toS10_3ep_initckpt" ;;
  h128_150ep)
    S1_CONFIG="configs/experiments/config_1p5_l4_hidden128_base_S1_150epoch_dense_l4k24.yaml"
    S1_NAME="s1only_1p5_l4_hidden128_base_150epoch_dense_l4k24"
    CURR_CONFIG="configs/experiments/config_1p5_l4_hidden128_dense_l4k24_curriculum_S2toS10_3ep_initckpt_s1_150ep.yaml"
    CURR_NAME="dense_l4k24_1p5_hidden128_curriculum_S2toS10_3ep_initckpt_s1_150ep" ;;
  h160_150ep)
    S1_CONFIG="configs/experiments/config_1p5_l4_hidden160_base_S1_150epoch_dense_l4k24.yaml"
    S1_NAME="s1only_1p5_l4_hidden160_base_150epoch_dense_l4k24"
    CURR_CONFIG="configs/experiments/config_1p5_l4_hidden160_dense_l4k24_curriculum_S2toS10_3ep_initckpt_s1_150ep.yaml"
    CURR_NAME="dense_l4k24_1p5_hidden160_curriculum_S2toS10_3ep_initckpt_s1_150ep" ;;
  *)
    echo "[pipeline] ERROR: unknown VARIANT='${VARIANT}' (h128, h160, h128_150ep, h160_150ep)." >&2
    exit 2 ;;
esac

GRAPH="graphs/graph_1p5_121x240_k8_levelk8-8-8-8-24_hybrid_row_aware_L4_v5.pt"
GRAPH_DATA="/lustre/home/mahmed/Hydro/kai_1p5_data/train"
CLIM="data/stats/kai_1p5_train_dayofyear_climatology.nc"

# stage selection mirrors the old flags: PREBUILD_ONLY=graph only; RUN_EVAL=0 stops
# after training; otherwise graph -> train -> clim -> eval.
if [[ "${PREBUILD_ONLY:-0}" == "1" ]]; then
  STAGES="graph"
elif [[ "${RUN_EVAL:-1}" != "1" ]]; then
  STAGES="graph,train"
else
  STAGES="graph,train,clim,eval"
fi

exec bash "${SCRIPT_DIR}/run_pipeline.sh" \
  --resolution 1p5 \
  --s1-config "${S1_CONFIG}" --s1-name "${S1_NAME}" \
  --curr-config "${CURR_CONFIG}" --curr-name "${CURR_NAME}" \
  --graph "${GRAPH}" --graph-data "${GRAPH_DATA}" \
  --clim "${CLIM}" \
  --stages "${STAGES}"
