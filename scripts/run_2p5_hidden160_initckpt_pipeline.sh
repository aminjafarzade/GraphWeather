#!/usr/bin/env bash
# =============================================================================
# run_2p5_hidden160_initckpt_pipeline.sh — BACK-COMPAT SHIM (P2 consolidation)
#
# hidden-160 twin of the winning 2p5 initckpt recipe (S1 100ep base -> S2..S10
# curriculum warm start -> eval/plot/qual/diag/dashmaps vs baselines + KAI).
# Mechanics now live in scripts/run_pipeline.sh.
#
# Usage (unchanged):
#   nohup env GPU=3 bash scripts/run_2p5_hidden160_initckpt_pipeline.sh \
#     > runs/2p5_hidden160_pipeline.out 2>&1 &
# Flags: FORCE_RETRAIN_S1=1 / FORCE_RETRAIN_CURR=1 rerun training;
#        RUN_EVAL/RUN_PLOT/RUN_QUAL/RUN_DIAG/RUN_DASHMAPS=0 skip a stage.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

STAGES="train"
[[ "${RUN_EVAL:-1}"     == "1" ]] && STAGES+=",eval"
[[ "${RUN_PLOT:-1}"     == "1" ]] && STAGES+=",plot"
[[ "${RUN_QUAL:-1}"     == "1" ]] && STAGES+=",qual"
[[ "${RUN_DIAG:-1}"     == "1" ]] && STAGES+=",diag"
[[ "${RUN_DASHMAPS:-1}" == "1" ]] && STAGES+=",dashmaps"

exec bash "${SCRIPT_DIR}/run_pipeline.sh" \
  --resolution 2p5 --gpu "${GPU:-3}" \
  --s1-config configs/experiments/config_2p5_l3_hidden160_base_S1_100epoch_dense_l3k24.yaml \
  --s1-name   s1only_2p5_l3_hidden160_base_100epoch_dense_l3k24 \
  --curr-config configs/experiments/config_2p5_l3_hidden160_dense_l3k24_curriculum_S2toS10_3ep_initckpt.yaml \
  --curr-name   dense_l3k24_hidden160_curriculum_S2toS10_3ep_initckpt \
  --primary-label hidden160 \
  --compare "runs/dense_l3k24_curriculum_S2toS10_3ep_initckpt runs/dense_l3k24_curriculum_S2toS10_3ep_flat_lr5e7 runs/dense_l3k24_orogtisr_lossw_scratch_S1x100_S2toS10x3" \
  --compare-labels "initckpt, flat_lr5e7, orogtisr_scratch" \
  --stages "${STAGES}"
