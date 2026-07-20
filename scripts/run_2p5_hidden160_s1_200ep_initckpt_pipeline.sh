#!/usr/bin/env bash
# =============================================================================
# run_2p5_hidden160_s1_200ep_initckpt_pipeline.sh — BACK-COMPAT SHIM (P2)
#
# hidden-160 initckpt recipe with a 200-epoch S1 base (does a longer-cooked S1
# base carry through the curriculum?). Same stages as the 100ep twin; only the
# S1/curriculum configs, the comparison set, and the plot labels differ.
# Mechanics now live in scripts/run_pipeline.sh.
#
# Usage (unchanged):
#   nohup env GPU=3 bash scripts/run_2p5_hidden160_s1_200ep_initckpt_pipeline.sh \
#     > runs/2p5_hidden160_s1_200ep_pipeline.out 2>&1 &
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
  --s1-config configs/experiments/config_2p5_l3_hidden160_base_S1_200epoch_dense_l3k24.yaml \
  --s1-name   s1only_2p5_l3_hidden160_base_200epoch_dense_l3k24 \
  --curr-config configs/experiments/config_2p5_l3_hidden160_dense_l3k24_curriculum_S2toS10_3ep_initckpt_s1_200ep.yaml \
  --curr-name   dense_l3k24_hidden160_curriculum_S2toS10_3ep_initckpt_s1_200ep \
  --primary-label hidden160_s1_200ep \
  --compare "runs/2p5_l3_h160_densel3k24_currS2toS10x3_initckpt runs/2p5_l3_h128_densel3k24_currS2toS10x3_initckpt runs/2p5_l3_h128_densel3k24_currS2toS10x3_flatlr5e7" \
  --compare-labels "hidden160_s1_100ep, initckpt_h128, flat_lr5e7" \
  --stages "${STAGES}"
