#!/usr/bin/env bash
# =============================================================================
# run_s1_spectral_dense_experiments.sh
#
# Runs the TWO S1 100-epoch probe variants end-to-end, IN PARALLEL on two GPUs,
# each via scripts/run_full_pipeline.sh (train -> eval -> comparison plot ->
# bias maps -> diagnostics), then builds one COMBINED comparison figure set.
#
#   GPU 2  ->  spectral-loss variant
#   GPU 3  ->  dense coarse-level (l3 k=24) variant
#
# Each variant is overlaid against:
#   - L3 base 128 (the 100-epoch base S1 probe)          <- COMPARISON baseline
#   - KAI 2.5 (experiments/kai_2.5.csv)                  <- auto by the pipeline
#
# After both finish, a 4-way overlay (spectral + dense + L3 base 128 + KAI) is
# written to comparisons/s1_spectral_vs_dense_vs_base_kai/.
#
# Usage:
#   bash scripts/run_s1_spectral_dense_experiments.sh
#   DRY_RUN=1 bash scripts/run_s1_spectral_dense_experiments.sh      # print only
#   GPU_SPECTRAL=2 GPU_DENSE=3 bash scripts/run_s1_spectral_dense_experiments.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

# --- knobs -------------------------------------------------------------------
GPU_SPECTRAL="${GPU_SPECTRAL:-2}"
GPU_DENSE="${GPU_DENSE:-3}"
DRY_RUN="${DRY_RUN:-0}"

# L3 base 128 baseline to compare against (same 100-epoch S1 base probe)
BASE_CMP="${BASE_CMP:-experiments/s1only_2p5_l3_hidden128_base_100epoch_overfit_probe}"
BASE_LABEL="${BASE_LABEL:-L3 base 128 (100ep)}"

# variant A: spectral loss
SPEC_CFG="configs/experiments/config_2p5_l3_hidden128_base_S1_100epoch_spectral_loss.yaml"
SPEC_NAME="s1only_2p5_l3_hidden128_base_100epoch_spectral_loss"
SPEC_LABEL="S1 spectral (100ep)"

# variant B: dense coarse level
DENSE_CFG="configs/experiments/config_2p5_l3_hidden128_base_S1_100epoch_dense_l3k24.yaml"
DENSE_NAME="s1only_2p5_l3_hidden128_base_100epoch_dense_l3k24"
DENSE_LABEL="S1 dense l3k24 (100ep)"

RUNS_DIR="runs"
COMBINED_DIR="comparisons/s1_spectral_vs_dense_vs_base_kai"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs "${COMBINED_DIR}"

SPEC_WRAP_LOG="logs/pipeline_${SPEC_NAME}_${TS}.log"
DENSE_WRAP_LOG="logs/pipeline_${DENSE_NAME}_${TS}.log"

echo "=============================================================="
echo " S1 spectral + dense experiments (parallel)"
echo "   GPU ${GPU_SPECTRAL} -> ${SPEC_NAME}"
echo "   GPU ${GPU_DENSE} -> ${DENSE_NAME}"
echo "   baseline overlay : ${BASE_CMP}  (+ KAI auto)"
echo "   combined figures : ${COMBINED_DIR}"
echo "   dry run          : ${DRY_RUN}"
echo "=============================================================="

# preflight: baseline must already be evaluated (needs its S10 rollout CSVs)
if [[ "${DRY_RUN}" == "0" && ! -f "${BASE_CMP}/evaluation_test_weekly52/S10/rollout_rmse.csv" ]]; then
  echo "WARNING: baseline eval CSV not found under ${BASE_CMP}/evaluation_test_weekly52/S10/;" >&2
  echo "         the per-variant comparison plots may skip the L3-base overlay." >&2
fi

# --- launch both full pipelines in parallel, pinned to their GPUs ------------
CUDA_VISIBLE_DEVICES="${GPU_SPECTRAL}" \
DRY_RUN="${DRY_RUN}" \
CONFIG="${SPEC_CFG}" \
CONFIG_NAME="${SPEC_NAME}" \
COMPARISON_EXPERIMENTS="${BASE_CMP}" \
COMPARISON_LABELS="${BASE_LABEL}" \
bash scripts/run_full_pipeline.sh > "${SPEC_WRAP_LOG}" 2>&1 &
PID_SPEC=$!
echo "launched spectral pipeline (GPU ${GPU_SPECTRAL}) pid=${PID_SPEC} -> ${SPEC_WRAP_LOG}"

CUDA_VISIBLE_DEVICES="${GPU_DENSE}" \
DRY_RUN="${DRY_RUN}" \
CONFIG="${DENSE_CFG}" \
CONFIG_NAME="${DENSE_NAME}" \
COMPARISON_EXPERIMENTS="${BASE_CMP}" \
COMPARISON_LABELS="${BASE_LABEL}" \
bash scripts/run_full_pipeline.sh > "${DENSE_WRAP_LOG}" 2>&1 &
PID_DENSE=$!
echo "launched dense pipeline    (GPU ${GPU_DENSE}) pid=${PID_DENSE} -> ${DENSE_WRAP_LOG}"

# --- wait for both (guarded so set -e does not abort on a non-zero child) -----
STAT_SPEC=0;  wait "${PID_SPEC}"  && STAT_SPEC=0  || STAT_SPEC=$?
STAT_DENSE=0; wait "${PID_DENSE}" && STAT_DENSE=0 || STAT_DENSE=$?
echo "spectral pipeline exit=${STAT_SPEC}   dense pipeline exit=${STAT_DENSE}"

# --- combined 4-way overlay (spectral + dense + L3 base 128 + KAI) -----------
SPEC_RUN="${RUNS_DIR}/${SPEC_NAME}"
DENSE_RUN="${RUNS_DIR}/${DENSE_NAME}"
spec_ok=0;  [[ -f "${SPEC_RUN}/evaluation_test_weekly52/S10/rollout_rmse.csv" ]] && spec_ok=1
dense_ok=0; [[ -f "${DENSE_RUN}/evaluation_test_weekly52/S10/rollout_rmse.csv" ]] && dense_ok=1

if [[ "${DRY_RUN}" == "1" || ( "${spec_ok}" == "1" && "${dense_ok}" == "1" ) ]]; then
  echo "== building combined comparison -> ${COMBINED_DIR} =="
  EXPERIMENTS="${SPEC_RUN} ${DENSE_RUN} ${BASE_CMP}" \
  LABELS="${SPEC_LABEL}, ${DENSE_LABEL}, ${BASE_LABEL}" \
  OUTPUT_DIR="${COMBINED_DIR}" \
  DRY_RUN="${DRY_RUN}" \
  bash scripts/run_comparison_on_models.sh
else
  echo "SKIP combined comparison: missing eval CSV (spectral_ok=${spec_ok}, dense_ok=${dense_ok})." >&2
  echo "  Fix the failed pipeline, then re-run this script (already-trained stages are skipped)," >&2
  echo "  or build it manually once both evals exist:" >&2
  echo "    EXPERIMENTS=\"${SPEC_RUN} ${DENSE_RUN} ${BASE_CMP}\" \\" >&2
  echo "    LABELS=\"${SPEC_LABEL}, ${DENSE_LABEL}, ${BASE_LABEL}\" \\" >&2
  echo "    OUTPUT_DIR=${COMBINED_DIR} bash scripts/run_comparison_on_models.sh" >&2
fi

echo ""
echo "############################################################"
echo "# DONE"
echo "#   spectral : ${SPEC_RUN}/            (pipeline log: ${SPEC_WRAP_LOG})"
echo "#   dense    : ${DENSE_RUN}/           (pipeline log: ${DENSE_WRAP_LOG})"
echo "#   per-variant comparison plots : comparisons/${SPEC_NAME}/ , comparisons/${DENSE_NAME}/"
echo "#   combined 4-way (spec+dense+base+KAI) : ${COMBINED_DIR}/"
echo "############################################################"

# non-zero overall exit if either pipeline failed
[[ "${STAT_SPEC}" == "0" && "${STAT_DENSE}" == "0" ]] || exit 1
