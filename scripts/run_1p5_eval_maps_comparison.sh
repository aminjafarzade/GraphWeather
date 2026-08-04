#!/usr/bin/env bash
# =============================================================================
# run_1p5_eval_maps_comparison.sh
#
# Post-training analysis for the 1.5-degree (121x240, poles included) kai_1p5
# models. Runs, for an ALREADY-TRAINED model, the three things the 2.5-degree
# runs get and the 1.5-degree ones did not:
#
#   1) climatology   — build data/stats/kai_1p5_train_dayofyear_climatology.nc
#                      from the train split (skipped if it already exists)
#   2) evaluation    — 10-day rollouts on the 2020 test split, weekly ICs,
#                      RMSE + ACC + persistence + bootstrap CIs
#   3) bias maps     — pred / truth / bias panels per variable per lead
#   4) comparison    — overlay the model against the KAI 1.5 reference
#                      (experiments/kai_1.5.csv) and any sibling 1p5 models
#
# The KAI 1.5 reference is a raw per-channel evaluation dump
# (lead_time,variable_idx,original_channel_idx,variable_name,rmse,acc) rather
# than the curated 2.5 layout; scripts/plot_experiment_rmse_acc.py reads both
# and drops the lead-0 identity row so the lead grids line up at 1..10.
#
# ---------------------------------------------------------------------------
# Usage
#   bash scripts/run_1p5_eval_maps_comparison.sh                  # default model
#   MODEL=l4_h160 GPU=0 bash scripts/run_1p5_eval_maps_comparison.sh
#   MODEL=all bash scripts/run_1p5_eval_maps_comparison.sh        # every trained 1p5 model
#
#   MODEL   : l3_h128 (default) | l4_h128 | l4_h160 | all
#   GPU     : CUDA device index as shown by nvidia-smi (default 3)
#   STAGES  : comma list of clim,eval,maps,compare (default all four)
#   DRY_RUN : 1 prints the commands without running them
#
# Individual stages can also be toggled with RUN_CLIM/RUN_EVAL/RUN_MAPS/RUN_COMPARE=0.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

# --- environment ---------------------------------------------------------------
GPU="${GPU:-3}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/graphweather-cu128/bin/python}"

MODEL="${MODEL:-l3_h128}"
DRY_RUN="${DRY_RUN:-0}"
DEVICE="${DEVICE:-cuda}"

# --- shared 1p5 settings --------------------------------------------------------
CLIMATOLOGY="${CLIMATOLOGY:-data/stats/kai_1p5_train_dayofyear_climatology.nc}"
KAI_CSV="${KAI_CSV:-data/baselines/kai_1p5.csv}"
KAI_LABEL="${KAI_LABEL:-KAI 1.5}"
EVAL_VARIABLES="${EVAL_VARIABLES:-z500 t2m t850 msl q700 u850}"
MAP_VARIABLES="${MAP_VARIABLES:-z500 t2m msl t850}"
LEAD_TIMES="${LEAD_TIMES:-1 3 5 10}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-10}"
SELECTION="${SELECTION:-stride}"
STRIDE="${STRIDE:-7}"
N_INITIAL_CONDITIONS="${N_INITIAL_CONDITIONS:-52}"
START_TIMESTEP="${START_TIMESTEP:-1}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
CONFIDENCE_LEVEL="${CONFIDENCE_LEVEL:-0.95}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"
AGGREGATE_MODE="${AGGREGATE_MODE:-year_mean}"
MAX_BATCHES="${MAX_BATCHES:-52}"
EVAL_DIR_NAME="${EVAL_DIR_NAME:-evaluation_test_weekly52}"
COMPARISON_ROOT="${COMPARISON_ROOT:-comparisons}"

read -r -a EVAL_VAR_ARR <<< "${EVAL_VARIABLES}"
read -r -a MAP_VAR_ARR  <<< "${MAP_VARIABLES}"
read -r -a LEAD_ARR     <<< "${LEAD_TIMES}"

# --- stage toggles --------------------------------------------------------------
RUN_CLIM="${RUN_CLIM:-1}"; RUN_EVAL="${RUN_EVAL:-1}"
RUN_MAPS="${RUN_MAPS:-1}"; RUN_COMPARE="${RUN_COMPARE:-1}"
if [[ -n "${STAGES:-}" ]]; then
  RUN_CLIM=0; RUN_EVAL=0; RUN_MAPS=0; RUN_COMPARE=0
  IFS=',' read -r -a _stages <<< "${STAGES}"
  for s in ${_stages[@]+"${_stages[@]}"}; do
    case "${s// /}" in
      clim)    RUN_CLIM=1 ;;
      eval)    RUN_EVAL=1 ;;
      maps)    RUN_MAPS=1 ;;
      compare) RUN_COMPARE=1 ;;
      *) echo "ERROR: unknown stage '${s}' (want clim,eval,maps,compare)" >&2; exit 2 ;;
    esac
  done
fi

# --- model registry -------------------------------------------------------------
# Each entry: <key>|<curriculum config>|<run name>
MODELS_ALL=(
  "l3_h128|configs/experiments/config_1p5_l3_hidden128_dense_l3k24_curriculum_S2toS10_3ep_initckpt.yaml|dense_l3k24_1p5_curriculum_S2toS10_3ep_initckpt"
  "l4_h128|configs/experiments/config_1p5_l4_hidden128_dense_l4k24_curriculum_S2toS10_3ep_initckpt.yaml|dense_l4k24_1p5_hidden128_curriculum_S2toS10_3ep_initckpt"
  "l4_h160|configs/experiments/config_1p5_l4_hidden160_dense_l4k24_curriculum_S2toS10_3ep_initckpt.yaml|dense_l4k24_1p5_hidden160_curriculum_S2toS10_3ep_initckpt"
)

# Resolve the run name from the config itself when possible, so a rename in the
# YAML does not silently point this script at a directory that does not exist.
resolve_run_name() {
  local config="$1" fallback="$2" name=""
  if [[ -f "${config}" ]]; then
    name="$(awk '/^(experiment_name|run_name):/ {print $2; exit}' "${config}" | tr -d '"'"'"'')"
  fi
  printf '%s' "${name:-${fallback}}"
}

selected=()
if [[ "${MODEL}" == "all" ]]; then
  selected=("${MODELS_ALL[@]}")
else
  for entry in "${MODELS_ALL[@]}"; do
    [[ "${entry%%|*}" == "${MODEL}" ]] && selected+=("${entry}")
  done
  if [[ "${#selected[@]}" -eq 0 ]]; then
    echo "ERROR: unknown MODEL='${MODEL}' (want l3_h128, l4_h128, l4_h160, or all)" >&2
    exit 2
  fi
fi

run_cmd() {
  printf '[cmd]'; printf ' %q' "$@"; printf '\n'
  if [[ "${DRY_RUN}" != "0" ]]; then return 0; fi
  "$@"
}

# run_logged <logfile> <cmd...> -- like run_cmd but tees to a log. The tee must
# live inside the guard, otherwise DRY_RUN=1 still creates empty log files (and
# with them run directories that look like a run happened there).
run_logged() {
  local log="$1"; shift
  printf '[cmd]'; printf ' %q' "$@"; printf '\n'
  if [[ "${DRY_RUN}" != "0" ]]; then return 0; fi
  mkdir -p "$(dirname "${log}")"
  "$@" 2>&1 | tee "${log}"
}

echo "=============================================================="
echo "[1p5] MODEL=${MODEL} GPU=${GPU} DRY_RUN=${DRY_RUN}"
echo "[1p5] stages: clim=${RUN_CLIM} eval=${RUN_EVAL} maps=${RUN_MAPS} compare=${RUN_COMPARE}"
echo "[1p5] dataset: /lustre/home/mahmed/Hydro/kai_1p5_data (121x240, test=2020)"
echo "=============================================================="

# --- 1) CLIMATOLOGY (shared across models, built once) ---------------------------
FIRST_CONFIG="$(printf '%s' "${selected[0]}" | cut -d'|' -f2)"
FIRST_NAME="$(resolve_run_name "${FIRST_CONFIG}" "$(printf '%s' "${selected[0]}" | cut -d'|' -f3)")"

if [[ "${RUN_CLIM}" == "1" ]]; then
  if [[ -f "${CLIMATOLOGY}" ]]; then
    echo "[clim] ${CLIMATOLOGY} exists — skipping."
  else
    echo "[clim] building 1p5 day-of-year climatology from the train split"
    mkdir -p "$(dirname "${CLIMATOLOGY}")"
    run_cmd "${PYTHON}" scripts/build_climatology.py \
      --config "${FIRST_CONFIG}" --config_name "${FIRST_NAME}" --resolution_mode 1p5 \
      --split train --output "${CLIMATOLOGY}"
  fi
fi

# --- per-model stages ------------------------------------------------------------
evaluated_dirs=(); evaluated_labels=()

for entry in "${selected[@]}"; do
  KEY="$(printf '%s' "${entry}" | cut -d'|' -f1)"
  CONFIG="$(printf '%s' "${entry}" | cut -d'|' -f2)"
  NAME="$(resolve_run_name "${CONFIG}" "$(printf '%s' "${entry}" | cut -d'|' -f3)")"
  RUN_DIR="runs/${NAME}"
  CKPT="${RUN_DIR}/best_ckpt.tar"
  EVAL_DIR="${RUN_DIR}/${EVAL_DIR_NAME}"
  MAPS_DIR="${RUN_DIR}/visualizations_test_biasmaps"

  echo
  echo "--------------------------------------------------------------"
  echo "[model ${KEY}] ${NAME}"
  echo "  config : ${CONFIG}"
  echo "  run dir: ${RUN_DIR}"
  echo "--------------------------------------------------------------"

  if [[ ! -f "${CONFIG}" ]]; then
    echo "[model ${KEY}] SKIP — config not found: ${CONFIG}" >&2
    continue
  fi
  if [[ ! -f "${CKPT}" && "${DRY_RUN}" == "0" ]]; then
    echo "[model ${KEY}] SKIP — no trained checkpoint at ${CKPT} (train it first)." >&2
    continue
  fi

  [[ "${DRY_RUN}" == "0" ]] && mkdir -p "${RUN_DIR}/logs"

  # --- 2) EVALUATION -------------------------------------------------------------
  if [[ "${RUN_EVAL}" == "1" ]]; then
    if [[ -f "${EVAL_DIR}/S10/rollout_rmse.csv" && "${FORCE_EVAL:-0}" != "1" ]]; then
      echo "[eval] ${EVAL_DIR}/S10/rollout_rmse.csv exists — skipping (FORCE_EVAL=1 to redo)."
    else
      echo "[eval] 10-day rollouts on the 2020 test split (${N_INITIAL_CONDITIONS} weekly ICs)"
      run_logged "${RUN_DIR}/logs/eval_1p5.log" "${PYTHON}" scripts/evaluate.py \
        --config "${CONFIG}" --config_name "${NAME}" --resolution_mode 1p5 \
        --checkpoint "${CKPT}" --split test \
        --fixed_rollout_steps "${ROLLOUT_STEPS}" \
        --selection "${SELECTION}" --stride "${STRIDE}" \
        --n_initial_conditions "${N_INITIAL_CONDITIONS}" \
        --start_timestep "${START_TIMESTEP}" --include_persistence \
        --variables "${EVAL_VAR_ARR[@]}" \
        --climatology_path "${CLIMATOLOGY}" \
        --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
        --confidence_level "${CONFIDENCE_LEVEL}" --bootstrap_seed "${BOOTSTRAP_SEED}" \
        --output_dir "${EVAL_DIR}" --device "${DEVICE}" --disable_wandb
    fi
  fi

  # --- 3) BIAS MAPS --------------------------------------------------------------
  if [[ "${RUN_MAPS}" == "1" ]]; then
    echo "[maps] pred / truth / bias panels -> ${MAPS_DIR}"
    map_cmd=(
      "${PYTHON}" scripts/visualize_rollout_maps.py
      --checkpoint "${CKPT}" --config "${CONFIG}" --config_name "${NAME}"
      --resolution_mode 1p5 --split test
      --aggregate_mode "${AGGREGATE_MODE}"
      --rollout_steps "${ROLLOUT_STEPS}"
      --lead_times "${LEAD_ARR[@]}"
      --variables "${MAP_VAR_ARR[@]}"
      --output_dir "${MAPS_DIR}" --device "${DEVICE}"
      --same_scale_across_leads
    )
    [[ "${AGGREGATE_MODE}" == "year_mean" && -n "${MAX_BATCHES}" ]] && map_cmd+=(--max_batches "${MAX_BATCHES}")
    run_logged "${RUN_DIR}/logs/maps_1p5.log" "${map_cmd[@]}"
  fi

  # a model only joins the comparison once it has curves to plot
  if [[ -f "${EVAL_DIR}/S10/rollout_rmse.csv" || "${DRY_RUN}" != "0" ]]; then
    evaluated_dirs+=("${RUN_DIR}")
    evaluated_labels+=("1p5 ${KEY}")
  fi
done

# --- 4) COMPARISON vs KAI 1.5 -----------------------------------------------------
if [[ "${RUN_COMPARE}" == "1" ]]; then
  echo
  if [[ "${#evaluated_dirs[@]}" -eq 0 ]]; then
    echo "[compare] no evaluated 1p5 models — nothing to plot." >&2
  elif [[ ! -f "${KAI_CSV}" ]]; then
    echo "[compare] KAI reference not found at ${KAI_CSV} — skipping." >&2
  else
    if [[ "${MODEL}" == "all" ]]; then
      OUT_DIR="${COMPARISON_ROOT}/1p5_all_models_vs_kai"
    else
      OUT_DIR="${COMPARISON_ROOT}/1p5_${MODEL}_vs_kai"
    fi
    echo "[compare] overlaying ${#evaluated_dirs[@]} model(s) + ${KAI_LABEL} -> ${OUT_DIR}"
    [[ "${DRY_RUN}" == "0" ]] && mkdir -p "${OUT_DIR}"
    run_logged "${OUT_DIR}/comparison.log" "${PYTHON}" scripts/plot_experiment_rmse_acc.py \
      --experiments "${evaluated_dirs[@]}" \
      --labels "${evaluated_labels[@]}" \
      --kai_csv "${KAI_CSV}" --kai_label "${KAI_LABEL}" \
      --stage_dir S10 --variables "${EVAL_VAR_ARR[@]}" \
      --output_dir "${OUT_DIR}" --plot_format png
  fi
fi

echo
echo "=============================================================="
echo "[1p5] done."
for d in ${evaluated_dirs[@]+"${evaluated_dirs[@]}"}; do
  echo "  eval : ${d}/${EVAL_DIR_NAME}"
  echo "  maps : ${d}/visualizations_test_biasmaps"
done
echo "=============================================================="
