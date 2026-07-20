#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh
#
# End-to-end pipeline for ONE GraphWeather model, overlaid against comparison
# models. Stages (each individually toggleable):
#   1. TRAIN        scripts/train.py                 -> experiments/<name>/best_ckpt.tar
#   2. EVAL         scripts/evaluate.py              -> RMSE/ACC, 10-day rollout on test
#   3. PLOT         scripts/plot_experiment_rmse_acc.py -> overlay vs comparison models
#   4. QUALITATIVE  scripts/visualize_rollout_maps.py   -> pred/gt/bias maps
#   5. DIAGNOSTICS  scripts/run_diagnostics.py       -> oversmoothing/attention/etc.
#
# Everything is printed to the terminal AND saved under experiments/<name>/logs/.
#
# -----------------------------------------------------------------------------
# USAGE EXAMPLES
#
# (1) Full run of one config against two already-evaluated comparison experiments
#     plus the KAI baseline CSV:
#
#     CONFIG=configs/experiments/config_2p5_l3_hidden128_l0mlp_S1_100epoch_overfit_probe.yaml \
#     CONFIG_NAME=s1only_2p5_l3_hidden128_l0mlp_100epoch_overfit_probe \
#     COMPARISON_EXPERIMENTS="main_raw_2p5_b4_acc3_bf16_delta_warmup_cosine_l3 s1only_2p5_l3_hidden128_base_100epoch_overfit_probe" \
#     COMPARISON_LABELS="L3 warmup cosine, L3 base S1-only" \
#     bash scripts/run_full_pipeline.sh
#
# (2) Re-plot only (model already trained + evaluated); skip everything else:
#
#     CONFIG=configs/experiments/config_2p5_l3_hidden128_l0mlp_S1_100epoch_overfit_probe.yaml \
#     CONFIG_NAME=s1only_2p5_l3_hidden128_l0mlp_100epoch_overfit_probe \
#     RUN_TRAIN=0 RUN_EVAL=0 RUN_QUALITATIVE=0 RUN_DIAGNOSTICS=0 \
#     COMPARISON_EXPERIMENTS="main_raw_2p5_b4_acc3_bf16_delta_warmup_cosine_l3" \
#     COMPARISON_LABELS="L3 warmup cosine" \
#     bash scripts/run_full_pipeline.sh
#
# Dry run (print every command, execute nothing):  DRY_RUN=1 bash scripts/run_full_pipeline.sh ...
# Pick a training GPU:                             CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_pipeline.sh ...
# =============================================================================
set -euo pipefail

# --- locate repo root (works regardless of caller's CWD) ---------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

# --- required inputs ---------------------------------------------------------
CONFIG="${CONFIG:-}"
CONFIG_NAME="${CONFIG_NAME:-}"

# --- shared defaults (match the 2.5-degree weekly-52 conventions) ------------
RESOLUTION_MODE="${RESOLUTION_MODE:-2p5}"
DEVICE="${DEVICE:-cuda}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-10}"
SELECTION="${SELECTION:-stride}"
STRIDE="${STRIDE:-7}"
N_INITIAL_CONDITIONS="${N_INITIAL_CONDITIONS:-52}"
START_TIMESTEP="${START_TIMESTEP:-1}"
CLIMATOLOGY_PATH="${CLIMATOLOGY_PATH:-data/stats/2p5_train_dayofyear_climatology.nc}"
KAI_CSV="${KAI_CSV:-experiments/kai_2.5.csv}"

EVAL_VARIABLES="${EVAL_VARIABLES:-z500 t2m t850 msl q700 u850}"
MAP_VARIABLES="${MAP_VARIABLES:-z500 t2m msl t850}"
LEAD_TIMES="${LEAD_TIMES:-1 3 5 10}"

AGGREGATE_MODE="${AGGREGATE_MODE:-year_mean}"   # year_mean | sample
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
MAX_BATCHES="${MAX_BATCHES:-}"
SAME_SCALE_ACROSS_LEADS="${SAME_SCALE_ACROSS_LEADS:-1}"

# comparison overlays. Experiments: space/comma separated dirs OR config_names.
# Labels: COMMA-separated (labels may contain spaces), one per comparison entry.
COMPARISON_EXPERIMENTS="${COMPARISON_EXPERIMENTS:-}"
COMPARISON_LABELS="${COMPARISON_LABELS:-}"

# eval statistics
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
CONFIDENCE_LEVEL="${CONFIDENCE_LEVEL:-0.95}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"

# optional per-stage passthrough args (space separated)
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

# --- stage toggles / behaviour ----------------------------------------------
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PLOT="${RUN_PLOT:-1}"
RUN_QUALITATIVE="${RUN_QUALITATIVE:-1}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
DRY_RUN="${DRY_RUN:-0}"

# --- derived paths -----------------------------------------------------------
if [[ -z "${CONFIG}" || -z "${CONFIG_NAME}" ]]; then
  echo "ERROR: CONFIG and CONFIG_NAME are required." >&2
  echo "  e.g. CONFIG=configs/experiments/<cfg>.yaml CONFIG_NAME=<name> bash scripts/run_full_pipeline.sh" >&2
  exit 2
fi

# Fresh output roots: NEW experiments go under runs/ (kept separate from the old
# 11 GB experiments/ tree), comparison figures under comparisons/. Override RUNS_DIR
# to point elsewhere (e.g. RUNS_DIR=experiments to reproduce the legacy layout).
RUNS_DIR="${RUNS_DIR:-runs}"
COMPARISONS_DIR="${COMPARISONS_DIR:-comparisons}"

# EXP_DIR / CHECKPOINT may be overridden to target an EXISTING experiment whose
# directory name differs from CONFIG_NAME, or a checkpoint other than best_ckpt.tar
# (used by the "…_on_model" wrapper scripts to evaluate previously-trained models).
EXP_DIR="${EXP_DIR:-${RUNS_DIR}/${CONFIG_NAME}}"
CKPT="${CHECKPOINT:-${EXP_DIR}/best_ckpt.tar}"
EVAL_DIR="${EXP_DIR}/evaluation_test_weekly52"     # name is important: the plotter auto-discovers it
PLOT_DIR="${PLOT_DIR:-${COMPARISONS_DIR}/${CONFIG_NAME}}"
QUAL_DIR="${EXP_DIR}/visualizations_test_biasmaps"
DIAG_DIR="${EXP_DIR}/diagnostics_full_eval"
LOG_DIR="${EXP_DIR}/logs"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/pipeline_${RUN_TS}.log"

# --- helpers -----------------------------------------------------------------
log_master() { printf '%s\n' "$*" | tee -a "${MASTER_LOG}"; }

# split "a b" / "a,b" into a named array (whitespace/comma delimited)
split_ws() {
  local __name="$1" __val="$2"
  # shellcheck disable=SC2034
  read -r -a "${__name}" <<< "${__val//,/ }"
}

# split "a, b c, d" into a named array on COMMAS only (labels may contain spaces)
split_comma() {
  local __name="$1" __val="$2"
  local -a __out=()
  local IFS=','
  local piece
  for piece in ${__val}; do
    piece="${piece#"${piece%%[![:space:]]*}"}"   # ltrim
    piece="${piece%"${piece##*[![:space:]]}"}"   # rtrim
    [[ -n "${piece}" ]] && __out+=("${piece}")
  done
  eval "${__name}=(\"\${__out[@]}\")"
}

# resolve a comparison entry: a path (has '/' or is a dir) stays; a bare name -> experiments/<name>
resolve_experiment() {
  local item="$1"
  if [[ "${item}" == */* || -d "${item}" ]]; then
    printf '%s' "${item}"          # explicit path
  elif [[ -d "${RUNS_DIR}/${item}" ]]; then
    printf '%s' "${RUNS_DIR}/${item}"   # new run
  else
    printf '%s' "experiments/${item}"   # legacy experiment
  fi
}

# summary tracking
SUMMARY_ROWS=()
record_summary() { SUMMARY_ROWS+=("$1|$2|$3"); }

print_cmd() {
  printf 'CMD:'
  printf ' %q' "$@"
  printf '\n'
}

# run a stage command through tee (terminal + per-stage log), capturing the
# REAL python exit status (PIPESTATUS[0], not tee's). Aborts on failure.
STAGE_ELAPSED=0
STAGE_STATUS=0
run_logged() {
  local stage="$1"; shift
  local logfile="${LOG_DIR}/${stage}_${RUN_TS}.log"
  log_master ""
  log_master "============================================================"
  log_master "STAGE: ${stage}   $(date '+%Y-%m-%d %H:%M:%S')"
  log_master "============================================================"
  print_cmd "$@" | tee -a "${MASTER_LOG}"
  if [[ "${DRY_RUN}" != "0" ]]; then
    log_master "[DRY_RUN] not executed. (log would be: ${logfile})"
    STAGE_STATUS=0
    STAGE_ELAPSED=0
    return 0
  fi
  local start end
  start=${SECONDS}
  set +e
  "$@" 2>&1 | tee -a "${logfile}"
  STAGE_STATUS=${PIPESTATUS[0]}
  set -e
  end=${SECONDS}
  STAGE_ELAPSED=$(( end - start ))
  log_master "[stage ${stage}] exit=${STAGE_STATUS} elapsed=${STAGE_ELAPSED}s log=${logfile}"
  if [[ "${STAGE_STATUS}" != "0" ]]; then
    log_master "ERROR: stage '${stage}' failed with exit code ${STAGE_STATUS}. See ${logfile}"
  fi
  return "${STAGE_STATUS}"
}

# banner printed on exit (success or failure)
FINISHED=0
finish() {
  [[ "${FINISHED}" == "1" ]] && return
  FINISHED=1
  set +e
  {
    echo ""
    echo "############################################################"
    echo "# PIPELINE SUMMARY  (${CONFIG_NAME})"
    echo "############################################################"
    printf '%-14s | %-28s | %s\n' "STAGE" "STATUS" "ELAPSED"
    printf '%-14s-+-%-28s-+-%s\n' "--------------" "----------------------------" "--------"
    local row stage status elapsed
    for row in ${SUMMARY_ROWS[@]+"${SUMMARY_ROWS[@]}"}; do
      IFS='|' read -r stage status elapsed <<< "${row}"
      printf '%-14s | %-28s | %s\n' "${stage}" "${status}" "${elapsed}"
    done
    echo ""
    echo "Key outputs:"
    echo "  checkpoint      : ${CKPT}"
    echo "  eval RMSE/ACC   : ${EVAL_DIR}/S10/rollout_rmse.csv , rollout_acc.csv"
    echo "  comparison plot : ${PLOT_DIR}/"
    echo "  bias maps       : ${QUAL_DIR}/"
    echo "  diagnostics     : ${DIAG_DIR}/"
    echo "  master log      : ${MASTER_LOG}"
    echo "############################################################"
  } | tee -a "${MASTER_LOG}"
}
trap finish EXIT

# --- pre-flight checks -------------------------------------------------------
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config file not found: ${CONFIG}" >&2
  exit 2
fi

# downstream stages need a checkpoint. If we are NOT training this run and the
# checkpoint is absent, fail early with a clear message.
_downstream_enabled=0
[[ "${RUN_EVAL}" != "0" || "${RUN_QUALITATIVE}" != "0" || "${RUN_DIAGNOSTICS}" != "0" ]] && _downstream_enabled=1
if [[ "${_downstream_enabled}" == "1" && "${RUN_TRAIN}" == "0" && ! -f "${CKPT}" ]]; then
  echo "ERROR: downstream stages enabled but checkpoint is missing and RUN_TRAIN=0: ${CKPT}" >&2
  echo "       Train first (RUN_TRAIN=1) or point CONFIG_NAME at an experiment that has best_ckpt.tar." >&2
  exit 2
fi

# parse list envs into arrays
split_ws  EVAL_VAR_ARR  "${EVAL_VARIABLES}"
split_ws  MAP_VAR_ARR   "${MAP_VARIABLES}"
split_ws  LEAD_ARR      "${LEAD_TIMES}"
split_ws  COMP_EXP_RAW  "${COMPARISON_EXPERIMENTS}"
split_comma COMP_LABEL_ARR "${COMPARISON_LABELS}"
split_ws  EXTRA_TRAIN_ARR "${EXTRA_TRAIN_ARGS}"
split_ws  EXTRA_EVAL_ARR  "${EXTRA_EVAL_ARGS}"

log_master "Repo root       : ${REPO_ROOT}"
log_master "Config          : ${CONFIG}"
log_master "Config name     : ${CONFIG_NAME}"
log_master "Experiment dir  : ${EXP_DIR}"
log_master "Stages          : train=${RUN_TRAIN} eval=${RUN_EVAL} plot=${RUN_PLOT} qualitative=${RUN_QUALITATIVE} diagnostics=${RUN_DIAGNOSTICS}"
log_master "Dry run         : ${DRY_RUN}"
log_master "Master log      : ${MASTER_LOG}"

# =============================================================================
# STAGE 1 — TRAIN
# =============================================================================
stage_train() {
  if [[ -f "${CKPT}" && "${FORCE_RETRAIN}" == "0" ]]; then
    log_master "== Skipping TRAIN: checkpoint already exists (${CKPT}); set FORCE_RETRAIN=1 to retrain =="
    record_summary "train" "SKIPPED (ckpt exists)" "-"
    return 0
  fi
  local cmd=(
    python scripts/train.py
    --config "${CONFIG}"
    --config_name "${CONFIG_NAME}"
    --resolution_mode "${RESOLUTION_MODE}"
    --exp_dir "${RUNS_DIR}"
    --experiment_name "${CONFIG_NAME}"
  )
  ((${#EXTRA_TRAIN_ARR[@]})) && cmd+=("${EXTRA_TRAIN_ARR[@]}")
  run_logged "01_train" "${cmd[@]}"
  record_summary "train" "OK (exit ${STAGE_STATUS})" "${STAGE_ELAPSED}s"
}

# =============================================================================
# STAGE 2 — EVAL (RMSE/ACC, fixed 10-step rollout on the test split)
# =============================================================================
stage_eval() {
  if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: EVAL requires a checkpoint but none found: ${CKPT}" >&2
    exit 2
  fi
  local cmd=(
    python scripts/evaluate.py
    --config "${CONFIG}"
    --config_name "${CONFIG_NAME}"
    --resolution_mode "${RESOLUTION_MODE}"
    --checkpoint "${CKPT}"
    --split test
    --fixed_rollout_steps "${ROLLOUT_STEPS}"
    --selection "${SELECTION}"
    --stride "${STRIDE}"
    --n_initial_conditions "${N_INITIAL_CONDITIONS}"
    --start_timestep "${START_TIMESTEP}"
    --include_persistence
    --variables "${EVAL_VAR_ARR[@]}"
    --climatology_path "${CLIMATOLOGY_PATH}"
    --bootstrap_samples "${BOOTSTRAP_SAMPLES}"
    --confidence_level "${CONFIDENCE_LEVEL}"
    --bootstrap_seed "${BOOTSTRAP_SEED}"
    --output_dir "${EVAL_DIR}"
    --device "${DEVICE}"
    --disable_wandb
  )
  ((${#EXTRA_EVAL_ARR[@]})) && cmd+=("${EXTRA_EVAL_ARR[@]}")
  run_logged "02_eval" "${cmd[@]}"
  record_summary "eval" "OK (exit ${STAGE_STATUS})" "${STAGE_ELAPSED}s"
}

# =============================================================================
# STAGE 3 — COMPARISON PLOT (overlay this model with the given comparison models)
# =============================================================================
stage_plot() {
  # build experiments + labels arrays (primary first, then comparisons)
  local -a plot_exps=("${EXP_DIR}")
  local it
  for it in ${COMP_EXP_RAW[@]+"${COMP_EXP_RAW[@]}"}; do
    plot_exps+=("$(resolve_experiment "${it}")")
  done
  local -a plot_labels=("${CONFIG_NAME}")
  local lb
  for lb in ${COMP_LABEL_ARR[@]+"${COMP_LABEL_ARR[@]}"}; do
    plot_labels+=("${lb}")
  done

  if [[ ! -f "${EVAL_DIR}/S10/rollout_rmse.csv" && "${DRY_RUN}" == "0" ]]; then
    log_master "WARNING: ${EVAL_DIR}/S10/rollout_rmse.csv not found; plot may fail (run EVAL first)."
  fi

  local cmd=(
    python scripts/plot_experiment_rmse_acc.py
    --experiments "${plot_exps[@]}"
  )
  # only pass --labels when we have exactly one label per experiment; else let
  # the plotter derive labels from directory names.
  if [[ "${#plot_labels[@]}" -eq "${#plot_exps[@]}" ]]; then
    cmd+=(--labels "${plot_labels[@]}")
  else
    log_master "NOTE: label count (${#plot_labels[@]}) != experiment count (${#plot_exps[@]}); omitting --labels (auto)."
  fi
  cmd+=(
    --kai_csv "${KAI_CSV}"
    --stage_dir S10
    --variables "${EVAL_VAR_ARR[@]}"
    --output_dir "${PLOT_DIR}"
    --plot_format png
  )
  run_logged "03_plot" "${cmd[@]}"
  record_summary "plot" "OK (exit ${STAGE_STATUS})" "${STAGE_ELAPSED}s"
}

# =============================================================================
# STAGE 4 — QUALITATIVE BIAS MAPS (pred / ground-truth / bias = pred-gt panels)
# =============================================================================
stage_qualitative() {
  if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: QUALITATIVE requires a checkpoint but none found: ${CKPT}" >&2
    exit 2
  fi
  local cmd=(
    python scripts/visualize_rollout_maps.py
    --checkpoint "${CKPT}"
    --config "${CONFIG}"
    --config_name "${CONFIG_NAME}"
    --resolution_mode "${RESOLUTION_MODE}"
    --split test
    --aggregate_mode "${AGGREGATE_MODE}"
    --rollout_steps "${ROLLOUT_STEPS}"
    --lead_times "${LEAD_ARR[@]}"
    --variables "${MAP_VAR_ARR[@]}"
    --output_dir "${QUAL_DIR}"
    --device "${DEVICE}"
  )
  [[ "${SAME_SCALE_ACROSS_LEADS}" != "0" ]] && cmd+=(--same_scale_across_leads)
  if [[ "${AGGREGATE_MODE}" == "sample" ]]; then
    cmd+=(--sample_index "${SAMPLE_INDEX}")
  elif [[ "${AGGREGATE_MODE}" == "year_mean" && -n "${MAX_BATCHES}" ]]; then
    cmd+=(--max_batches "${MAX_BATCHES}")
  fi
  run_logged "04_qualitative" "${cmd[@]}"
  record_summary "qualitative" "OK (exit ${STAGE_STATUS})" "${STAGE_ELAPSED}s"
}

# =============================================================================
# STAGE 5 — DIAGNOSTICS (oversmoothing / attention / layer / rollout-curve)
#
# CAVEAT: the diagnostics rollout curve reports S2..S10 as NaN whenever the
# valid loader only provides one target step (load_only_current_rollout /
# target_rollout_steps=1). That NaN is a MISSING-TARGET artifact of this probe,
# NOT model divergence. The real 10-day RMSE/ACC comes from STAGE 2 (evaluate.py)
# which does a true autoregressive rollout with ground truth at every lead.
#
# NOTE: run_diagnostics.py uses --config-name (HYPHEN), unlike the other scripts.
# =============================================================================
stage_diagnostics() {
  if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: DIAGNOSTICS requires a checkpoint but none found: ${CKPT}" >&2
    exit 2
  fi
  local cmd=(
    python scripts/run_diagnostics.py
    --config "${CONFIG}"
    --config-name "${CONFIG_NAME}"
    --checkpoint "${CKPT}"
    --split valid
    --output "${DIAG_DIR}"
    --resolution_mode "${RESOLUTION_MODE}"
    --disable_wandb
  )
  [[ -n "${DEVICE}" ]] && cmd+=(--device "${DEVICE}")
  run_logged "05_diagnostics" "${cmd[@]}"
  record_summary "diagnostics" "OK (exit ${STAGE_STATUS})" "${STAGE_ELAPSED}s"
}

# --- driver ------------------------------------------------------------------
if [[ "${RUN_TRAIN}" != "0" ]]; then stage_train; else record_summary "train" "SKIPPED (RUN_TRAIN=0)" "-"; fi
if [[ "${RUN_EVAL}" != "0" ]]; then stage_eval; else record_summary "eval" "SKIPPED (RUN_EVAL=0)" "-"; fi
if [[ "${RUN_PLOT}" != "0" ]]; then stage_plot; else record_summary "plot" "SKIPPED (RUN_PLOT=0)" "-"; fi
if [[ "${RUN_QUALITATIVE}" != "0" ]]; then stage_qualitative; else record_summary "qualitative" "SKIPPED (RUN_QUALITATIVE=0)" "-"; fi
if [[ "${RUN_DIAGNOSTICS}" != "0" ]]; then stage_diagnostics; else record_summary "diagnostics" "SKIPPED (RUN_DIAGNOSTICS=0)" "-"; fi

log_master ""
log_master "All requested stages completed."
