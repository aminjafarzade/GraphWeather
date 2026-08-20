#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — the single parameterized launcher (P2 consolidation)
#
# Replaces the sprawl of bespoke run_*_pipeline.sh copies. Two modes:
#
#   SINGLE-PHASE   one config trained + evaluated + plotted + mapped + diagnosed
#     scripts/run_pipeline.sh --config <p> --config-name <n> [--stages ...]
#
#   TWO-PHASE      S1 base pretrain -> S2..S10 curriculum warm-started from it,
#                  then the downstream stages run on the curriculum run
#     scripts/run_pipeline.sh --s1-config <p> --s1-name <n> \
#                             --curr-config <p> --curr-name <n> [--graph <p>] ...
#
# Stages (ordered): graph, train, clim, eval, plot, qual, diag, dashmaps.
# Select a subset with --stages graph,train,eval (default: all EXCEPT graph +
# dashmaps, which run only when --graph is given / dashmaps requested).
#
# The interpreter is PINNED by default to the graphweather-cu128 env — bare
# `python` silently misbehaves on the sm_120 box (see CONTRIBUTING.md §6.2).
# Override with --python <path> or PYTHON=<path>.
#
# Back-compat: every flag also has an env-var fallback with the SAME name the old
# run_full_pipeline.sh used (CONFIG, CONFIG_NAME, RUN_TRAIN, ...), so the old
# scripts can shim into this one by forwarding their environment unchanged.
#
# Dry run (print every command, execute nothing):  --dry-run  (or DRY_RUN=1)
# =============================================================================
set -euo pipefail

# --- locate repo root (works regardless of caller's CWD) ---------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

# --- pinned interpreter + caches (overridable) -------------------------------
PYTHON="${PYTHON:-$HOME/miniconda3/envs/graphweather-cu128/bin/python}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"

# --- required / mode inputs --------------------------------------------------
CONFIG="${CONFIG:-}"
CONFIG_NAME="${CONFIG_NAME:-}"
S1_CONFIG="${S1_CONFIG:-}"
S1_NAME="${S1_NAME:-}"
CURR_CONFIG="${CURR_CONFIG:-}"
CURR_NAME="${CURR_NAME:-}"
GPU="${GPU:-}"                       # if set, exported as CUDA_VISIBLE_DEVICES
STAGES="${STAGES:-}"                 # csv/space list; empty => default toggles

# --- shared defaults (match the 2.5-degree weekly-52 conventions) ------------
RESOLUTION_MODE="${RESOLUTION_MODE:-2p5}"
DEVICE="${DEVICE:-cuda}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-10}"
SELECTION="${SELECTION:-stride}"
STRIDE="${STRIDE:-7}"
N_INITIAL_CONDITIONS="${N_INITIAL_CONDITIONS:-52}"
START_TIMESTEP="${START_TIMESTEP:-1}"
# Resolution-dependent: left empty here and resolved AFTER arg parsing, so
# --resolution changes the default. An env var or --clim/--kai-csv flag wins.
CLIMATOLOGY_PATH="${CLIMATOLOGY_PATH:-}"
KAI_CSV="${KAI_CSV:-}"

GRAPH_PATH="${GRAPH_PATH:-}"         # if set, the graph stage builds it when absent
GRAPH_DATA="${GRAPH_DATA:-}"         # optional --data override for build_graph.py

RMSE_BACKEND="${RMSE_BACKEND:-current}"   # current | weatherbench2 | both
EXTERNAL_BASELINE_CSV="${EXTERNAL_BASELINE_CSV:-}"
EXTERNAL_BASELINE_LABEL="${EXTERNAL_BASELINE_LABEL:-}"

EVAL_VARIABLES="${EVAL_VARIABLES:-z500 t2m t850 msl q700 u850}"
MAP_VARIABLES="${MAP_VARIABLES:-z500 t2m msl t850}"
LEAD_TIMES="${LEAD_TIMES:-1 3 5 10}"

AGGREGATE_MODE="${AGGREGATE_MODE:-year_mean}"   # year_mean | sample
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
MAX_BATCHES="${MAX_BATCHES:-}"
SAME_SCALE_ACROSS_LEADS="${SAME_SCALE_ACROSS_LEADS:-1}"

COMPARISON_EXPERIMENTS="${COMPARISON_EXPERIMENTS:-}"
COMPARISON_LABELS="${COMPARISON_LABELS:-}"
PRIMARY_LABEL="${PRIMARY_LABEL:-}"   # plot legend label for THIS run (default: config name)

BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
CONFIDENCE_LEVEL="${CONFIDENCE_LEVEL:-0.95}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"
SEED="${SEED:-777}"

EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

# --- stage toggles (env fallback; --stages overrides all of them) ------------
RUN_GRAPH="${RUN_GRAPH:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_CLIM="${RUN_CLIM:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PLOT="${RUN_PLOT:-1}"
RUN_QUALITATIVE="${RUN_QUALITATIVE:-1}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
RUN_DASHMAPS="${RUN_DASHMAPS:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_RETRAIN_S1="${FORCE_RETRAIN_S1:-0}"
FORCE_RETRAIN_CURR="${FORCE_RETRAIN_CURR:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# --- flag parser (flags win over env fallbacks) ------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resolution)              RESOLUTION_MODE="$2"; shift 2;;
    --config)                  CONFIG="$2"; shift 2;;
    --config-name|--config_name) CONFIG_NAME="$2"; shift 2;;
    --s1-config)               S1_CONFIG="$2"; shift 2;;
    --s1-name)                 S1_NAME="$2"; shift 2;;
    --curr-config)             CURR_CONFIG="$2"; shift 2;;
    --curr-name)               CURR_NAME="$2"; shift 2;;
    --stages)                  STAGES="$2"; shift 2;;
    --gpu)                     GPU="$2"; shift 2;;
    --python)                  PYTHON="$2"; shift 2;;
    --compare)                 COMPARISON_EXPERIMENTS="$2"; shift 2;;
    --compare-labels)          COMPARISON_LABELS="$2"; shift 2;;
    --primary-label)           PRIMARY_LABEL="$2"; shift 2;;
    --rmse-backend)            RMSE_BACKEND="$2"; shift 2;;
    --graph)                   GRAPH_PATH="$2"; shift 2;;
    --graph-data)              GRAPH_DATA="$2"; shift 2;;
    --clim)                    CLIMATOLOGY_PATH="$2"; shift 2;;
    --kai-csv)                 KAI_CSV="$2"; shift 2;;
    --external-baseline-csv)   EXTERNAL_BASELINE_CSV="$2"; shift 2;;
    --external-baseline-label) EXTERNAL_BASELINE_LABEL="$2"; shift 2;;
    --dry-run)                 DRY_RUN=1; shift;;
    -h|--help)                 usage; exit 0;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2;;
  esac
done

# --- resolution-aware defaults (flag/env overrides above win) ----------------
# 2p5 resolves to the historical defaults byte-for-byte; 1p5 picks the 1.5-deg
# climatology (built by the clim stage on first run) and KAI reference CSV.
CLIMATOLOGY_PATH="${CLIMATOLOGY_PATH:-data/stats/${RESOLUTION_MODE}_train_dayofyear_climatology.nc}"
KAI_CSV="${KAI_CSV:-data/baselines/kai_${RESOLUTION_MODE}.csv}"

# --- GPU pinning -------------------------------------------------------------
if [[ -n "${GPU}" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi

# --- decide mode -------------------------------------------------------------
TWO_PHASE=0
if [[ -n "${S1_CONFIG}" || -n "${CURR_CONFIG}" ]]; then
  TWO_PHASE=1
  [[ -n "${S1_CONFIG}" && -n "${S1_NAME}" && -n "${CURR_CONFIG}" && -n "${CURR_NAME}" ]] || {
    echo "ERROR: two-phase needs --s1-config/--s1-name and --curr-config/--curr-name." >&2; exit 2; }
  # downstream (eval/plot/qual/diag/dashmaps) targets the curriculum run
  CONFIG="${CURR_CONFIG}"
  CONFIG_NAME="${CURR_NAME}"
else
  [[ -n "${CONFIG}" && -n "${CONFIG_NAME}" ]] || {
    echo "ERROR: single-phase needs --config and --config-name (or CONFIG/CONFIG_NAME)." >&2; exit 2; }
fi

# --- --stages overrides the RUN_* toggles ------------------------------------
if [[ -n "${STAGES}" ]]; then
  RUN_GRAPH=0 RUN_TRAIN=0 RUN_CLIM=0 RUN_EVAL=0 RUN_PLOT=0 RUN_QUALITATIVE=0 RUN_DIAGNOSTICS=0 RUN_DASHMAPS=0
  _stages_norm="${STAGES//,/ }"
  for s in ${_stages_norm}; do
    case "$s" in
      graph) RUN_GRAPH=1;;
      train) RUN_TRAIN=1;;
      clim)  RUN_CLIM=1;;
      eval)  RUN_EVAL=1;;
      plot)  RUN_PLOT=1;;
      qual|qualitative) RUN_QUALITATIVE=1;;
      diag|diagnostics) RUN_DIAGNOSTICS=1;;
      dashmaps) RUN_DASHMAPS=1;;
      *) echo "ERROR: unknown stage '${s}' (graph,train,clim,eval,plot,qual,diag,dashmaps)." >&2; exit 2;;
    esac
  done
fi

# --- output roots + derived paths (from the downstream/target run) -----------
RUNS_DIR="${RUNS_DIR:-runs}"
COMPARISONS_DIR="${COMPARISONS_DIR:-comparisons}"
EXP_DIR="${EXP_DIR:-${RUNS_DIR}/${CONFIG_NAME}}"
CKPT="${CHECKPOINT:-${EXP_DIR}/best_ckpt.tar}"
EVAL_DIR="${EXP_DIR}/evaluation_test_weekly52"     # name matters: the plotter auto-discovers it
PLOT_DIR="${PLOT_DIR:-${COMPARISONS_DIR}/${CONFIG_NAME}}"
QUAL_DIR="${EXP_DIR}/visualizations_test_biasmaps"
DIAG_DIR="${EXP_DIR}/diagnostics_full_eval"
LOG_DIR="${EXP_DIR}/logs"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
STARTED_AT="$(date -Is)"
[[ "${DRY_RUN}" == "0" ]] && mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/pipeline_${RUN_TS}.log"

# --- helpers -----------------------------------------------------------------
log_master() {
  if [[ "${DRY_RUN}" == "0" ]]; then printf '%s\n' "$*" | tee -a "${MASTER_LOG}"; else printf '%s\n' "$*"; fi
}

split_ws() { local __name="$1" __val="$2"; read -r -a "${__name}" <<< "${__val//,/ }"; }
split_comma() {
  local __name="$1" __val="$2"; local -a __out=(); local IFS=','; local piece
  for piece in ${__val}; do
    piece="${piece#"${piece%%[![:space:]]*}"}"; piece="${piece%"${piece##*[![:space:]]}"}"
    [[ -n "${piece}" ]] && __out+=("${piece}")
  done
  eval "${__name}=(\"\${__out[@]}\")"
}

resolve_experiment() {
  local item="$1"
  if [[ "${item}" == */* || -d "${item}" ]]; then printf '%s' "${item}"
  elif [[ -d "${RUNS_DIR}/${item}" ]]; then printf '%s' "${RUNS_DIR}/${item}"
  else printf '%s' "experiments/${item}"; fi
}

SUMMARY_ROWS=()
record_summary() { SUMMARY_ROWS+=("$1|$2|$3"); }
print_cmd() { printf 'CMD:'; printf ' %q' "$@"; printf '\n'; }

STAGE_ELAPSED=0
STAGE_STATUS=0
run_logged() {
  local stage="$1"; shift
  local logfile="${LOG_DIR}/${stage}_${RUN_TS}.log"
  log_master ""
  log_master "============================================================"
  log_master "STAGE: ${stage}   $(date '+%Y-%m-%d %H:%M:%S')"
  log_master "============================================================"
  print_cmd "$@" | { [[ "${DRY_RUN}" == "0" ]] && tee -a "${MASTER_LOG}" || cat; }
  if [[ "${DRY_RUN}" != "0" ]]; then
    log_master "[DRY_RUN] not executed. (log would be: ${logfile})"
    STAGE_STATUS=0; STAGE_ELAPSED=0; return 0
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

# stage_complete <run_dir> <config> <config_name> -- true only when the stage ran
# to its configured max_epochs (a checkpoint alone proves nothing: best_ckpt.tar
# is written after epoch 1). Incomplete stages fall through so train.py resumes.
stage_complete() {
  local run_dir="$1" config="$2" name="$3"
  [[ -f "${run_dir}/best_ckpt.tar" ]] || return 1
  local metrics="${run_dir}/training_metrics.csv"
  [[ -f "${metrics}" ]] || return 1
  local want done_
  want="$("${PYTHON}" -c "
import sys,yaml
d=yaml.safe_load(open('${config}'))
print(int(d.get('${name}',{}).get('max_epochs',0)))" 2>/dev/null || echo 0)"
  done_="$(( $(wc -l < "${metrics}") - 1 ))"
  [[ "${want}" -gt 0 && "${done_}" -ge "${want}" ]]
}

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
    echo "  run dir         : ${EXP_DIR}"
    echo "  checkpoint      : ${CKPT}"
    echo "  eval RMSE/ACC   : ${EVAL_DIR}/S10/rollout_rmse.csv , rollout_acc.csv"
    echo "  comparison plot : ${PLOT_DIR}/"
    echo "  master log      : ${MASTER_LOG}"
    echo "############################################################"
  } | { [[ "${DRY_RUN}" == "0" ]] && tee -a "${MASTER_LOG}" || cat; }
}
trap finish EXIT

# --- pre-flight --------------------------------------------------------------
_cfg_check="${CONFIG}"
[[ "${TWO_PHASE}" == "1" ]] && _cfg_check="${S1_CONFIG}"
if [[ ! -f "${_cfg_check}" ]]; then
  echo "ERROR: config file not found: ${_cfg_check}" >&2; exit 2
fi

# downstream stages need a checkpoint; if not training and it's absent, fail early
_downstream_enabled=0
[[ "${RUN_EVAL}" != "0" || "${RUN_QUALITATIVE}" != "0" || "${RUN_DIAGNOSTICS}" != "0" || "${RUN_DASHMAPS}" != "0" ]] && _downstream_enabled=1
if [[ "${TWO_PHASE}" == "0" && "${_downstream_enabled}" == "1" && "${RUN_TRAIN}" == "0" && ! -f "${CKPT}" ]]; then
  echo "ERROR: downstream stages enabled but checkpoint is missing and train is off: ${CKPT}" >&2
  exit 2
fi

split_ws  EVAL_VAR_ARR  "${EVAL_VARIABLES}"
split_ws  MAP_VAR_ARR   "${MAP_VARIABLES}"
split_ws  LEAD_ARR      "${LEAD_TIMES}"
split_ws  COMP_EXP_RAW  "${COMPARISON_EXPERIMENTS}"
split_comma COMP_LABEL_ARR "${COMPARISON_LABELS}"
split_ws  EXTRA_TRAIN_ARR "${EXTRA_TRAIN_ARGS}"
split_ws  EXTRA_EVAL_ARR  "${EXTRA_EVAL_ARGS}"

log_master "Repo root       : ${REPO_ROOT}"
log_master "Interpreter     : ${PYTHON}"
log_master "Mode            : $([[ "${TWO_PHASE}" == "1" ]] && echo two-phase || echo single-phase)"
log_master "Resolution      : ${RESOLUTION_MODE}"
log_master "Target run       : ${CONFIG_NAME}  (${EXP_DIR})"
log_master "Stages          : graph=${RUN_GRAPH} train=${RUN_TRAIN} clim=${RUN_CLIM} eval=${RUN_EVAL} plot=${RUN_PLOT} qual=${RUN_QUALITATIVE} diag=${RUN_DIAGNOSTICS} dashmaps=${RUN_DASHMAPS}"
log_master "Dry run         : ${DRY_RUN}"

# =============================================================================
# STAGE bodies
# =============================================================================
stage_graph() {
  if [[ -z "${GRAPH_PATH}" ]]; then return 0; fi
  if [[ -f "${GRAPH_PATH}" ]]; then
    log_master "== Skipping GRAPH: already present (${GRAPH_PATH}) =="
    record_summary "graph" "SKIPPED (exists)" "-"; return 0
  fi
  local cmd=( "${PYTHON}" scripts/build_graph.py
    --config "${_cfg_check}" --config_name "$([[ "${TWO_PHASE}" == "1" ]] && echo "${S1_NAME}" || echo "${CONFIG_NAME}")"
    --resolution_mode "${RESOLUTION_MODE}" )
  [[ -n "${GRAPH_DATA}" ]] && cmd+=(--data "${GRAPH_DATA}")
  cmd+=(--output "${GRAPH_PATH}")
  run_logged "00_graph" "${cmd[@]}"
  record_summary "graph" "OK (exit ${STAGE_STATUS})" "${STAGE_ELAPSED}s"
}

# train one run (config, name); resume-aware in two-phase, ckpt-guarded in single
train_one() {
  local cfg="$1" name="$2" force="$3"
  local run_dir="${RUNS_DIR}/${name}" ckpt="${RUNS_DIR}/${name}/best_ckpt.tar"
  if [[ "${TWO_PHASE}" == "1" ]]; then
    if stage_complete "${run_dir}" "${cfg}" "${name}" && [[ "${force}" != "1" ]]; then
      log_master "== Skipping TRAIN ${name}: completed all epochs (force to redo) =="
      record_summary "train:${name}" "SKIPPED (complete)" "-"; return 0
    fi
    [[ -f "${ckpt}" ]] && log_master "== ${name} has a checkpoint but is NOT complete — resuming (resume: true) =="
  else
    if [[ -f "${ckpt}" && "${force}" == "0" ]]; then
      log_master "== Skipping TRAIN: checkpoint already exists (${ckpt}); FORCE_RETRAIN=1 to retrain =="
      record_summary "train" "SKIPPED (ckpt exists)" "-"; return 0
    fi
  fi
  # Multi-GPU: a comma-separated GPU list (GPU=5,6,7) launches one rank per
  # GPU via torchrun; the trainer splits the global batch across ranks so the
  # recipe is unchanged. A single GPU (or empty) runs exactly as before.
  local nproc=1
  [[ "${GPU}" == *,* ]] && nproc=$(awk -F',' '{print NF}' <<< "${GPU}")
  local cmd=( "${PYTHON}" )
  [[ "${TWO_PHASE}" == "1" ]] && cmd+=(-u)   # unbuffered for live tee logging (matches the initckpt scripts)
  if (( nproc > 1 )); then
    cmd+=( -m torch.distributed.run --standalone --nproc-per-node="${nproc}" )
    log_master "== Multi-GPU train: ${nproc} ranks over GPUs ${GPU} =="
  fi
  cmd+=( scripts/train.py
    --config "${cfg}" --config_name "${name}"
    --resolution_mode "${RESOLUTION_MODE}" --exp_dir "${RUNS_DIR}" --experiment_name "${name}" )
  ((${#EXTRA_TRAIN_ARR[@]})) && cmd+=("${EXTRA_TRAIN_ARR[@]}")
  run_logged "01_train_${name}" "${cmd[@]}"
  record_summary "train:${name}" "OK (exit ${STAGE_STATUS})" "${STAGE_ELAPSED}s"
}

stage_train() {
  if [[ "${TWO_PHASE}" == "1" ]]; then
    train_one "${S1_CONFIG}" "${S1_NAME}" "${FORCE_RETRAIN_S1}"
    if [[ ! -f "${RUNS_DIR}/${S1_NAME}/best_ckpt.tar" && "${DRY_RUN}" == "0" ]]; then
      echo "ERROR: S1 checkpoint missing after training: ${RUNS_DIR}/${S1_NAME}/best_ckpt.tar" >&2; exit 1
    fi
    train_one "${CURR_CONFIG}" "${CURR_NAME}" "${FORCE_RETRAIN_CURR}"
    if [[ ! -f "${CKPT}" && "${DRY_RUN}" == "0" ]]; then
      echo "ERROR: curriculum checkpoint missing after training: ${CKPT}" >&2; exit 1
    fi
  else
    train_one "${CONFIG}" "${CONFIG_NAME}" "${FORCE_RETRAIN}"
  fi
}

stage_clim() {
  if [[ -f "${CLIMATOLOGY_PATH}" ]]; then
    log_master "== Skipping CLIM: already present (${CLIMATOLOGY_PATH}) =="
    record_summary "clim" "SKIPPED (exists)" "-"; return 0
  fi
  local cmd=( "${PYTHON}" scripts/build_climatology.py
    --config "${CONFIG}" --config_name "${CONFIG_NAME}" --resolution_mode "${RESOLUTION_MODE}"
    --split train --output "${CLIMATOLOGY_PATH}" )
  run_logged "02_clim" "${cmd[@]}"
  record_summary "clim" "OK (exit ${STAGE_STATUS})" "${STAGE_ELAPSED}s"
}

stage_eval() {
  if [[ ! -f "${CKPT}" && "${DRY_RUN}" == "0" ]]; then echo "ERROR: EVAL requires a checkpoint: ${CKPT}" >&2; exit 2; fi
  local cmd=( "${PYTHON}" scripts/evaluate.py
    --config "${CONFIG}" --config_name "${CONFIG_NAME}" --resolution_mode "${RESOLUTION_MODE}"
    --checkpoint "${CKPT}" --split test --fixed_rollout_steps "${ROLLOUT_STEPS}"
    --selection "${SELECTION}" --stride "${STRIDE}"
    --n_initial_conditions "${N_INITIAL_CONDITIONS}" --start_timestep "${START_TIMESTEP}"
    --include_persistence --variables "${EVAL_VAR_ARR[@]}"
    --climatology_path "${CLIMATOLOGY_PATH}"
    --bootstrap_samples "${BOOTSTRAP_SAMPLES}" --confidence_level "${CONFIDENCE_LEVEL}"
    --bootstrap_seed "${BOOTSTRAP_SEED}" --output_dir "${EVAL_DIR}"
    --device "${DEVICE}" --disable_wandb )
  [[ "${RMSE_BACKEND}" != "current" ]] && cmd+=(--rmse_backend "${RMSE_BACKEND}")
  [[ -n "${EXTERNAL_BASELINE_CSV}" ]] && cmd+=(--external_baseline_csv "${EXTERNAL_BASELINE_CSV}")
  [[ -n "${EXTERNAL_BASELINE_LABEL}" ]] && cmd+=(--external_baseline_label "${EXTERNAL_BASELINE_LABEL}")
  ((${#EXTRA_EVAL_ARR[@]})) && cmd+=("${EXTRA_EVAL_ARR[@]}")
  run_logged "03_eval" "${cmd[@]}"
  record_summary "eval" "OK (exit ${STAGE_STATUS})" "${STAGE_ELAPSED}s"
}

stage_plot() {
  local -a plot_exps=("${EXP_DIR}"); local it
  for it in ${COMP_EXP_RAW[@]+"${COMP_EXP_RAW[@]}"}; do plot_exps+=("$(resolve_experiment "${it}")"); done
  local -a plot_labels=("${PRIMARY_LABEL:-${CONFIG_NAME}}"); local lb
  for lb in ${COMP_LABEL_ARR[@]+"${COMP_LABEL_ARR[@]}"}; do plot_labels+=("${lb}"); done
  if [[ ! -f "${EVAL_DIR}/S10/rollout_rmse.csv" && "${DRY_RUN}" == "0" ]]; then
    log_master "WARNING: ${EVAL_DIR}/S10/rollout_rmse.csv not found; plot may fail (run EVAL first)."
  fi
  local cmd=( "${PYTHON}" scripts/plot_experiment_rmse_acc.py --experiments "${plot_exps[@]}" )
  if [[ "${#plot_labels[@]}" -eq "${#plot_exps[@]}" ]]; then cmd+=(--labels "${plot_labels[@]}")
  else log_master "NOTE: label count (${#plot_labels[@]}) != experiment count (${#plot_exps[@]}); omitting --labels."; fi
  cmd+=( --kai_csv "${KAI_CSV}" --stage_dir S10 --variables "${EVAL_VAR_ARR[@]}"
         --output_dir "${PLOT_DIR}" --plot_format png )
  run_logged "04_plot" "${cmd[@]}"
  record_summary "plot" "OK (exit ${STAGE_STATUS})" "${STAGE_ELAPSED}s"
}

stage_qualitative() {
  if [[ ! -f "${CKPT}" && "${DRY_RUN}" == "0" ]]; then echo "ERROR: QUALITATIVE requires a checkpoint: ${CKPT}" >&2; exit 2; fi
  local cmd=( "${PYTHON}" scripts/visualize_rollout_maps.py
    --checkpoint "${CKPT}" --config "${CONFIG}" --config_name "${CONFIG_NAME}"
    --resolution_mode "${RESOLUTION_MODE}" --split test --aggregate_mode "${AGGREGATE_MODE}"
    --rollout_steps "${ROLLOUT_STEPS}" --lead_times "${LEAD_ARR[@]}" --variables "${MAP_VAR_ARR[@]}"
    --output_dir "${QUAL_DIR}" --device "${DEVICE}" )
  [[ "${SAME_SCALE_ACROSS_LEADS}" != "0" ]] && cmd+=(--same_scale_across_leads)
  if [[ "${AGGREGATE_MODE}" == "sample" ]]; then cmd+=(--sample_index "${SAMPLE_INDEX}")
  elif [[ "${AGGREGATE_MODE}" == "year_mean" && -n "${MAX_BATCHES}" ]]; then cmd+=(--max_batches "${MAX_BATCHES}"); fi
  run_logged "05_qualitative" "${cmd[@]}"
  record_summary "qualitative" "OK (exit ${STAGE_STATUS})" "${STAGE_ELAPSED}s"
}

stage_diagnostics() {
  if [[ ! -f "${CKPT}" && "${DRY_RUN}" == "0" ]]; then echo "ERROR: DIAGNOSTICS requires a checkpoint: ${CKPT}" >&2; exit 2; fi
  # NOTE: run_diagnostics.py uses --config-name (HYPHEN), unlike the other scripts.
  local cmd=( "${PYTHON}" scripts/run_diagnostics.py
    --config "${CONFIG}" --config-name "${CONFIG_NAME}" --checkpoint "${CKPT}"
    --split valid --output "${DIAG_DIR}" --resolution_mode "${RESOLUTION_MODE}" --disable_wandb )
  [[ -n "${DEVICE}" ]] && cmd+=(--device "${DEVICE}")
  run_logged "06_diagnostics" "${cmd[@]}"
  record_summary "diagnostics" "OK (exit ${STAGE_STATUS})" "${STAGE_ELAPSED}s"
}

stage_dashmaps() {
  local cmd=( "${PYTHON}" -m dashboard.generate_maps --run "${CONFIG_NAME}" )
  run_logged "07_dashmaps" "${cmd[@]}"
  record_summary "dashmaps" "OK (exit ${STAGE_STATUS})" "${STAGE_ELAPSED}s"
}

write_run_manifest() {
  [[ "${DRY_RUN}" != "0" ]] && return 0
  [[ -d "${EXP_DIR}" ]] || return 0
  local manifest git_sha finished_at
  manifest="${EXP_DIR}/run.json"
  git_sha="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  finished_at="$(date -Is)"
  cat > "${manifest}" <<JSON
{
  "schema": "gw-run-manifest/1",
  "status": "complete",
  "resolution": "${RESOLUTION_MODE}",
  "config_name": "${CONFIG_NAME}",
  "config": "${CONFIG}",
  "seed": ${SEED},
  "horizon": ${ROLLOUT_STEPS},
  "git_sha": "${git_sha}",
  "started_at": "${STARTED_AT}",
  "finished_at": "${finished_at}"
}
JSON
  log_master "Wrote run manifest: ${manifest}"
}

# --- driver ------------------------------------------------------------------
if [[ "${RUN_GRAPH}" != "0" ]]; then stage_graph; fi
if [[ "${RUN_TRAIN}" != "0" ]]; then stage_train; else record_summary "train" "SKIPPED" "-"; fi
if [[ "${RUN_CLIM}" != "0" ]]; then stage_clim; else record_summary "clim" "SKIPPED" "-"; fi
if [[ "${RUN_EVAL}" != "0" ]]; then stage_eval; else record_summary "eval" "SKIPPED" "-"; fi
if [[ "${RUN_PLOT}" != "0" ]]; then stage_plot; else record_summary "plot" "SKIPPED" "-"; fi
if [[ "${RUN_QUALITATIVE}" != "0" ]]; then stage_qualitative; else record_summary "qualitative" "SKIPPED" "-"; fi
if [[ "${RUN_DIAGNOSTICS}" != "0" ]]; then stage_diagnostics; else record_summary "diagnostics" "SKIPPED" "-"; fi
if [[ "${RUN_DASHMAPS}" != "0" ]]; then stage_dashmaps; else record_summary "dashmaps" "SKIPPED" "-"; fi

write_run_manifest
log_master ""
log_master "All requested stages completed."
