#!/usr/bin/env bash
# =============================================================================
# run_comparison_on_models.sh
#
# Overlay the 10-day RMSE/ACC curves of SEVERAL already-evaluated models into one
# comparison figure set (no training, no evaluation). Each experiment must already
# contain evaluation_test_weekly52/S10/{rollout_rmse.csv,rollout_acc.csv} -- produce
# those first with scripts/run_eval_on_model.sh (or the full pipeline).
#
# Usage (env-driven):
#   EXPERIMENTS="experiments/modelA experiments/modelB raw_l3_hidden128" \
#   LABELS="Model A, Model B, L3 hidden128" \
#   OUTPUT_DIR=comparisons/prev_models \
#   bash scripts/run_comparison_on_models.sh
#
#   EXPERIMENTS : space/comma list of experiment dirs OR config_names
#                 (a bare name resolves to experiments/<name>).
#   LABELS      : comma-separated, one per experiment (labels may contain spaces).
#                 Omitted / count-mismatch -> labels auto-derived from dir names.
#   KAI_CSV     : baseline CSV to overlay (default experiments/kai_2.5.csv; '' disables).
#   VARIABLES   : plotted variables (default: z500 t2m t850 msl q700 u850).
#   STAGE_DIR   : rollout subdir to read (default S10).
#   Add DRY_RUN=1 to preview the command.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

EXPERIMENTS="${EXPERIMENTS:-}"
LABELS="${LABELS:-}"
KAI_CSV="${KAI_CSV:-data/baselines/kai_2p5.csv}"
VARIABLES="${VARIABLES:-z500 t2m t850 msl q700 u850}"
STAGE_DIR="${STAGE_DIR:-S10}"
OUTPUT_DIR="${OUTPUT_DIR:-comparisons/comparison_rmse_acc}"
PLOT_FORMAT="${PLOT_FORMAT:-png}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${EXPERIMENTS}" ]]; then
  echo "ERROR: set EXPERIMENTS to a space/comma list of experiment dirs or config_names." >&2
  exit 2
fi

RUNS_DIR="${RUNS_DIR:-runs}"
resolve_experiment() {
  local it="$1"
  if [[ "${it}" == */* || -d "${it}" ]]; then printf '%s' "${it}"       # explicit path
  elif [[ -d "${RUNS_DIR}/${it}" ]]; then printf '%s' "${RUNS_DIR}/${it}"  # new run
  else printf '%s' "experiments/${it}"; fi                                # legacy experiment
}

# experiments -> array (whitespace/comma delimited)
read -r -a _exp_raw <<< "${EXPERIMENTS//,/ }"
exps=()
for it in ${_exp_raw[@]+"${_exp_raw[@]}"}; do
  [[ -n "${it}" ]] && exps+=("$(resolve_experiment "${it}")")
done

# labels -> array (comma delimited; labels may contain spaces)
labels=()
IFS=',' read -r -a _lbl <<< "${LABELS}"
for l in ${_lbl[@]+"${_lbl[@]}"}; do
  l="${l#"${l%%[![:space:]]*}"}"; l="${l%"${l##*[![:space:]]}"}"
  [[ -n "${l}" ]] && labels+=("${l}")
done

read -r -a var_arr <<< "${VARIABLES//,/ }"

LOG="${OUTPUT_DIR}/comparison_$(date +%Y%m%d_%H%M%S).log"

cmd=("${PYTHON:-/lustre/home/ziya/miniconda3/envs/graphweather-cu128/bin/python}" scripts/plot_experiment_rmse_acc.py --experiments "${exps[@]}")
if [[ "${#labels[@]}" -gt 0 && "${#labels[@]}" -eq "${#exps[@]}" ]]; then
  cmd+=(--labels "${labels[@]}")
elif [[ "${#labels[@]}" -gt 0 ]]; then
  echo "NOTE: label count (${#labels[@]}) != experiment count (${#exps[@]}); omitting --labels (auto)."
fi
cmd+=(--kai_csv "${KAI_CSV}" --stage_dir "${STAGE_DIR}" --variables "${var_arr[@]}" --output_dir "${OUTPUT_DIR}" --plot_format "${PLOT_FORMAT}")

echo "Comparison over ${#exps[@]} experiment(s) -> ${OUTPUT_DIR}"
printf 'CMD:'; printf ' %q' "${cmd[@]}"; printf '\n'
if [[ "${DRY_RUN}" != "0" ]]; then
  echo "[DRY_RUN] not executed. (log would be: ${LOG})"
  exit 0
fi
mkdir -p "${OUTPUT_DIR}"
"${cmd[@]}" 2>&1 | tee "${LOG}"
