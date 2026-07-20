#!/usr/bin/env bash
set -euo pipefail

# 1.5-degree (121x240, poles included) L4 / 5-level two-stage training on the
# kai_1p5 dataset (/lustre/home/mahmed/Hydro/kai_1p5_data), initckpt-style:
#   1) S1-only 100-epoch base pretrain
#   2) S2->S10 curriculum fine-tune warm-started from stage 1's best_ckpt.tar
#      (cosine 5e-5 -> 1e-6, 3 epochs per horizon, 27 epochs)
#   3) weekly test evaluation (2020, 10-day rollouts) against the 1p5
#      day-of-year climatology built from the train split.
#
# This is the L4 (num_graph_levels=5, level_k 8-8-8-8-24) counterpart of
# scripts/run_1p5_kai_dense_l3k24_initckpt_pipeline.sh, parameterised over the
# hidden width so h128 and h160 share one script.
#
# IMPORTANT: both variants use the SAME graph file
#   graphs/graph_1p5_121x240_k8_levelk8-8-8-8-24_hybrid_row_aware_L4_v5.pt
# save_graph() (src/graph_builder.py:1113) writes it non-atomically, so two
# concurrent auto-builds would corrupt it. This script builds the graph up
# front and refuses to continue if it is missing, which makes it safe to run
# the h128 and h160 variants concurrently on two GPUs *provided the first
# invocation has finished building the graph* (or you pre-build it once with
# scripts/build_graph.py, see PREBUILD below).
#
# Usage (detached, GPU index as shown by nvidia-smi):
#   nohup env GPU=0 VARIANT=h160 bash scripts/run_1p5_kai_l4_initckpt_pipeline.sh \
#     > runs/1p5_kai_l4_h160_pipeline.out 2>&1 &
# Flags: VARIANT=h128|h160|h128_150ep|h160_150ep (default h128); the *_150ep variants
#        train a 150-epoch single-step base instead of 100; FORCE_RETRAIN_S1=1 / FORCE_RETRAIN_CURR=1
#        rerun a finished stage; RUN_EVAL=0 skips stage 3; PYTHON=<path> overrides
#        the interpreter; PREBUILD_ONLY=1 builds the shared graph and exits;
#        DRY_RUN=1 prints every stage command without running anything.

# --- GPU -----------------------------------------------------------------------
GPU="${GPU:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
PYTHON="${PYTHON:-/lustre/home/ziya/miniconda3/envs/graphweather-cu128/bin/python}"

# --- locate repo root (works from anywhere) --------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

# --- variant -------------------------------------------------------------------
VARIANT="${VARIANT:-h128}"
case "${VARIANT}" in
  h128)
    S1_CONFIG="configs/experiments/config_1p5_l4_hidden128_base_S1_100epoch_dense_l4k24.yaml"
    S1_NAME="s1only_1p5_l4_hidden128_base_100epoch_dense_l4k24"
    CURR_CONFIG="configs/experiments/config_1p5_l4_hidden128_dense_l4k24_curriculum_S2toS10_3ep_initckpt.yaml"
    CURR_NAME="dense_l4k24_1p5_hidden128_curriculum_S2toS10_3ep_initckpt"
    ;;
  h160)
    S1_CONFIG="configs/experiments/config_1p5_l4_hidden160_base_S1_100epoch_dense_l4k24.yaml"
    S1_NAME="s1only_1p5_l4_hidden160_base_100epoch_dense_l4k24"
    CURR_CONFIG="configs/experiments/config_1p5_l4_hidden160_dense_l4k24_curriculum_S2toS10_3ep_initckpt.yaml"
    CURR_NAME="dense_l4k24_1p5_hidden160_curriculum_S2toS10_3ep_initckpt"
    ;;
  # 150-epoch single-step base variants. The curriculum stage is unchanged
  # (S2->S10, 3 epochs per horizon); only the base length and the warm start differ.
  h128_150ep)
    S1_CONFIG="configs/experiments/config_1p5_l4_hidden128_base_S1_150epoch_dense_l4k24.yaml"
    S1_NAME="s1only_1p5_l4_hidden128_base_150epoch_dense_l4k24"
    CURR_CONFIG="configs/experiments/config_1p5_l4_hidden128_dense_l4k24_curriculum_S2toS10_3ep_initckpt_s1_150ep.yaml"
    CURR_NAME="dense_l4k24_1p5_hidden128_curriculum_S2toS10_3ep_initckpt_s1_150ep"
    ;;
  h160_150ep)
    S1_CONFIG="configs/experiments/config_1p5_l4_hidden160_base_S1_150epoch_dense_l4k24.yaml"
    S1_NAME="s1only_1p5_l4_hidden160_base_150epoch_dense_l4k24"
    CURR_CONFIG="configs/experiments/config_1p5_l4_hidden160_dense_l4k24_curriculum_S2toS10_3ep_initckpt_s1_150ep.yaml"
    CURR_NAME="dense_l4k24_1p5_hidden160_curriculum_S2toS10_3ep_initckpt_s1_150ep"
    ;;
  *)
    echo "[pipeline] ERROR: unknown VARIANT='${VARIANT}' (expected h128, h160, h128_150ep or h160_150ep)." >&2
    exit 2
    ;;
esac

S1_RUN_DIR="runs/${S1_NAME}"
S1_CKPT="${S1_RUN_DIR}/best_ckpt.tar"
CURR_RUN_DIR="runs/${CURR_NAME}"
CURR_CKPT="${CURR_RUN_DIR}/best_ckpt.tar"

GRAPH="graphs/graph_1p5_121x240_k8_levelk8-8-8-8-24_hybrid_row_aware_L4_v5.pt"
CLIMATOLOGY="data/stats/kai_1p5_train_dayofyear_climatology.nc"
RUN_EVAL="${RUN_EVAL:-1}"

DRY_RUN="${DRY_RUN:-0}"

# Every stage runs through this, so DRY_RUN=1 previews the whole pipeline instead
# of silently launching a multi-day job.
run_stage() {
  printf '[cmd]'; printf ' %q' "$@"; printf '\n'
  if [[ "${DRY_RUN}" != "0" ]]; then return 0; fi
  "$@"
}

# run_logged <logfile> <cmd...> -- like run_stage but tees to a log. The tee has to
# live INSIDE the guard: left outside it still runs under DRY_RUN, and under
# `set -o pipefail` a missing log directory then fails the whole pipeline.
run_logged() {
  local log="$1"; shift
  printf '[cmd]'; printf ' %q' "$@"; printf '\n'
  if [[ "${DRY_RUN}" != "0" ]]; then return 0; fi
  mkdir -p "$(dirname "${log}")"
  "$@" 2>&1 | tee "${log}"
}

# stage_complete <run_dir> <config> <config_name>
#
# True only when the stage ran to its configured max_epochs. A checkpoint on its own
# proves nothing: best_ckpt.tar is written after the FIRST epoch, so keying the skip
# on file existence silently skips an interrupted stage and carries a barely-trained
# checkpoint into the next one. Incomplete stages must fall through and let
# train.py resume (these configs all set resume: true).
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

if [[ "${DRY_RUN}" != "0" ]]; then
  echo "[pipeline] DRY_RUN=1 — printing commands only, nothing will be executed."
else
  mkdir -p "${S1_RUN_DIR}/logs" "${CURR_RUN_DIR}/logs" graphs
fi

echo "[pipeline] VARIANT=${VARIANT} GPU=${GPU} PYTHON=${PYTHON}"
echo "[pipeline] dataset: /lustre/home/mahmed/Hydro/kai_1p5_data (121x240, test=2020)"

# --- 0) BUILD the shared 5-level graph (serialised, never concurrent) -------------
if [[ ! -f "${GRAPH}" ]]; then
  echo "[stage 0] building shared L4 graph -> ${GRAPH}"
  "${PYTHON}" scripts/build_graph.py \
    --config "${S1_CONFIG}" --config_name "${S1_NAME}" --resolution_mode 1p5 \
    --data /lustre/home/mahmed/Hydro/kai_1p5_data/train \
    --output "${GRAPH}"
else
  echo "[stage 0] ${GRAPH} already present — reusing."
fi

if [[ ! -f "${GRAPH}" ]]; then
  echo "[pipeline] ERROR: ${GRAPH} missing after stage 0 — aborting." >&2
  exit 1
fi

if [[ "${PREBUILD_ONLY:-0}" == "1" ]]; then
  echo "[pipeline] PREBUILD_ONLY=1 — graph ready, exiting before training."
  exit 0
fi

# --- 1) TRAIN S1 base -------------------------------------------------------------
if stage_complete "${S1_RUN_DIR}" "${S1_CONFIG}" "${S1_NAME}" && [[ "${FORCE_RETRAIN_S1:-0}" != "1" ]]; then
  echo "[stage 1] ${S1_NAME} already completed all its epochs — skipping (FORCE_RETRAIN_S1=1 to redo)."
else
  if [[ -f "${S1_CKPT}" ]]; then
    echo "[stage 1] WARNING: ${S1_NAME} has a checkpoint but has NOT finished its epochs."
    echo "[stage 1]          Continuing it (resume: true) rather than skipping to the curriculum."
  fi
  echo "[stage 1] training S1 base: ${S1_NAME}"
  run_logged "${S1_RUN_DIR}/logs/train.log" "${PYTHON}" -u scripts/train.py \
    --config "${S1_CONFIG}" --config_name "${S1_NAME}" \
    --resolution_mode 1p5 --exp_dir runs --experiment_name "${S1_NAME}"
fi

if [[ ! -f "${S1_CKPT}" && "${DRY_RUN}" == "0" ]]; then
  echo "[pipeline] ERROR: ${S1_CKPT} not found after stage 1 — aborting." >&2
  exit 1
fi

# --- 2) TRAIN S2->S10 curriculum (init_from_checkpoint is set in the YAML) --------
if stage_complete "${CURR_RUN_DIR}" "${CURR_CONFIG}" "${CURR_NAME}" && [[ "${FORCE_RETRAIN_CURR:-0}" != "1" ]]; then
  echo "[stage 2] ${CURR_NAME} already completed all its epochs — skipping (FORCE_RETRAIN_CURR=1 to redo)."
else
  if [[ -f "${CURR_CKPT}" ]]; then
    echo "[stage 2] WARNING: ${CURR_NAME} has a checkpoint but has NOT finished its epochs; continuing it."
  fi
  echo "[stage 2] training curriculum: ${CURR_NAME} (warm start from ${S1_CKPT})"
  run_logged "${CURR_RUN_DIR}/logs/train.log" "${PYTHON}" -u scripts/train.py \
    --config "${CURR_CONFIG}" --config_name "${CURR_NAME}" \
    --resolution_mode 1p5 --exp_dir runs --experiment_name "${CURR_NAME}"
fi

if [[ ! -f "${CURR_CKPT}" && "${DRY_RUN}" == "0" ]]; then
  echo "[pipeline] ERROR: ${CURR_CKPT} not found after stage 2 — aborting." >&2
  exit 1
fi

# --- 3) EVAL on test year 2020 (weekly ICs, 10-day rollouts) ----------------------
if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "[stage 3] RUN_EVAL=${RUN_EVAL} — skipping evaluation."
  exit 0
fi

if [[ ! -f "${CLIMATOLOGY}" ]]; then
  echo "[stage 3] building 1p5 day-of-year climatology from train split -> ${CLIMATOLOGY}"
  run_logged "${CURR_RUN_DIR}/logs/climatology.log" "${PYTHON}" scripts/build_climatology.py \
    --config "${CURR_CONFIG}" --config_name "${CURR_NAME}" --resolution_mode 1p5 \
    --split train --output "${CLIMATOLOGY}"
fi

echo "[stage 3] evaluating ${CURR_CKPT} on the 2020 test split"
run_logged "${CURR_RUN_DIR}/logs/eval.log" "${PYTHON}" scripts/evaluate.py \
  --config "${CURR_CONFIG}" --config_name "${CURR_NAME}" --resolution_mode 1p5 \
  --checkpoint "${CURR_CKPT}" --split test --fixed_rollout_steps 10 \
  --selection stride --stride 7 --n_initial_conditions 52 \
  --start_timestep 1 --include_persistence \
  --variables z500 t2m t850 msl q700 u850 \
  --climatology_path "${CLIMATOLOGY}" \
  --bootstrap_samples 1000 --confidence_level 0.95 --bootstrap_seed 42 \
  --output_dir "${CURR_RUN_DIR}/evaluation_test_weekly52" --device cuda --disable_wandb

echo "[pipeline] done. Checkpoints: ${S1_CKPT} , ${CURR_CKPT}"
echo "[pipeline] evaluation: ${CURR_RUN_DIR}/evaluation_test_weekly52"
