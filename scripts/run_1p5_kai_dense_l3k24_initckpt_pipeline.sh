#!/usr/bin/env bash
set -euo pipefail

# 1.5-degree (121x240, poles included) two-stage training on the kai_1p5 dataset
# (/lustre/home/mahmed/Hydro/kai_1p5_data), initckpt-style:
#   1) S1-only 100-epoch base pretrain  (s1only_1p5_l3_hidden128_base_100epoch_dense_l3k24)
#   2) S2->S10 curriculum fine-tune warm-started from stage 1's best_ckpt.tar
#      (dense_l3k24_1p5_curriculum_S2toS10_3ep_initckpt, cosine 5e-5 -> 1e-6)
#   3) weekly test evaluation (2020, 10-day rollouts) after building the 1p5
#      day-of-year climatology from the train split.
#
# The dataset's channel order differs from era5_67 (levels-major, 250 hPa instead
# of 800 hPa) — all variable lookups are name-based at runtime and the delta
# stats npz (data/stats/kai_1p5_delta_stats.npz) is auto-computed positionally
# from THIS dataset on first run, so no manual reordering is needed anywhere.
# The graph graphs/graph_1p5_121x240_k8_l3k24_hybrid_row_aware_L3_v4.pt is
# auto-built from the training file's own lat/lon on first run.
#
# Usage (detached, GPU index as shown by nvidia-smi):
#   nohup env GPU=2 bash scripts/run_1p5_kai_dense_l3k24_initckpt_pipeline.sh \
#     > runs/1p5_kai_pipeline.out 2>&1 &
# Flags: FORCE_RETRAIN_S1=1 / FORCE_RETRAIN_CURR=1 rerun a finished stage;
#        RUN_EVAL=0 skips stage 3; PYTHON=<path> overrides the interpreter.

# --- GPU -----------------------------------------------------------------------
GPU="${GPU:-2}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
PYTHON="${PYTHON:-/lustre/home/ziya/miniconda3/envs/graphweather-cu128/bin/python}"

# --- locate repo root (works from anywhere) --------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

S1_CONFIG="configs/experiments/config_1p5_l3_hidden128_base_S1_100epoch_dense_l3k24.yaml"
S1_NAME="s1only_1p5_l3_hidden128_base_100epoch_dense_l3k24"
S1_RUN_DIR="runs/${S1_NAME}"
S1_CKPT="${S1_RUN_DIR}/best_ckpt.tar"

CURR_CONFIG="configs/experiments/config_1p5_l3_hidden128_dense_l3k24_curriculum_S2toS10_3ep_initckpt.yaml"
CURR_NAME="dense_l3k24_1p5_curriculum_S2toS10_3ep_initckpt"
CURR_RUN_DIR="runs/${CURR_NAME}"
CURR_CKPT="${CURR_RUN_DIR}/best_ckpt.tar"

CLIMATOLOGY="data/stats/kai_1p5_train_dayofyear_climatology.nc"
RUN_EVAL="${RUN_EVAL:-1}"

mkdir -p "${S1_RUN_DIR}/logs" "${CURR_RUN_DIR}/logs"

echo "[pipeline] GPU=${GPU} PYTHON=${PYTHON}"
echo "[pipeline] dataset: /lustre/home/mahmed/Hydro/kai_1p5_data (121x240, test=2020)"

# --- 1) TRAIN S1 base -------------------------------------------------------------
if [[ -f "${S1_CKPT}" && "${FORCE_RETRAIN_S1:-0}" != "1" ]]; then
  echo "[stage 1] ${S1_CKPT} exists — skipping S1 base training (FORCE_RETRAIN_S1=1 to redo)."
else
  echo "[stage 1] training S1 base: ${S1_NAME}"
  "${PYTHON}" -u scripts/train.py \
    --config "${S1_CONFIG}" --config_name "${S1_NAME}" \
    --resolution_mode 1p5 --exp_dir runs --experiment_name "${S1_NAME}" \
    2>&1 | tee "${S1_RUN_DIR}/logs/train.log"
fi

if [[ ! -f "${S1_CKPT}" ]]; then
  echo "[pipeline] ERROR: ${S1_CKPT} not found after stage 1 — aborting." >&2
  exit 1
fi

# --- 2) TRAIN S2->S10 curriculum (init_from_checkpoint is set in the YAML) --------
if [[ -f "${CURR_CKPT}" && "${FORCE_RETRAIN_CURR:-0}" != "1" ]]; then
  echo "[stage 2] ${CURR_CKPT} exists — skipping curriculum training (FORCE_RETRAIN_CURR=1 to redo)."
else
  echo "[stage 2] training curriculum: ${CURR_NAME} (warm start from ${S1_CKPT})"
  "${PYTHON}" -u scripts/train.py \
    --config "${CURR_CONFIG}" --config_name "${CURR_NAME}" \
    --resolution_mode 1p5 --exp_dir runs --experiment_name "${CURR_NAME}" \
    2>&1 | tee "${CURR_RUN_DIR}/logs/train.log"
fi

if [[ ! -f "${CURR_CKPT}" ]]; then
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
  "${PYTHON}" scripts/build_climatology.py \
    --config "${CURR_CONFIG}" --config_name "${CURR_NAME}" --resolution_mode 1p5 \
    --split train --output "${CLIMATOLOGY}" \
    2>&1 | tee "${CURR_RUN_DIR}/logs/climatology.log"
fi

echo "[stage 3] evaluating ${CURR_CKPT} on the 2020 test split"
"${PYTHON}" scripts/evaluate.py \
  --config "${CURR_CONFIG}" --config_name "${CURR_NAME}" --resolution_mode 1p5 \
  --checkpoint "${CURR_CKPT}" --split test --fixed_rollout_steps 10 \
  --selection stride --stride 7 --n_initial_conditions 52 \
  --start_timestep 1 --include_persistence \
  --variables z500 t2m t850 msl q700 u850 \
  --climatology_path "${CLIMATOLOGY}" \
  --bootstrap_samples 1000 --confidence_level 0.95 --bootstrap_seed 42 \
  --output_dir "${CURR_RUN_DIR}/evaluation_test_weekly52" --device cuda --disable_wandb \
  2>&1 | tee "${CURR_RUN_DIR}/logs/eval.log"

echo "[pipeline] done. Checkpoints: ${S1_CKPT} , ${CURR_CKPT}"
echo "[pipeline] evaluation: ${CURR_RUN_DIR}/evaluation_test_weekly52"
