#!/usr/bin/env bash
# =============================================================================
# run_dense_l3k24_orogtisr_lossw_scratch.sh
#
# End-to-end pipeline for the from-scratch two-phase run
#   dense_l3k24_orogtisr_lossw_scratch_S1x100_S2toS10x3
#   (static forcings orog+TISR+pos, per-channel + lead-time loss weighting,
#    lr 1e-5 const; 100 epochs S1 pretrain -> 3 epochs/stage curriculum S2..S10)
#
# Stages: 1) TRAIN  2) EVAL (quantitative)  3) QUALITATIVE (bias maps)  4) DIAGNOSTICS
#
# -----------------------------------------------------------------------------
# CHOOSE YOUR GPU (this is the only thing you normally set):
#   GPU=2 bash scripts/run_dense_l3k24_orogtisr_lossw_scratch.sh
#   GPU=3 bash scripts/run_dense_l3k24_orogtisr_lossw_scratch.sh
#   ONLY nvidia-smi indices 2 and 3 (the L40S sm_89 cards) run this torch build.
#   Indices 0,1 are Blackwell sm_120 and will fail with a "no kernel" error.
#   CUDA_DEVICE_ORDER=PCI_BUS_ID is forced below so GPU matches nvidia-smi.
#
# The 127-epoch TRAIN stage is long -> run it detached, e.g.:
#   nohup env GPU=2 bash scripts/run_dense_l3k24_orogtisr_lossw_scratch.sh \
#     > runs/dense_l3k24_orogtisr_lossw_scratch_S1x100_S2toS10x3/pipeline.out 2>&1 &
#   # or inside tmux/screen.
#
# RUN ONLY SOME STAGES (e.g. eval/qual/diag after training finished elsewhere):
#   GPU=3 RUN_TRAIN=0 bash scripts/run_dense_l3k24_orogtisr_lossw_scratch.sh
#   GPU=2 RUN_TRAIN=1 RUN_EVAL=0 RUN_QUAL=0 RUN_DIAG=0 bash ...   # train only
# =============================================================================
set -euo pipefail

# --- GPU (your choice) -------------------------------------------------------
GPU="${GPU:-2}"                              # nvidia-smi index; 2 or 3 only
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # make CUDA_VISIBLE_DEVICES match nvidia-smi
export CUDA_VISIBLE_DEVICES="${GPU}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
PYTHON="${PYTHON:-/lustre/home/ziya/miniconda3/envs/graphweather-cu128/bin/python}"  # pinned: bare python is unsafe on sm_120

# --- locate repo root (works from anywhere) ----------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

# --- what to run -------------------------------------------------------------
CONFIG="configs/experiments/config_2p5_l3_hidden128_dense_l3k24_orogtisr_lossw_scratch_S1x100_S2toS10x3.yaml"
NAME="dense_l3k24_orogtisr_lossw_scratch_S1x100_S2toS10x3"
RUN_DIR="runs/${NAME}"
CKPT="${RUN_DIR}/best_ckpt.tar"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_QUAL="${RUN_QUAL:-1}"
RUN_DIAG="${RUN_DIAG:-1}"

# eval settings (weekly52 10-day rollout on test)
N_ICS="${N_ICS:-51}"
STRIDE="${STRIDE:-7}"
DIAG_BATCHES="${DIAG_BATCHES:-8}"

# --- preflight ---------------------------------------------------------------
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CONFIG}" >&2; exit 2
fi
# downstream stages need a checkpoint; if not training this run, require it now
_need_ckpt=0
[[ "${RUN_EVAL}" != "0" || "${RUN_QUAL}" != "0" || "${RUN_DIAG}" != "0" ]] && _need_ckpt=1
if [[ "${_need_ckpt}" == "1" && "${RUN_TRAIN}" == "0" && ! -f "${CKPT}" ]]; then
  echo "ERROR: eval/qual/diag enabled but checkpoint is missing: ${CKPT}" >&2
  echo "       Run with RUN_TRAIN=1 first, or wait for best_ckpt.tar to exist." >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/logs"
echo "############################################################"
echo "# run=${NAME}"
echo "# GPU=${GPU} (CUDA_DEVICE_ORDER=PCI_BUS_ID) | python=${PYTHON}"
echo "# stages: train=${RUN_TRAIN} eval=${RUN_EVAL} qual=${RUN_QUAL} diag=${RUN_DIAG}"
echo "############################################################"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader -i "${GPU}" 2>/dev/null || echo "(nvidia-smi -i ${GPU} unavailable)"

# --- 1) TRAIN  (127 epochs: 100 x S1 pretrain, then 3/stage S2..S10) ---------
if [[ "${RUN_TRAIN}" != "0" ]]; then
  echo; echo "=== [1/4] TRAIN  $(date '+%F %T') ==="
  "${PYTHON}" -u scripts/train.py \
    --config "${CONFIG}" --config_name "${NAME}" \
    --resolution_mode 2p5 --exp_dir runs --experiment_name "${NAME}" \
    2>&1 | tee "${RUN_DIR}/logs/train.log"
fi

# --- 2) EVAL  (quantitative RMSE/ACC, weekly52 10-day rollout) ----------------
if [[ "${RUN_EVAL}" != "0" ]]; then
  echo; echo "=== [2/4] EVAL  $(date '+%F %T') ==="
  "${PYTHON}" scripts/evaluate.py \
    --config "${CONFIG}" --config_name "${NAME}" --resolution_mode 2p5 \
    --checkpoint "${CKPT}" --split test --fixed_rollout_steps 10 \
    --selection stride --stride "${STRIDE}" --n_initial_conditions "${N_ICS}" \
    --start_timestep 1 --include_persistence \
    --variables z500 t2m t850 msl q700 u850 \
    --climatology_path data/stats/2p5_train_dayofyear_climatology.nc \
    --bootstrap_samples 1000 --confidence_level 0.95 --bootstrap_seed 42 \
    --output_dir "${RUN_DIR}/evaluation_test_weekly52" --device cuda --disable_wandb \
    2>&1 | tee "${RUN_DIR}/logs/eval.log"
fi

# --- 3) QUALITATIVE  (pred / ground-truth / bias maps) -----------------------
if [[ "${RUN_QUAL}" != "0" ]]; then
  echo; echo "=== [3/4] QUALITATIVE  $(date '+%F %T') ==="
  "${PYTHON}" scripts/visualize_rollout_maps.py \
    --checkpoint "${CKPT}" --config "${CONFIG}" --config_name "${NAME}" \
    --resolution_mode 2p5 --split test --aggregate_mode year_mean \
    --rollout_steps 10 --lead_times 1 3 5 10 --variables z500 t2m msl t850 \
    --same_scale_across_leads \
    --output_dir "${RUN_DIR}/visualizations_test_biasmaps" --device cuda \
    2>&1 | tee "${RUN_DIR}/logs/qualitative.log"
fi

# --- 4) DIAGNOSTICS  (oversmoothing/attention + new Dirichlet/graph/spectrum) -
# NOTE: run_diagnostics.py uses --config-name (HYPHEN), unlike the others.
if [[ "${RUN_DIAG}" != "0" ]]; then
  echo; echo "=== [4/4] DIAGNOSTICS  $(date '+%F %T') ==="
  "${PYTHON}" scripts/run_diagnostics.py \
    --config "${CONFIG}" --config-name "${NAME}" \
    --checkpoint "${CKPT}" --split test \
    --output "${RUN_DIR}/diagnostics_full_eval" --resolution_mode 2p5 \
    --max_full_diag_batches "${DIAG_BATCHES}" --device cuda --disable_wandb \
    2>&1 | tee "${RUN_DIR}/logs/diagnostics.log"
fi

echo; echo "############################################################"
echo "# DONE  $(date '+%F %T')"
echo "#   checkpoint : ${CKPT}"
echo "#   eval       : ${RUN_DIR}/evaluation_test_weekly52/S10/rollout_{rmse,acc}.csv"
echo "#   bias maps  : ${RUN_DIR}/visualizations_test_biasmaps/"
echo "#   diagnostics: ${RUN_DIR}/diagnostics_full_eval/ (graph_structure_metrics.csv, *_power_spectrum.csv, dirichlet cols)"
echo "############################################################"
