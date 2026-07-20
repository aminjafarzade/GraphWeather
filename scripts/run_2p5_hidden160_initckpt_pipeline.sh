#!/usr/bin/env bash
set -euo pipefail

# hidden-160 twin of the winning 2p5 initckpt recipe:
#   1) S1-only 100-epoch base pretrain (hidden 160, 5 heads)
#   2) S2->S10 curriculum x3 epochs, cosine 5e-5 -> 1e-6, warm start from stage 1
#   3) weekly52 test evaluation (2018, 10-day rollouts, persistence + bootstrap)
#   4) comparison plots vs initckpt / flat_lr5e7 / orogtisr-scratch + KAI
#   5) qualitative bias maps (year-mean panels)
#   6) full diagnostics (valid split)
#   7) per-day dashboard maps (dashboard/generate_maps.py)
#
# Usage (detached):
#   nohup env GPU=3 bash scripts/run_2p5_hidden160_initckpt_pipeline.sh \
#     > runs/2p5_hidden160_pipeline.out 2>&1 &
# Flags: FORCE_RETRAIN_S1=1 / FORCE_RETRAIN_CURR=1 rerun training;
#        RUN_EVAL/RUN_PLOT/RUN_QUAL/RUN_DIAG/RUN_DASHMAPS=0 skip a stage.

GPU="${GPU:-3}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
PYTHON="${PYTHON:-/lustre/home/ziya/miniconda3/envs/graphweather-cu128/bin/python}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

S1_CONFIG="configs/experiments/config_2p5_l3_hidden160_base_S1_100epoch_dense_l3k24.yaml"
S1_NAME="s1only_2p5_l3_hidden160_base_100epoch_dense_l3k24"
S1_RUN_DIR="runs/${S1_NAME}"
S1_CKPT="${S1_RUN_DIR}/best_ckpt.tar"

CURR_CONFIG="configs/experiments/config_2p5_l3_hidden160_dense_l3k24_curriculum_S2toS10_3ep_initckpt.yaml"
CURR_NAME="dense_l3k24_hidden160_curriculum_S2toS10_3ep_initckpt"
CURR_RUN_DIR="runs/${CURR_NAME}"
CURR_CKPT="${CURR_RUN_DIR}/best_ckpt.tar"

CLIMATOLOGY="data/stats/2p5_train_dayofyear_climatology.nc"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PLOT="${RUN_PLOT:-1}"
RUN_QUAL="${RUN_QUAL:-1}"
RUN_DIAG="${RUN_DIAG:-1}"
RUN_DASHMAPS="${RUN_DASHMAPS:-1}"
COMPARE_RUNS=(
  "runs/dense_l3k24_curriculum_S2toS10_3ep_initckpt"
  "runs/dense_l3k24_curriculum_S2toS10_3ep_flat_lr5e7"
  "runs/dense_l3k24_orogtisr_lossw_scratch_S1x100_S2toS10x3"
)

mkdir -p "${S1_RUN_DIR}/logs" "${CURR_RUN_DIR}/logs"
echo "[pipeline] GPU=${GPU} PYTHON=${PYTHON} (hidden 160, 2p5)"

if [[ -f "${S1_CKPT}" && "${FORCE_RETRAIN_S1:-0}" != "1" ]]; then
  echo "[stage 1] ${S1_CKPT} exists — skipping S1 base training."
else
  echo "[stage 1] training S1 base: ${S1_NAME}"
  "${PYTHON}" -u scripts/train.py \
    --config "${S1_CONFIG}" --config_name "${S1_NAME}" \
    --resolution_mode 2p5 --exp_dir runs --experiment_name "${S1_NAME}" \
    2>&1 | tee "${S1_RUN_DIR}/logs/train.log"
fi

[[ -f "${S1_CKPT}" ]] || { echo "[pipeline] ERROR: ${S1_CKPT} missing after stage 1" >&2; exit 1; }

if [[ -f "${CURR_CKPT}" && "${FORCE_RETRAIN_CURR:-0}" != "1" ]]; then
  echo "[stage 2] ${CURR_CKPT} exists — skipping curriculum training."
else
  echo "[stage 2] training curriculum: ${CURR_NAME} (warm start from ${S1_CKPT})"
  "${PYTHON}" -u scripts/train.py \
    --config "${CURR_CONFIG}" --config_name "${CURR_NAME}" \
    --resolution_mode 2p5 --exp_dir runs --experiment_name "${CURR_NAME}" \
    2>&1 | tee "${CURR_RUN_DIR}/logs/train.log"
fi

[[ -f "${CURR_CKPT}" ]] || { echo "[pipeline] ERROR: ${CURR_CKPT} missing after stage 2" >&2; exit 1; }

if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "[stage 3] RUN_EVAL=${RUN_EVAL} — skipping evaluation."
else
echo "[stage 3] evaluating ${CURR_CKPT} on the 2018 test split (weekly52)"
"${PYTHON}" scripts/evaluate.py \
  --config "${CURR_CONFIG}" --config_name "${CURR_NAME}" --resolution_mode 2p5 \
  --checkpoint "${CURR_CKPT}" --split test --fixed_rollout_steps 10 \
  --selection stride --stride 7 --n_initial_conditions 52 \
  --start_timestep 1 --include_persistence \
  --variables z500 t2m t850 msl q700 u850 \
  --climatology_path "${CLIMATOLOGY}" \
  --bootstrap_samples 1000 --confidence_level 0.95 --bootstrap_seed 42 \
  --output_dir "${CURR_RUN_DIR}/evaluation_test_weekly52" --device cuda --disable_wandb \
  2>&1 | tee "${CURR_RUN_DIR}/logs/eval.log"
fi

if [[ "${RUN_PLOT}" == "1" ]]; then
  echo "[stage 4] comparison plots vs baselines + KAI"
  "${PYTHON}" scripts/plot_experiment_rmse_acc.py \
    --experiments "${CURR_RUN_DIR}" "${COMPARE_RUNS[@]}" \
    --labels hidden160 initckpt flat_lr5e7 orogtisr_scratch \
    --kai_csv experiments/kai_2.5.csv --stage_dir S10 \
    --variables z500 t2m t850 msl q700 u850 \
    --output_dir "comparisons/${CURR_NAME}" --plot_format png \
    2>&1 | tee "${CURR_RUN_DIR}/logs/plot.log"
fi

if [[ "${RUN_QUAL}" == "1" ]]; then
  echo "[stage 5] qualitative bias maps (year-mean panels)"
  "${PYTHON}" scripts/visualize_rollout_maps.py \
    --checkpoint "${CURR_CKPT}" \
    --config "${CURR_CONFIG}" --config_name "${CURR_NAME}" --resolution_mode 2p5 \
    --split test --aggregate_mode year_mean --rollout_steps 10 \
    --lead_times 1 3 5 10 --variables z500 t2m t850 msl \
    --output_dir "${CURR_RUN_DIR}/visualizations_test_biasmaps" \
    --device cuda --same_scale_across_leads \
    2>&1 | tee "${CURR_RUN_DIR}/logs/qualitative.log"
fi

if [[ "${RUN_DIAG}" == "1" ]]; then
  echo "[stage 6] full diagnostics (valid split)"
  "${PYTHON}" scripts/run_diagnostics.py \
    --config "${CURR_CONFIG}" --config-name "${CURR_NAME}" \
    --checkpoint "${CURR_CKPT}" --split valid \
    --output "${CURR_RUN_DIR}/diagnostics_full_eval" \
    --resolution_mode 2p5 --disable_wandb --device cuda \
    2>&1 | tee "${CURR_RUN_DIR}/logs/diagnostics.log"
fi

if [[ "${RUN_DASHMAPS}" == "1" ]]; then
  echo "[stage 7] per-day dashboard maps"
  "${PYTHON}" -m dashboard.generate_maps --run "${CURR_NAME}" \
    2>&1 | tee "${CURR_RUN_DIR}/logs/dashmaps.log"
fi

echo "[pipeline] ALL DONE:"
echo "  checkpoints: ${S1_CKPT} , ${CURR_CKPT}"
echo "  evaluation:  ${CURR_RUN_DIR}/evaluation_test_weekly52"
echo "  comparison:  comparisons/${CURR_NAME}"
echo "  qualitative: ${CURR_RUN_DIR}/visualizations_test_biasmaps + visualizations_dashboard"
echo "  diagnostics: ${CURR_RUN_DIR}/diagnostics_full_eval"
