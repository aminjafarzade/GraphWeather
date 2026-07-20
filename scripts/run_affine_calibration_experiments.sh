#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

COMMON_ARGS=(
  --resolution_mode 2p5
  --fit_split valid
  --eval_split test
  --fixed_rollout_steps 10
  --selection stride
  --stride 7
  --variables z500 t2m t850 msl q700 u850
  --climatology_path data/stats/2p5_train_dayofyear_climatology.nc
  --include_persistence
  --bootstrap_samples 1000
  --confidence_level 0.95
  --bootstrap_seed 42
  --device "$DEVICE"
)

EXP=experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128
CFG=configs/weather_dual_resolution_l3_hidden128.yaml

python scripts/fit_affine_calibration.py \
  --config "$CFG" \
  --checkpoint "$EXP/best_ckpt.tar" \
  "${COMMON_ARGS[@]}" \
  --output_dir "$EXP/affine_calibration"

H128_CALIBRATION_DIR="$EXP/affine_calibration"

# This repository's hidden160 fixed-orography run is configured by
# weather_dual_resolution_l3_hidden160.yaml (experiment_name ends in fixed_orog).
EXP=experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden160_fixed_orog
CFG=configs/weather_dual_resolution_l3_hidden160.yaml

python scripts/fit_affine_calibration.py \
  --config "$CFG" \
  --checkpoint "$EXP/best_ckpt.tar" \
  "${COMMON_ARGS[@]}" \
  --reference_comparison_dir "$H128_CALIBRATION_DIR" \
  --output_dir "$EXP/affine_calibration"
