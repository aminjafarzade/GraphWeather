#!/usr/bin/env bash
set -euo pipefail

CLIM="${CLIM:-data/stats/2p5_train_dayofyear_climatology.nc}"

EXP=experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_orog_tisr_fixed
CFG=configs/weather_dual_resolution_l3_orog_tisr_fixed.yaml
python scripts/evaluate.py \
  --config "$CFG" \
  --resolution_mode 2p5 \
  --checkpoint "$EXP/best_ckpt.tar" \
  --split test \
  --fixed_rollout_steps 10 \
  --selection stride \
  --stride 7 \
  --variables z500 t2m t850 msl q700 u850 \
  --include_persistence \
  --climatology_path "$CLIM" \
  --bootstrap_samples 1000 \
  --confidence_level 0.95 \
  --bootstrap_seed 42 \
  --output_dir "$EXP/evaluation_test_weekly"

EXP=experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128_orog_tisr_fixed
CFG=configs/weather_dual_resolution_l3_hidden128_orog_tisr_fixed.yaml
python scripts/evaluate.py \
  --config "$CFG" \
  --resolution_mode 2p5 \
  --checkpoint "$EXP/best_ckpt.tar" \
  --split test \
  --fixed_rollout_steps 10 \
  --selection stride \
  --stride 7 \
  --variables z500 t2m t850 msl q700 u850 \
  --include_persistence \
  --climatology_path "$CLIM" \
  --bootstrap_samples 1000 \
  --confidence_level 0.95 \
  --bootstrap_seed 42 \
  --output_dir "$EXP/evaluation_test_weekly"

EXP=experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden160_orog_tisr_fixed
CFG=configs/weather_dual_resolution_l3_hidden160_orog_tisr_fixed.yaml
python scripts/evaluate.py \
  --config "$CFG" \
  --resolution_mode 2p5 \
  --checkpoint "$EXP/best_ckpt.tar" \
  --split test \
  --fixed_rollout_steps 10 \
  --selection stride \
  --stride 7 \
  --variables z500 t2m t850 msl q700 u850 \
  --include_persistence \
  --climatology_path "$CLIM" \
  --bootstrap_samples 1000 \
  --confidence_level 0.95 \
  --bootstrap_seed 42 \
  --output_dir "$EXP/evaluation_test_weekly"
