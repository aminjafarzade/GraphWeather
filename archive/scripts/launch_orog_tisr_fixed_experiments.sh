#!/usr/bin/env bash
set -euo pipefail

python scripts/train.py \
  --config configs/weather_dual_resolution_l3_orog_tisr_fixed.yaml \
  --resolution_mode 2p5 \
  --experiment_name main_raw_2p5_b4_acc3_bf16_delta_l3_orog_tisr_fixed

python scripts/train.py \
  --config configs/weather_dual_resolution_l3_hidden128_orog_tisr_fixed.yaml \
  --resolution_mode 2p5 \
  --experiment_name main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128_orog_tisr_fixed

python scripts/train.py \
  --config configs/weather_dual_resolution_l3_hidden160_orog_tisr_fixed.yaml \
  --resolution_mode 2p5 \
  --experiment_name main_raw_2p5_b4_acc3_bf16_delta_l3_hidden160_orog_tisr_fixed
