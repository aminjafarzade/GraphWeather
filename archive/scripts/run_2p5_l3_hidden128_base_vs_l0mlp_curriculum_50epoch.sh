#!/usr/bin/env bash
set -euo pipefail

cd /lustre/home/ziya/GNN/GraphWeather5p625
mkdir -p logs

echo "=== Running L3 hidden128 BASE curriculum S1-to-S10 50-epoch experiment ==="
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/experiments/config_2p5_l3_hidden128_base_curriculum_S1toS10_50epoch.yaml \
  --config_name full_2p5_l3_hidden128_base_curriculum_S1toS10_50epoch \
  --resolution_mode 2p5 \
  2>&1 | tee logs/full_2p5_l3_hidden128_base_curriculum_S1toS10_50epoch.log

echo "=== Running L3 hidden128 L0-MLP curriculum S1-to-S10 50-epoch experiment ==="
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/experiments/config_2p5_l3_hidden128_l0mlp_curriculum_S1toS10_50epoch.yaml \
  --config_name full_2p5_l3_hidden128_l0mlp_curriculum_S1toS10_50epoch \
  --resolution_mode 2p5 \
  2>&1 | tee logs/full_2p5_l3_hidden128_l0mlp_curriculum_S1toS10_50epoch.log

echo "=== Finished both L3 hidden128 base vs L0-MLP curriculum experiments ==="
