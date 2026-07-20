#!/usr/bin/env bash
set -euo pipefail

cd /lustre/home/ziya/GNN/GraphWeather5p625
mkdir -p logs

echo "=== Running L3 hidden128 BASE S1-only 100 epoch overfit probe ==="
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/experiments/config_2p5_l3_hidden128_base_S1_100epoch_overfit_probe.yaml \
  --config_name s1only_2p5_l3_hidden128_base_100epoch_overfit_probe \
  --resolution_mode 2p5 \
  2>&1 | tee logs/s1only_2p5_l3_hidden128_base_100epoch_overfit_probe.log

echo "=== Running L3 hidden128 L0-MLP S1-only 100 epoch overfit probe ==="
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/experiments/config_2p5_l3_hidden128_l0mlp_S1_100epoch_overfit_probe.yaml \
  --config_name s1only_2p5_l3_hidden128_l0mlp_100epoch_overfit_probe \
  --resolution_mode 2p5 \
  2>&1 | tee logs/s1only_2p5_l3_hidden128_l0mlp_100epoch_overfit_probe.log

echo "=== Finished 100-epoch S1-only overfit probes ==="
