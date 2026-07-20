#!/usr/bin/env bash
set -euo pipefail

CLIM="${CLIM:-data/stats/2p5_train_dayofyear_climatology.nc}"
OUT_ROOT="${OUT_ROOT:-experiments/eval_target_override_comparisons}"

run_comparison() {
  local label="$1"
  local config="$2"
  local exp_dir="$3"
  local checkpoint="${exp_dir}/best_ckpt.tar"
  local out_dir="${OUT_ROOT}/${label}"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Skipping ${label}: missing ${checkpoint}"
    return 0
  fi

  python scripts/compare_eval_target_override.py \
    --config "${config}" \
    --resolution_mode 2p5 \
    --checkpoint "${checkpoint}" \
    --split test \
    --fixed_rollout_steps 10 \
    --selection stride \
    --stride 7 \
    --variables z500 t2m t850 msl q700 u850 \
    --include_persistence \
    --climatology_path "${CLIM}" \
    --bootstrap_samples 1000 \
    --confidence_level 0.95 \
    --bootstrap_seed 42 \
    --output_dir "${out_dir}"
}

run_comparison \
  "base_l3_hidden96" \
  "configs/weather_dual_resolution_l3.yaml" \
  "experiments/main_raw_2p5_b4_acc3_bf16_delta_warmup_cosine_l3"

run_comparison \
  "base_l3_hidden128" \
  "configs/weather_dual_resolution_l3_hidden128.yaml" \
  "experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128"

run_comparison \
  "heavy_l3_unet" \
  "configs/weather_dual_resolution_l3_heavy_unet.yaml" \
  "experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_heavy_unet_l0-3_l1-2_l2-2_l3-2_refine2"
