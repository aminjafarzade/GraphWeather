#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLIM="${CLIM:-$ROOT/data/stats/2p5_train_dayofyear_climatology.nc}"
KAI="${KAI:-$ROOT/experiments/kai_2.5.csv}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
CONFIDENCE_LEVEL="${CONFIDENCE_LEVEL:-0.95}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"
RMSE_BACKEND="${RMSE_BACKEND:-weatherbench2}"
OUTPUT_NAME="${OUTPUT_NAME:-evaluation_test_weekly_weatherbench2}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

run_eval() {
  local name="$1"
  local cfg="$2"
  local exp="$ROOT/experiments/$name"
  local output_dir="$exp/$OUTPUT_NAME"

  require_file "$cfg"
  require_file "$exp/best_ckpt.tar"
  require_file "$CLIM"
  require_file "$KAI"

  echo
  echo "============================================================"
  echo "Evaluating:  $name"
  echo "Config:      $cfg"
  echo "Checkpoint:  $exp/best_ckpt.tar"
  echo "Output dir:  $output_dir"
  echo "============================================================"

  local cmd=(
    python -u scripts/evaluate.py
    --config "$cfg"
    --resolution_mode 2p5
    --checkpoint "$exp/best_ckpt.tar"
    --split test
    --fixed_rollout_steps 10
    --selection stride
    --stride 7
    --variables z500 t2m t850 msl q700 u850
    --include_persistence
    --climatology_path "$CLIM"
    --bootstrap_samples "$BOOTSTRAP_SAMPLES"
    --confidence_level "$CONFIDENCE_LEVEL"
    --bootstrap_seed "$BOOTSTRAP_SEED"
    --rmse_backend "$RMSE_BACKEND"
    --external_baseline_csv "$KAI"
    --external_baseline_label "KAI 2.5"
    --output_dir "$output_dir"
  )

  if [[ -n "${DEVICE:-}" ]]; then
    cmd+=(--device "$DEVICE")
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf ' %q' "${cmd[@]}"
    echo
  else
    "${cmd[@]}"
  fi
}

run_eval \
  "main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128" \
  "$ROOT/configs/weather_dual_resolution_l3_hidden128.yaml"

run_eval \
  "main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128_dense_l3k24_fixed_orog" \
  "$ROOT/configs/weather_dual_resolution_l3_hidden128_dense_l3k24_fixed_orog.yaml"

run_eval \
  "main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128_lead_conditioned_fixed_orog" \
  "$ROOT/configs/weather_dual_resolution_l3_hidden128_lead_conditioned_fixed_orog.yaml"

run_eval \
  "main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128_scalar_gated_pooling_fixed_orog" \
  "$ROOT/configs/weather_dual_resolution_l3_hidden128_scalar_gated_pooling_fixed_orog.yaml"

run_eval \
  "main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128_scalar_gated_skip_fixed_orog" \
  "$ROOT/configs/weather_dual_resolution_l3_hidden128_scalar_gated_skip_fixed_orog.yaml"

echo
echo "Done. WeatherBench2/KAI outputs are under experiments/*/$OUTPUT_NAME"
