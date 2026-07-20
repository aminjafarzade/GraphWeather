#!/usr/bin/env bash
set -euo pipefail

cd /lustre/home/ziya/GNN/GraphWeather5p625

DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-test}"
AGGREGATE_MODE="${AGGREGATE_MODE:-sample}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-10}"
LEAD_TIMES="${LEAD_TIMES:-1 3 5 10}"
VARIABLES="${VARIABLES:-z500 t2m msl t850}"
MAX_BATCHES="${MAX_BATCHES:-}"

read -r -a LEAD_TIME_ARGS <<< "$LEAD_TIMES"
read -r -a VARIABLE_ARGS <<< "$VARIABLES"

if [[ "$AGGREGATE_MODE" == "sample" ]]; then
  sample_label=$(printf "sample%03d" "$SAMPLE_INDEX")
  output_suffix="visualizations_${SPLIT}_${sample_label}"
else
  output_suffix="visualizations_${SPLIT}_${AGGREGATE_MODE}"
fi

run_visualization() {
  local exp_dir="$1"
  local config_path="$2"
  local output_dir="$exp_dir/$output_suffix"

  local cmd=(
    python scripts/visualize_rollout_maps.py
    --checkpoint "$exp_dir/best_ckpt.tar"
    --config "$config_path"
    --resolution_mode 2p5
    --split "$SPLIT"
    --aggregate_mode "$AGGREGATE_MODE"
    --rollout_steps "$ROLLOUT_STEPS"
    --lead_times "${LEAD_TIME_ARGS[@]}"
    --variables "${VARIABLE_ARGS[@]}"
    --output_dir "$output_dir"
    --device "$DEVICE"
  )

  if [[ "$AGGREGATE_MODE" == "sample" ]]; then
    cmd+=(--sample_index "$SAMPLE_INDEX")
  fi
  if [[ "$AGGREGATE_MODE" == "year_mean" && -n "$MAX_BATCHES" ]]; then
    cmd+=(--max_batches "$MAX_BATCHES")
  fi

  echo "Running qualitative visualization: $exp_dir"
  "${cmd[@]}"
}

run_visualization \
  experiments/main_raw_2p5_b4_acc3_bf16_delta_warmup_cosine_l3 \
  configs/weather_dual_resolution_l3.yaml

run_visualization \
  experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_stage_warmup_cosine \
  configs/weather_dual_resolution_l3_stage_warmup_cosine.yaml
