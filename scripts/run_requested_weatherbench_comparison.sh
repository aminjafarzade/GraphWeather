#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
else
  DRY_RUN="${DRY_RUN:-0}"
fi

DEVICE="${DEVICE:-cuda}"
RESOLUTION_MODE="${RESOLUTION_MODE:-2p5}"
SPLIT="${SPLIT:-test}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-10}"
SELECTION="${SELECTION:-stride}"
STRIDE="${STRIDE:-7}"
N_INITIAL_CONDITIONS="${N_INITIAL_CONDITIONS:-52}"
START_TIMESTEP="${START_TIMESTEP:-1}"
CLIMATOLOGY_PATH="${CLIMATOLOGY_PATH:-data/stats/2p5_train_dayofyear_climatology.nc}"
RMSE_BACKEND="${RMSE_BACKEND:-weatherbench2}"
EXPERIMENT_SET="${EXPERIMENT_SET:-all}"

RUN_EVAL="${RUN_EVAL:-1}"
RUN_QUALITATIVE="${RUN_QUALITATIVE:-1}"
RUN_PLOT="${RUN_PLOT:-1}"
SKIP_EXISTING_EVAL="${SKIP_EXISTING_EVAL:-0}"
SKIP_EXISTING_QUALITATIVE="${SKIP_EXISTING_QUALITATIVE:-0}"

EVAL_VARIABLES="${EVAL_VARIABLES:-z500 t2m t850 msl q700 u850 u500}"
PLOT_VARIABLES="${PLOT_VARIABLES:-$EVAL_VARIABLES}"
MAP_VARIABLES="${MAP_VARIABLES:-z500 t2m msl t850 u500}"
LEAD_TIMES="${LEAD_TIMES:-1 3 5 10}"

BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
CONFIDENCE_LEVEL="${CONFIDENCE_LEVEL:-0.95}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"
PLOT_CONFIDENCE_INTERVALS="${PLOT_CONFIDENCE_INTERVALS:-0}"

AGGREGATE_MODE="${AGGREGATE_MODE:-sample}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
MAX_BATCHES="${MAX_BATCHES:-}"
SAME_SCALE_ACROSS_LEADS="${SAME_SCALE_ACROSS_LEADS:-1}"
USE_CARTOPY="${USE_CARTOPY:-1}"
CONVERT_Z_TO_HEIGHT="${CONVERT_Z_TO_HEIGHT:-0}"
SAVE_ARRAYS="${SAVE_ARRAYS:-0}"

KAI_CSV="${KAI_CSV:-experiments/kai_2.5.csv}"
KAI_LABEL="${KAI_LABEL:-kai 2p5}"
if [[ "$EXPERIMENT_SET" == "l3" || "$EXPERIMENT_SET" == "l3_only" ]]; then
  INCLUDE_KAI_IN_EVAL="${INCLUDE_KAI_IN_EVAL:-0}"
  INCLUDE_KAI_IN_PLOT="${INCLUDE_KAI_IN_PLOT:-0}"
else
  INCLUDE_KAI_IN_EVAL="${INCLUDE_KAI_IN_EVAL:-1}"
  INCLUDE_KAI_IN_PLOT="${INCLUDE_KAI_IN_PLOT:-1}"
fi

if [[ -z "${EVAL_OUTPUT_NAME:-}" ]]; then
  EVAL_OUTPUT_NAME="evaluation_${SPLIT}_weekly52_${RMSE_BACKEND}_comparison"
fi

if [[ -z "${QUAL_OUTPUT_NAME:-}" ]]; then
  if [[ "$AGGREGATE_MODE" == "sample" ]]; then
    QUAL_OUTPUT_NAME="$(printf "visualizations_%s_sample%03d_comparison" "$SPLIT" "$SAMPLE_INDEX")"
  else
    QUAL_OUTPUT_NAME="visualizations_${SPLIT}_${AGGREGATE_MODE}_comparison"
  fi
fi

if [[ -z "${COMPARISON_OUTPUT_DIR:-}" ]]; then
  if [[ "$EXPERIMENT_SET" == "l3" || "$EXPERIMENT_SET" == "l3_only" ]]; then
    COMPARISON_OUTPUT_DIR="experiments/comparison_l3_only_weatherbench_${RMSE_BACKEND}_rmse_acc"
  else
    COMPARISON_OUTPUT_DIR="experiments/comparison_requested_weatherbench_${RMSE_BACKEND}_rmse_acc"
  fi
fi

EXP_DIRS=(
  "experiments/main_raw_2p5_b4_acc3_bf16_delta_warmup_cosine_l3"
  "experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128"
  "experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden160_fixed_orog"
  "experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_heavy_unet_l0-3_l1-2_l2-2_l3-2_refine2"
  "experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_blocks3"
  "experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128_dense_l3k24_fixed_orog"
  "experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128_lead_conditioned_fixed_orog"
  "experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128_scalar_gated_pooling_fixed_orog"
  "experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128_scalar_gated_skip_fixed_orog"
  "experiments/main_raw_2p5_b4_acc3_bf16_delta_warmup_cosine_static_forcing"
)

LABELS=(
  "Base L3 96"
  "Base L3 128"
  "L3 160"
  "Heavy Unet L3 96"
  "Base L3 Dense L3 96"
  "Dense L3 128"
  "Lead Conditioned L3 128"
  "Gated Pooling L3 128"
  "L3 Gate Pooling 128"
  "L2 96"
)

CONFIGS=(
  "configs/weather_dual_resolution_l3.yaml"
  "configs/weather_dual_resolution_l3_hidden128.yaml"
  "configs/weather_dual_resolution_l3_hidden160.yaml"
  "configs/weather_dual_resolution_l3_heavy_unet.yaml"
  "configs/weather_dual_resolution_l3_blocks3.yaml"
  "configs/weather_dual_resolution_l3_hidden128_dense_l3k24_fixed_orog.yaml"
  "configs/weather_dual_resolution_l3_hidden128_lead_conditioned_fixed_orog.yaml"
  "configs/weather_dual_resolution_l3_hidden128_scalar_gated_pooling_fixed_orog.yaml"
  "configs/weather_dual_resolution_l3_hidden128_scalar_gated_skip_fixed_orog.yaml"
  "configs/weather_dual_resolution.yaml"
)

CONFIG_NAMES=(
  "raw_l3"
  "raw_l3_hidden128"
  "raw_l3_hidden160"
  "raw_l3_heavy_unet"
  "raw_l3_blocks3"
  "raw_l3_hidden128_dense_l3k24_fixed_orog"
  "raw_l3_hidden128_lead_conditioned_fixed_orog"
  "raw_l3_hidden128_scalar_gated_pooling_fixed_orog"
  "raw_l3_hidden128_scalar_gated_skip_fixed_orog"
  "raw_static_forcing"
)

case "$EXPERIMENT_SET" in
  all)
    ;;
  l3|l3_only)
    EXP_DIRS=("${EXP_DIRS[@]:0:9}")
    LABELS=("${LABELS[@]:0:9}")
    CONFIGS=("${CONFIGS[@]:0:9}")
    CONFIG_NAMES=("${CONFIG_NAMES[@]:0:9}")
    ;;
  *)
    echo "EXPERIMENT_SET must be one of: all, l3, l3_only" >&2
    exit 1
    ;;
esac

read -r -a EVAL_VARIABLE_ARGS <<< "${EVAL_VARIABLES//,/ }"
read -r -a PLOT_VARIABLE_ARGS <<< "${PLOT_VARIABLES//,/ }"
read -r -a MAP_VARIABLE_ARGS <<< "${MAP_VARIABLES//,/ }"
read -r -a LEAD_TIME_ARGS <<< "${LEAD_TIMES//,/ }"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

print_cmd() {
  printf '  %q' "$@"
  printf '\n'
}

run_or_print() {
  if [[ "$DRY_RUN" == "1" ]]; then
    print_cmd "$@"
  else
    "$@"
  fi
}

eval_complete() {
  local output_dir="$1"
  case "$RMSE_BACKEND" in
    weatherbench2)
      [[ -f "$output_dir/weatherbench2_rollout_rmse.csv" && -f "$output_dir/weatherbench2_rollout_acc.csv" ]]
      ;;
    current)
      [[ -f "$output_dir/S${ROLLOUT_STEPS}/rollout_rmse.csv" && -f "$output_dir/S${ROLLOUT_STEPS}/rollout_acc.csv" ]]
      ;;
    both)
      [[ -f "$output_dir/weatherbench2_rollout_rmse.csv" && -f "$output_dir/weatherbench2_rollout_acc.csv" && -f "$output_dir/S${ROLLOUT_STEPS}/rollout_rmse.csv" && -f "$output_dir/S${ROLLOUT_STEPS}/rollout_acc.csv" ]]
      ;;
    *)
      echo "RMSE_BACKEND must be one of: current, weatherbench2, both" >&2
      exit 1
      ;;
  esac
}

qualitative_complete() {
  local output_dir="$1"
  local prefix
  if [[ "$AGGREGATE_MODE" == "sample" ]]; then
    prefix="$(printf "sample%03d" "$SAMPLE_INDEX")"
  else
    prefix="yearmean"
  fi
  [[ -f "$output_dir/${prefix}_${MAP_VARIABLE_ARGS[0]}_rollout_maps.png" ]]
}

run_evaluation() {
  local index="$1"
  local exp_dir="${EXP_DIRS[$index]}"
  local label="${LABELS[$index]}"
  local cfg="${CONFIGS[$index]}"
  local config_name="${CONFIG_NAMES[$index]}"
  local checkpoint="$exp_dir/best_ckpt.tar"
  local output_dir="$exp_dir/$EVAL_OUTPUT_NAME"

  require_file "$cfg"
  require_file "$checkpoint"
  if [[ -n "$CLIMATOLOGY_PATH" ]]; then
    require_file "$CLIMATOLOGY_PATH"
  fi

  if [[ "$SKIP_EXISTING_EVAL" != "0" ]] && eval_complete "$output_dir"; then
    echo "Skipping existing evaluation: $label -> $output_dir"
    return
  fi

  local cmd=(
    python -u scripts/evaluate.py
    --config "$cfg"
    --config_name "$config_name"
    --resolution_mode "$RESOLUTION_MODE"
    --checkpoint "$checkpoint"
    --split "$SPLIT"
    --fixed_rollout_steps "$ROLLOUT_STEPS"
    --selection "$SELECTION"
    --variables "${EVAL_VARIABLE_ARGS[@]}"
    --include_persistence
    --rmse_backend "$RMSE_BACKEND"
    --output_dir "$output_dir"
  )

  if [[ -n "$STRIDE" ]]; then
    cmd+=(--stride "$STRIDE")
  fi
  if [[ -n "$N_INITIAL_CONDITIONS" ]]; then
    cmd+=(--n_initial_conditions "$N_INITIAL_CONDITIONS")
  fi
  if [[ -n "$START_TIMESTEP" ]]; then
    cmd+=(--start_timestep "$START_TIMESTEP")
  fi
  if [[ -n "$CLIMATOLOGY_PATH" ]]; then
    cmd+=(--climatology_path "$CLIMATOLOGY_PATH")
  fi
  if [[ -n "$DEVICE" ]]; then
    cmd+=(--device "$DEVICE")
  fi
  if [[ -n "$BOOTSTRAP_SAMPLES" && "$BOOTSTRAP_SAMPLES" != "0" ]]; then
    cmd+=(
      --bootstrap_samples "$BOOTSTRAP_SAMPLES"
      --confidence_level "$CONFIDENCE_LEVEL"
      --bootstrap_seed "$BOOTSTRAP_SEED"
    )
  fi
  if [[ "$PLOT_CONFIDENCE_INTERVALS" != "0" ]]; then
    cmd+=(--plot_confidence_intervals)
  fi
  if [[ "$INCLUDE_KAI_IN_EVAL" != "0" && -n "$KAI_CSV" ]]; then
    require_file "$KAI_CSV"
    cmd+=(--external_baseline_csv "$KAI_CSV" --external_baseline_label "$KAI_LABEL")
  fi

  echo
  echo "Evaluating $label"
  echo "  checkpoint: $checkpoint"
  echo "  output:     $output_dir"
  run_or_print "${cmd[@]}"
}

run_qualitative() {
  local index="$1"
  local exp_dir="${EXP_DIRS[$index]}"
  local label="${LABELS[$index]}"
  local cfg="${CONFIGS[$index]}"
  local config_name="${CONFIG_NAMES[$index]}"
  local checkpoint="$exp_dir/best_ckpt.tar"
  local output_dir="$exp_dir/$QUAL_OUTPUT_NAME"

  require_file "$cfg"
  require_file "$checkpoint"

  if [[ "$SKIP_EXISTING_QUALITATIVE" != "0" ]] && qualitative_complete "$output_dir"; then
    echo "Skipping existing qualitative maps: $label -> $output_dir"
    return
  fi

  local cmd=(
    python -u scripts/visualize_rollout_maps.py
    --checkpoint "$checkpoint"
    --config "$cfg"
    --config_name "$config_name"
    --resolution_mode "$RESOLUTION_MODE"
    --split "$SPLIT"
    --aggregate_mode "$AGGREGATE_MODE"
    --rollout_steps "$ROLLOUT_STEPS"
    --lead_times "${LEAD_TIME_ARGS[@]}"
    --variables "${MAP_VARIABLE_ARGS[@]}"
    --output_dir "$output_dir"
  )

  if [[ -n "$DEVICE" ]]; then
    cmd+=(--device "$DEVICE")
  fi
  if [[ "$AGGREGATE_MODE" == "sample" ]]; then
    cmd+=(--sample_index "$SAMPLE_INDEX")
  fi
  if [[ "$AGGREGATE_MODE" == "year_mean" && -n "$MAX_BATCHES" ]]; then
    cmd+=(--max_batches "$MAX_BATCHES")
  fi
  if [[ "$SAME_SCALE_ACROSS_LEADS" != "0" ]]; then
    cmd+=(--same_scale_across_leads)
  fi
  if [[ "$USE_CARTOPY" == "0" ]]; then
    cmd+=(--no_use_cartopy)
  fi
  if [[ "$CONVERT_Z_TO_HEIGHT" != "0" ]]; then
    cmd+=(--convert_z_to_height)
  fi
  if [[ "$SAVE_ARRAYS" != "0" ]]; then
    cmd+=(--save_arrays)
  fi

  echo
  echo "Writing qualitative maps for $label"
  echo "  output: $output_dir"
  run_or_print "${cmd[@]}"
}

run_combined_plot() {
  if [[ "$INCLUDE_KAI_IN_PLOT" != "0" && -n "$KAI_CSV" ]]; then
    require_file "$KAI_CSV"
  fi

  local eval_dirs=()
  local index
  for index in "${!EXP_DIRS[@]}"; do
    eval_dirs+=("${EXP_DIRS[$index]}/$EVAL_OUTPUT_NAME")
  done

  local cmd=(
    python -u scripts/plot_experiment_rmse_acc.py
    --experiments "${eval_dirs[@]}"
    --labels "${LABELS[@]}"
    --kai_label "$KAI_LABEL"
    --stage_dir "S${ROLLOUT_STEPS}"
    --variables "${PLOT_VARIABLE_ARGS[@]}"
    --output_dir "$COMPARISON_OUTPUT_DIR"
  )

  if [[ "$INCLUDE_KAI_IN_PLOT" != "0" && -n "$KAI_CSV" ]]; then
    cmd+=(--kai_csv "$KAI_CSV")
  else
    cmd+=(--kai_csv "")
  fi

  echo
  echo "Building combined RMSE/ACC plots"
  echo "  output: $COMPARISON_OUTPUT_DIR"
  run_or_print "${cmd[@]}"
}

if [[ "${#EXP_DIRS[@]}" -ne "${#LABELS[@]}" || "${#EXP_DIRS[@]}" -ne "${#CONFIGS[@]}" || "${#EXP_DIRS[@]}" -ne "${#CONFIG_NAMES[@]}" ]]; then
  echo "Experiment, label, config, and config-name arrays must have the same length." >&2
  exit 1
fi

for index in "${!EXP_DIRS[@]}"; do
  require_file "${EXP_DIRS[$index]}/best_ckpt.tar"
done

if [[ "$RUN_EVAL" != "0" ]]; then
  for index in "${!EXP_DIRS[@]}"; do
    run_evaluation "$index"
  done
fi

if [[ "$RUN_QUALITATIVE" != "0" ]]; then
  for index in "${!EXP_DIRS[@]}"; do
    run_qualitative "$index"
  done
fi

if [[ "$RUN_PLOT" != "0" ]]; then
  run_combined_plot
fi

echo
echo "Done."
echo "Combined quantitative plots: $COMPARISON_OUTPUT_DIR"
echo "Per-experiment qualitative maps: */$QUAL_OUTPUT_NAME"
