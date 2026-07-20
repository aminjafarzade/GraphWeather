#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

EXP_DIR="${EXP_DIR:-experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128}"
CONFIG="${CONFIG:-configs/weather_dual_resolution_l3_hidden128.yaml}"
CONFIG_NAME="${CONFIG_NAME:-raw_l3_hidden128}"
CHECKPOINT="${CHECKPOINT:-$EXP_DIR/best_ckpt.tar}"
RESOLUTION_MODE="${RESOLUTION_MODE:-2p5}"
DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-test}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-10}"

RUN_EVAL="${RUN_EVAL:-1}"
RUN_QUALITATIVE="${RUN_QUALITATIVE:-1}"
DRY_RUN="${DRY_RUN:-0}"

# RMSE/ACC defaults match the other 2.5-degree final-report evaluations.
EVAL_VARIABLES="${EVAL_VARIABLES:-z500 t2m t850 msl q700 u850}"
SELECTION="${SELECTION:-stride}"
STRIDE="${STRIDE:-7}"
N_INITIAL_CONDITIONS="${N_INITIAL_CONDITIONS:-52}"
START_TIMESTEP="${START_TIMESTEP:-1}"
CLIMATOLOGY_PATH="${CLIMATOLOGY_PATH:-data/stats/2p5_train_dayofyear_climatology.nc}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
CONFIDENCE_LEVEL="${CONFIDENCE_LEVEL:-0.95}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"
PLOT_CONFIDENCE_INTERVALS="${PLOT_CONFIDENCE_INTERVALS:-0}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-}"

# Qualitative map defaults.
AGGREGATE_MODE="${AGGREGATE_MODE:-sample}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
LEAD_TIMES="${LEAD_TIMES:-1 3 5 10}"
MAP_VARIABLES="${MAP_VARIABLES:-z500 t2m msl t850}"
MAX_BATCHES="${MAX_BATCHES:-}"
SAME_SCALE_ACROSS_LEADS="${SAME_SCALE_ACROSS_LEADS:-1}"
USE_CARTOPY="${USE_CARTOPY:-1}"
CONVERT_Z_TO_HEIGHT="${CONVERT_Z_TO_HEIGHT:-0}"
SAVE_ARRAYS="${SAVE_ARRAYS:-0}"
QUAL_OUTPUT_DIR="${QUAL_OUTPUT_DIR:-}"

read -r -a eval_variable_args <<< "${EVAL_VARIABLES//,/ }"
read -r -a lead_time_args <<< "${LEAD_TIMES//,/ }"
read -r -a map_variable_args <<< "${MAP_VARIABLES//,/ }"

if [[ -z "$EVAL_OUTPUT_DIR" ]]; then
  if [[ "$SELECTION" == "stride" && "$STRIDE" == "7" && "$N_INITIAL_CONDITIONS" == "52" ]]; then
    EVAL_OUTPUT_DIR="$EXP_DIR/evaluation_${SPLIT}_weekly52"
  else
    EVAL_OUTPUT_DIR="$EXP_DIR/evaluation_${SPLIT}_fixed${ROLLOUT_STEPS}"
  fi
fi

if [[ -z "$QUAL_OUTPUT_DIR" ]]; then
  if [[ "$AGGREGATE_MODE" == "sample" ]]; then
    sample_label="$(printf "sample%03d" "$SAMPLE_INDEX")"
    QUAL_OUTPUT_DIR="$EXP_DIR/visualizations_${SPLIT}_${sample_label}"
  else
    QUAL_OUTPUT_DIR="$EXP_DIR/visualizations_${SPLIT}_${AGGREGATE_MODE}"
  fi
fi

if [[ "$RUN_EVAL" != "0" ]]; then
  eval_cmd=(
    python scripts/evaluate.py
    --config "$CONFIG"
    --config_name "$CONFIG_NAME"
    --resolution_mode "$RESOLUTION_MODE"
    --checkpoint "$CHECKPOINT"
    --split "$SPLIT"
    --fixed_rollout_steps "$ROLLOUT_STEPS"
    --selection "$SELECTION"
    --include_persistence
    --output_dir "$EVAL_OUTPUT_DIR"
    --device "$DEVICE"
  )

  if ((${#eval_variable_args[@]})); then
    eval_cmd+=(--variables "${eval_variable_args[@]}")
  fi
  if [[ -n "$STRIDE" ]]; then
    eval_cmd+=(--stride "$STRIDE")
  fi
  if [[ -n "$N_INITIAL_CONDITIONS" ]]; then
    eval_cmd+=(--n_initial_conditions "$N_INITIAL_CONDITIONS")
  fi
  if [[ -n "$START_TIMESTEP" ]]; then
    eval_cmd+=(--start_timestep "$START_TIMESTEP")
  fi
  if [[ -n "$CLIMATOLOGY_PATH" ]]; then
    eval_cmd+=(--climatology_path "$CLIMATOLOGY_PATH")
  fi
  if [[ -n "$BOOTSTRAP_SAMPLES" && "$BOOTSTRAP_SAMPLES" != "0" ]]; then
    eval_cmd+=(
      --bootstrap_samples "$BOOTSTRAP_SAMPLES"
      --confidence_level "$CONFIDENCE_LEVEL"
      --bootstrap_seed "$BOOTSTRAP_SEED"
    )
  fi
  if [[ "$PLOT_CONFIDENCE_INTERVALS" != "0" ]]; then
    eval_cmd+=(--plot_confidence_intervals)
  fi

  echo "Running RMSE/ACC evaluation:"
  printf '  %q' "${eval_cmd[@]}"
  printf '\n'
  if [[ "$DRY_RUN" == "0" ]]; then
    "${eval_cmd[@]}"
  fi
fi

if [[ "$RUN_QUALITATIVE" != "0" ]]; then
  qualitative_cmd=(
    python scripts/visualize_rollout_maps.py
    --checkpoint "$CHECKPOINT"
    --config "$CONFIG"
    --config_name "$CONFIG_NAME"
    --resolution_mode "$RESOLUTION_MODE"
    --split "$SPLIT"
    --aggregate_mode "$AGGREGATE_MODE"
    --rollout_steps "$ROLLOUT_STEPS"
    --lead_times "${lead_time_args[@]}"
    --variables "${map_variable_args[@]}"
    --output_dir "$QUAL_OUTPUT_DIR"
    --device "$DEVICE"
  )

  if [[ "$AGGREGATE_MODE" == "sample" ]]; then
    qualitative_cmd+=(--sample_index "$SAMPLE_INDEX")
  fi
  if [[ "$AGGREGATE_MODE" == "year_mean" && -n "$MAX_BATCHES" ]]; then
    qualitative_cmd+=(--max_batches "$MAX_BATCHES")
  fi
  if [[ "$SAME_SCALE_ACROSS_LEADS" != "0" ]]; then
    qualitative_cmd+=(--same_scale_across_leads)
  fi
  if [[ "$USE_CARTOPY" == "0" ]]; then
    qualitative_cmd+=(--no_use_cartopy)
  fi
  if [[ "$CONVERT_Z_TO_HEIGHT" != "0" ]]; then
    qualitative_cmd+=(--convert_z_to_height)
  fi
  if [[ "$SAVE_ARRAYS" != "0" ]]; then
    qualitative_cmd+=(--save_arrays)
  fi

  echo "Running qualitative rollout maps:"
  printf '  %q' "${qualitative_cmd[@]}"
  printf '\n'
  if [[ "$DRY_RUN" == "0" ]]; then
    "${qualitative_cmd[@]}"
  fi
fi
