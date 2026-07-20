#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

SPLIT="${SPLIT:-valid}"
DEVICE="${DEVICE:-}"
MAX_FULL_DIAG_BATCHES="${MAX_FULL_DIAG_BATCHES:-}"

runs=(
  "/lustre/home/ziya/GNN/GraphWeather5p625/experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_hidden128_scalar_gated_skip_fixed_orog|/lustre/home/ziya/GNN/GraphWeather5p625/configs/weather_dual_resolution_l3_hidden128_scalar_gated_skip_fixed_orog.yaml"
)

for item in "${runs[@]}"; do
  IFS="|" read -r exp cfg <<< "${item}"
  checkpoint="${exp}/best_ckpt.tar"
  output="${exp}/diagnostics_full_eval"

  if [[ ! -f "${cfg}" ]]; then
    echo "Missing config: ${cfg}" >&2
    exit 1
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi

  cmd=(
    python scripts/run_diagnostics.py
    --config "${cfg}"
    --resolution_mode 2p5
    --checkpoint "${checkpoint}"
    --split "${SPLIT}"
    --output "${output}"
    --disable_wandb
  )

  if [[ -n "${DEVICE}" ]]; then
    cmd+=(--device "${DEVICE}")
  fi
  if [[ -n "${MAX_FULL_DIAG_BATCHES}" ]]; then
    cmd+=(--max_full_diag_batches "${MAX_FULL_DIAG_BATCHES}")
  fi

  echo "Running diagnostics for ${exp}"
  echo "Output: ${output}"
  "${cmd[@]}"
done

echo "All diagnostics completed."
