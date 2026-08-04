#!/usr/bin/env bash
# =============================================================================
# run_2p5_hidden160_arms.sh — the 5 single-variable arms on the 2p5
# hidden-160 dense-L3K24 100-epoch recipe.
#
# Each arm = S1 100 epochs from scratch -> S2..S10 curriculum at 3 epochs each,
# warm-started from THAT ARM'S OWN S1 checkpoint. Every arm differs from the
# control in exactly one dimension.
#
#   G  lossw_varup    loss_channel_weighting.variable_upweights {t2m:2, z500:2}
#   H  pool_nomax     pooling.mean_type=area_weighted, include_max=false
#   C  lossw_itv      loss_channel_weighting.inverse_tendency_variance (GraphCast s_j)
#   D  edge_gate      model.edge_encoding rbf/harmonics/gate/bias_mlp
#   E  boundary_mlp   model.boundary_mlp + head_init_std
#   A  wd1e2_notisr   weight_decay 1.0e-2
#   B  lossw_floor02_notisr  loss_channel_weighting floor 0.2 @ reference_level 1000
#   --  notisr        the NEW CONTROL / baseline (forcing-correct orog+tisr)
#
# Usage:
#   bash scripts/run_2p5_hidden160_arms.sh                 # list the arms
#   bash scripts/run_2p5_hidden160_arms.sh edge_gate       # one arm
#   ARM=edge_gate GPU=2 bash scripts/run_2p5_hidden160_arms.sh
#   bash scripts/run_2p5_hidden160_arms.sh all             # all five, sequentially
#   DRY_RUN=1 bash scripts/run_2p5_hidden160_arms.sh all   # print, execute nothing
#
# One arm per GPU is the intended parallel mode -- launch each in its own tmux:
#   tmux new-session -d -s armD -c "$PWD" \
#     'GPU=2 bash scripts/run_2p5_hidden160_arms.sh edge_gate; exec bash'
#
# Notes:
#   * clim is deliberately NOT in --stages: data/stats/2p5_train_dayofyear_climatology.nc
#     already exists and must stay fixed so every arm and the control are scored
#     identically.
#   * Goes straight to run_pipeline.sh, never run_full_pipeline.sh (which exports
#     RUN_DASHMAPS=0 as a default and would drop the dashmaps stage).
#   * run_pipeline.sh does NOT derive init_from_checkpoint from --s1-name; each
#     arm's curriculum config names its own S1 run explicitly.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

ARMS=(notisr lossw_varup pool_nomax lossw_itv edge_gate boundary_mlp wd1e2_notisr lossw_floor02_notisr)

# per-arm default GPU; override with GPU=<n> (applies to every arm you launch)
declare -A ARM_GPU=(
  [notisr]=0
  [lossw_varup]=1
  [pool_nomax]=2
  [lossw_itv]=3
  [edge_gate]=0
  [boundary_mlp]=1
  [wd1e2_notisr]=2
  [lossw_floor02_notisr]=3
)

# The control curriculum run every arm is plotted against. This is the REAL
# directory: runs/ also carries a back-compat symlink under the old name
# (dense_l3k24_hidden160_curriculum_S2toS10_3ep_initckpt).
COMPARE_RUN="${COMPARE_RUN:-runs/2p5_l3_h160_densel3k24_currS2toS10x3_initckpt_notisr}"
COMPARE_LABELS="${COMPARE_LABELS:-control_h160_notisr}"

STAGES="${STAGES:-train,eval,plot,qual,diag,dashmaps}"

usage() {
  sed -n '2,36p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  printf 'Available arms: %s\n' "${ARMS[*]}"
}

run_arm() {
  local arm="$1"
  local s1_cfg="configs/experiments/config_2p5_l3_hidden160_base_S1_100epoch_dense_l3k24_${arm}.yaml"
  local cu_cfg="configs/experiments/config_2p5_l3_hidden160_dense_l3k24_curriculum_S2toS10_3ep_initckpt_${arm}.yaml"
  local s1_name="2p5_l3_h160_densel3k24_s1x100_${arm}"
  local cu_name="2p5_l3_h160_densel3k24_currS2toS10x3_initckpt_${arm}"
  local gpu="${GPU:-${ARM_GPU[$arm]}}"

  [[ -f "${s1_cfg}" ]] || { echo "ERROR: missing ${s1_cfg}" >&2; return 2; }
  [[ -f "${cu_cfg}" ]] || { echo "ERROR: missing ${cu_cfg}" >&2; return 2; }

  echo "############################################################"
  echo "# ARM ${arm}   gpu=${gpu}"
  echo "#   S1   : ${s1_name}"
  echo "#   CURR : ${cu_name}"
  echo "############################################################"

  local -a cmd=(
    bash "${SCRIPT_DIR}/run_pipeline.sh"
    --resolution 2p5
    --s1-config "${s1_cfg}"   --s1-name   "${s1_name}"
    --curr-config "${cu_cfg}" --curr-name "${cu_name}"
    --primary-label "h160_${arm}"
    --stages "${STAGES}"
    --gpu "${gpu}"
  )
  # the new control IS the baseline, so it has nothing to compare against
  if [[ "${arm}" != "notisr" ]]; then
    cmd+=(--compare "${COMPARE_RUN}" --compare-labels "${COMPARE_LABELS}")
  fi
  [[ "${DRY_RUN:-0}" != "0" ]] && cmd+=(--dry-run)
  "${cmd[@]}"
}

TARGET="${1:-${ARM:-}}"

if [[ -z "${TARGET}" || "${TARGET}" == "-h" || "${TARGET}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${TARGET}" == "all" ]]; then
  for arm in "${ARMS[@]}"; do run_arm "${arm}"; done
  exit 0
fi

for arm in "${ARMS[@]}"; do
  if [[ "${arm}" == "${TARGET}" ]]; then run_arm "${arm}"; exit $?; fi
done

echo "ERROR: unknown arm '${TARGET}'. Available: ${ARMS[*]}" >&2
exit 2
