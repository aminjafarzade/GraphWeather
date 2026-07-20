#!/usr/bin/env bash
# =============================================================================
# launch_1p5_l4_150ep.sh
#
# Start both 1.5-degree L4 runs (150-epoch single-step base + S2->S10 curriculum
# at 3 epochs per horizon) in detached tmux sessions, one per GPU:
#
#   l4h160  -> GPU0  hidden 160, batch 4 x accum 3   (~47 h,  peak 77 GB at S10)
#   l4h128  -> GPU3  hidden 128, batch 2 x accum 6   (~59 h,  peak 31 GB at S10)
#
# Both have rollout checkpointing OFF and EMA on (decay 0.999). Effective batch
# is 12 in every stage, so the optimisation recipe matches the 100-epoch configs.
#
# Usage:
#   bash scripts/launch_1p5_l4_150ep.sh              # start both
#   ONLY=h160 bash scripts/launch_1p5_l4_150ep.sh    # start one (h160 | h128)
#   DRY_RUN=1 bash scripts/launch_1p5_l4_150ep.sh    # print what would happen
#
# Then:   tmux attach -t l4h160     (detach again with Ctrl-b then d)
#         tmux kill-session -t l4h160
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-both}"
PIPELINE="scripts/run_1p5_kai_l4_initckpt_pipeline.sh"

# session | gpu | variant | logfile
JOBS=(
  "l4h160|0|h160_150ep|runs/1p5_l4_h160_150ep.out"
  "l4h128|3|h128_150ep|runs/1p5_l4_h128_150ep.out"
)

command -v tmux >/dev/null || { echo "ERROR: tmux not found." >&2; exit 1; }
[[ -f "${PIPELINE}" ]] || { echo "ERROR: ${PIPELINE} not found." >&2; exit 1; }

started=()
for job in "${JOBS[@]}"; do
  IFS='|' read -r sess gpu variant log <<< "${job}"
  case "${ONLY}" in
    both) ;;
    h160) [[ "${sess}" == "l4h160" ]] || continue ;;
    h128) [[ "${sess}" == "l4h128" ]] || continue ;;
    *) echo "ERROR: ONLY must be h160, h128 or both (got '${ONLY}')." >&2; exit 2 ;;
  esac

  # Refuse to clobber a session that is already running this job.
  if tmux has-session -t "${sess}" 2>/dev/null; then
    echo "[skip] tmux session '${sess}' already exists — attach with: tmux attach -t ${sess}"
    continue
  fi

  # Warn (do not block) if the target GPU is already in use by anyone.
  used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)"
  if [[ "${used}" -gt 1000 ]]; then
    echo "[warn] GPU${gpu} already has ${used} MiB in use — ${sess} may run slowly or OOM."
  fi

  cmd="env GPU=${gpu} VARIANT=${variant} bash ${PIPELINE} 2>&1 | tee ${log}"
  echo "[start] ${sess}: GPU${gpu} VARIANT=${variant} -> ${log}"
  if [[ "${DRY_RUN}" != "0" ]]; then
    echo "        tmux new-session -d -s ${sess} -c ${REPO_ROOT} '${cmd}; exec bash'"
    continue
  fi
  # 'exec bash' keeps the pane alive after the job exits so the tail stays readable.
  tmux new-session -d -s "${sess}" -c "${REPO_ROOT}" "${cmd}; exec bash"
  started+=("${sess}")
done

echo
if [[ "${DRY_RUN}" != "0" ]]; then
  echo "DRY_RUN=1 — nothing was started."
  exit 0
fi
if [[ "${#started[@]}" -eq 0 ]]; then
  echo "Nothing new started."
else
  echo "Started: ${started[*]}"
fi
echo
echo "  attach   : tmux attach -t l4h160        (detach: Ctrl-b then d)"
echo "  progress : tail -f runs/1p5_l4_h160_150ep.out"
echo "  gpus     : nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"
echo "  stop     : tmux kill-session -t l4h160"
tmux ls 2>/dev/null | grep -E '^l4h(160|128):' || true
