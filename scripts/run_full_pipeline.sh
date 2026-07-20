#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh — BACK-COMPAT SHIM (P2 consolidation)
#
# The pipeline body now lives in scripts/run_pipeline.sh. This shim preserves the
# historical env-var interface (CONFIG, CONFIG_NAME, RUN_TRAIN, RUN_EVAL,
# RUN_PLOT, RUN_QUALITATIVE, RUN_DIAGNOSTICS, COMPARISON_*, DRY_RUN, ...) so
# existing callers keep working unchanged.
#
# Two intentional differences from the old script:
#   - the interpreter is pinned to the graphweather-cu128 env (bare `python` was
#     unsafe on the sm_120 box); override with PYTHON=<path>.
#   - the new graph/clim/dashmaps stages are kept OFF here, so the stage set is
#     exactly the historical train,eval,plot,qual,diag.
#
# New work should call scripts/run_pipeline.sh directly (flag interface).
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export RUN_GRAPH="${RUN_GRAPH:-0}"
export RUN_CLIM="${RUN_CLIM:-0}"
export RUN_DASHMAPS="${RUN_DASHMAPS:-0}"

exec bash "${SCRIPT_DIR}/run_pipeline.sh" "$@"
