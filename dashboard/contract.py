"""gw-run/1 data contract: dataclasses, validators, and the Problem object.

Everything a RunRecord contains is either read verbatim from an artifact or is
``None`` accompanied by a ``Problem`` entry (invariant I3). Validators in this
module are pure: they take parsed payloads and return (value, [Problem]) —
they never touch the filesystem. See dashboard/docs/02-DATA-CONTRACT.md.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

SCHEMA_VERSION = "gw-run/1"

# Discovery priority for the primary evaluation dir — mirrors
# scripts/plot_experiment_rmse_acc.py:36-42.
EVAL_DIR_PRIORITY = [
    "evaluation_test_weekly52",
    "evaluation_test_with_kai",
    "evaluation_test_allstarts",
    "evaluation_test",
    "evaluation",
]

# Units are NOT stored in the eval artifacts. This table is DECLARED, not
# derived, and /api/meta marks it as such (docs/01-GROUND-TRUTH.md).
DECLARED_UNITS = {
    "z500": "m²/s²",
    "t850": "K",
    "t2m": "K",
    "msl": "Pa",
    "q700": "kg/kg",
    "u850": "m/s",
    "u10": "m/s",
    "v10": "m/s",
}

# Top-level JSON keys the current evaluator emits (src/evaluator.py:3487-3510).
# Older files miss some of the optional set — that is schema drift (warning),
# never a failure.
JSON_KEYS_REQUIRED = ("lead_times", "checkpoints")
JSON_KEYS_OPTIONAL = (
    "eval_fixed_rollout_steps",
    "selection",
    "climatology",
    "bootstrap",
    "features",
    "evaluation_target_override",
    "orog_tisr_sensitivity",
    "affine_calibration",
    "external_baselines",
    "external_baseline_fairness_note",
)

PROBLEM_REASONS = frozenset({
    "missing_file",
    "parse_error",
    "axis_mismatch",
    "missing_key",
    "unexpected_value",
    "schema_drift",
    "stale_partial",
})

RUN_STATUSES = ("in_progress", "trained", "evaluated", "invalid", "removed")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Problem:
    """The structured format-error (invariant I3). Shown verbatim in the UI."""

    run_id: str
    eval_id: Optional[str]
    artifact: str
    path: str
    expected: str
    found: str
    reason: str
    severity: str  # "error" -> run/eval invalid; "warning" -> shown, usable
    detected_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.reason not in PROBLEM_REASONS:
            raise ValueError(f"unknown Problem reason: {self.reason!r}")
        if self.severity not in ("error", "warning"):
            raise ValueError(f"unknown Problem severity: {self.severity!r}")


@dataclass
class MetricSeries:
    mean: list  # floats or None (NaN passed through as null, never filled)
    ci_lower: Optional[list] = None
    ci_upper: Optional[list] = None


@dataclass
class VariableSeries:
    channel: Optional[int]
    variable_idx: Optional[int]
    rmse: MetricSeries
    acc: MetricSeries


@dataclass
class EvalRecord:
    eval_id: str
    is_primary: bool
    is_valid: bool
    horizon: Optional[int]
    lead_times: list
    selection: dict
    climatology_path: Optional[str]
    bootstrap: dict
    checkpoint: dict
    variables: list
    series: dict            # {var: VariableSeries}
    baselines: dict         # {"persistence": {var: {"rmse": MetricSeries, "acc": MetricSeries}}} | {}
    units: dict             # declared, per variable present
    artifacts: dict         # relpaths (relative to the run dir)


@dataclass
class DiagRecord:
    epoch: Optional[int]
    rollout_curve: Optional[dict]     # {"steps": [...], "losses": [...]}
    variance_ratio: Optional[dict]    # {var: [f x L]}
    power_spectrum: Optional[dict]    # {var: {wavenumbers, truth, pred_by_lead}}
    attention: Optional[str]
    scalars: Optional[str]
    final_json: Optional[str]
    # additive (still gw-run/1): parsed views so EVERY run with a diagnostics
    # dir shows something meaningful, incl. the older artifact generation
    attention_table: Optional[list] = None    # [{layer, entropy, ...}]
    spectral_rmse: Optional[dict] = None      # old format: {var: {wavenumbers, rmse}}


@dataclass
class QualRecord:
    metadata: dict
    maps: dict              # {var: relpath.png} (legacy multi-lead panels)
    channel_mapping: Optional[str]
    # additive: per-(variable, day) maps produced by dashboard/generate_maps.py
    days: list = field(default_factory=list)
    by_day: dict = field(default_factory=dict)   # {var: {"<day>": relpath.png}}
    generated_at: Optional[str] = None           # cache-busting stamp for the UI


@dataclass
class Architecture:
    resolution_mode: Optional[str] = None
    grid_shape: Optional[list] = None
    level_shapes: Optional[list] = None
    node_counts: Optional[list] = None
    edge_counts: Optional[list] = None
    hidden_dim: Optional[int] = None
    num_heads: Optional[int] = None
    level_k_neighbors: Optional[list] = None
    graph_connectivity_strategy: Optional[str] = None
    blocks: dict = field(default_factory=dict)
    params_millions: Optional[float] = None
    lr_schedule: dict = field(default_factory=dict)
    rollout: dict = field(default_factory=dict)
    forcings: dict = field(default_factory=dict)
    init_from_checkpoint: Optional[str] = None
    tags: list = field(default_factory=list)


@dataclass
class RunRecord:
    run_id: str
    path: str
    status: str
    discovered_at: str
    last_scanned_at: str
    fingerprint: dict
    architecture: Architecture
    evaluations: list       # [EvalRecord]
    diagnostics: Optional[DiagRecord]
    qualitative: Optional[QualRecord]
    problems: list          # [Problem]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in RUN_STATUSES:
            raise ValueError(f"unknown run status: {self.status!r}")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# validators (pure — no filesystem access)
# ---------------------------------------------------------------------------

def _is_bad_number(x: Any) -> bool:
    return isinstance(x, float) and (math.isnan(x) or math.isinf(x))


def sanitize_series(values: Any) -> tuple[Optional[list], int]:
    """Return (list with NaN/inf replaced by None, count replaced).

    Never interpolates; a bad value becomes null (invariant I3).
    """
    if not isinstance(values, list):
        return None, 0
    out, bad = [], 0
    for v in values:
        if _is_bad_number(v):
            out.append(None)
            bad += 1
        else:
            out.append(v)
    return out, bad


def sanitize_tree(obj: Any) -> Any:
    """Recursively replace NaN/inf with None in verbatim context payloads.

    Used for non-metric dicts we pass through as-is (selection, bootstrap,
    metadata): NaN is not representable in strict JSON, so it becomes null.
    Metric arrays are handled by sanitize_series with an explicit Problem.
    """
    if isinstance(obj, dict):
        return {k: sanitize_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_tree(v) for v in obj]
    if _is_bad_number(obj):
        return None
    return obj


def validate_metric_block(
    block: Any,
    horizon: int,
    *,
    run_id: str,
    eval_id: str,
    var: str,
    metric: str,
    json_path: str,
    severity_on_axis: str = "error",
) -> tuple[Optional[MetricSeries], list]:
    """Validate one {mean, ci_lower, ci_upper} block against the lead axis."""
    problems: list = []

    def problem(expected: str, found: str, reason: str, severity: str) -> None:
        problems.append(Problem(
            run_id=run_id, eval_id=eval_id, artifact="fixed_metrics_json",
            path=json_path, expected=expected, found=found,
            reason=reason, severity=severity,
        ))

    if not isinstance(block, dict) or "mean" not in block:
        problem(f"series.{var}.{metric}.mean present", "missing", "missing_key", severity_on_axis)
        return None, problems

    mean, bad = sanitize_series(block.get("mean"))
    if mean is None:
        problem(f"series.{var}.{metric}.mean is a list", type(block.get("mean")).__name__,
                "unexpected_value", severity_on_axis)
        return None, problems
    if len(mean) != horizon:
        problem(f"len(series.{var}.{metric}.mean) == lead_times length ({horizon})",
                f"{len(mean)} values", "axis_mismatch", severity_on_axis)
        return None, problems
    if bad:
        problem(f"series.{var}.{metric}.mean all finite",
                f"{bad} NaN/inf value(s) passed through as null", "unexpected_value", "warning")

    cis = {}
    for ci_key in ("ci_lower", "ci_upper"):
        raw = block.get(ci_key)
        if raw is None:
            cis[ci_key] = None
            continue
        ci, ci_bad = sanitize_series(raw)
        if ci is None or len(ci) != len(mean):
            problem(f"len(series.{var}.{metric}.{ci_key}) == len(mean) ({len(mean)})",
                    "wrong type" if ci is None else f"{len(ci)} values",
                    "axis_mismatch", "warning")
            cis[ci_key] = None
            continue
        if ci_bad:
            problem(f"series.{var}.{metric}.{ci_key} all finite",
                    f"{ci_bad} NaN/inf value(s) passed through as null",
                    "unexpected_value", "warning")
        cis[ci_key] = ci

    return MetricSeries(mean=mean, ci_lower=cis["ci_lower"], ci_upper=cis["ci_upper"]), problems


def validate_eval_payload(
    payload: dict,
    *,
    run_id: str,
    eval_id: str,
    json_path: str,
) -> tuple[Optional[dict], list]:
    """Validate the parsed fixed{N}_global_best_metrics.json payload.

    Returns (fields dict for EvalRecord construction | None if fatally invalid,
    [Problem]). Fatal = an error-severity Problem was emitted; the caller marks
    the eval invalid but still surfaces it.
    """
    problems: list = []

    def problem(expected: str, found: str, reason: str, severity: str,
                artifact: str = "fixed_metrics_json") -> None:
        problems.append(Problem(
            run_id=run_id, eval_id=eval_id, artifact=artifact, path=json_path,
            expected=expected, found=found, reason=reason, severity=severity,
        ))

    # -- schema drift on optional keys (older evaluator versions) ------------
    missing_optional = [k for k in JSON_KEYS_OPTIONAL if k not in payload]
    if missing_optional:
        problem(f"top-level keys {list(JSON_KEYS_OPTIONAL)}",
                f"missing: {missing_optional}", "schema_drift", "warning")

    # -- required: the lead axis ---------------------------------------------
    for k in JSON_KEYS_REQUIRED:
        if k not in payload:
            problem(f"top-level key '{k}' present", "missing", "missing_key", "error")
    if any(k not in payload for k in JSON_KEYS_REQUIRED):
        return None, problems

    lead_times = payload["lead_times"]
    if not isinstance(lead_times, list) or not lead_times:
        problem("lead_times is a non-empty list", repr(lead_times)[:80], "unexpected_value", "error")
        return None, problems

    declared_n = payload.get("eval_fixed_rollout_steps")
    if declared_n is None:
        # Older file: the axis itself is still declared data; its length is not
        # a guess. Drift already flagged above if the key set is short.
        horizon = len(lead_times)
    else:
        horizon = int(declared_n)
        if len(lead_times) != horizon:
            problem(f"len(lead_times) == eval_fixed_rollout_steps ({horizon})",
                    f"{len(lead_times)} lead values", "axis_mismatch", "error")
            return None, problems

    # -- required: global_best metrics ---------------------------------------
    checkpoints = payload.get("checkpoints") or {}
    global_best = checkpoints.get("global_best")
    if not isinstance(global_best, dict):
        problem("checkpoints.global_best present", "missing", "missing_key", "error")
        return None, problems
    metrics = global_best.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        problem("checkpoints.global_best.metrics non-empty", "missing or empty",
                "missing_key", "error")
        return None, problems

    series: dict = {}
    fatal = False
    for var, block in metrics.items():
        block = block if isinstance(block, dict) else {}
        rmse, p1 = validate_metric_block(block.get("rmse"), horizon, run_id=run_id,
                                         eval_id=eval_id, var=var, metric="rmse",
                                         json_path=json_path)
        acc, p2 = validate_metric_block(block.get("acc"), horizon, run_id=run_id,
                                        eval_id=eval_id, var=var, metric="acc",
                                        json_path=json_path)
        problems.extend(p1)
        problems.extend(p2)
        if rmse is None or acc is None:
            fatal = True
            continue
        series[var] = VariableSeries(
            channel=block.get("channel"),
            variable_idx=block.get("variable_idx"),
            rmse=rmse, acc=acc,
        )
    if not series:
        problem("at least one valid variable series", "none parsed", "missing_key", "error")
        return None, problems

    # -- optional: persistence baseline (omitted + warning if malformed) -----
    baselines: dict = {}
    persistence = checkpoints.get("persistence")
    if isinstance(persistence, dict) and isinstance(persistence.get("metrics"), dict):
        pers_series: dict = {}
        for var, block in persistence["metrics"].items():
            block = block if isinstance(block, dict) else {}
            rmse, p1 = validate_metric_block(block.get("rmse"), horizon, run_id=run_id,
                                             eval_id=eval_id, var=var,
                                             metric="persistence.rmse",
                                             json_path=json_path,
                                             severity_on_axis="warning")
            acc, p2 = validate_metric_block(block.get("acc"), horizon, run_id=run_id,
                                            eval_id=eval_id, var=var,
                                            metric="persistence.acc",
                                            json_path=json_path,
                                            severity_on_axis="warning")
            problems.extend(p1)
            problems.extend(p2)
            if rmse is not None and acc is not None:
                pers_series[var] = {"rmse": rmse, "acc": acc}
        if pers_series:
            baselines["persistence"] = pers_series

    selection = sanitize_tree(payload.get("selection")) if isinstance(payload.get("selection"), dict) else {}
    bootstrap = sanitize_tree(payload.get("bootstrap")) if isinstance(payload.get("bootstrap"), dict) else {}
    climatology = payload.get("climatology")
    climatology_path = None
    if isinstance(climatology, dict):
        climatology_path = climatology.get("climatology_path") or climatology.get("path")
    elif isinstance(climatology, str):
        climatology_path = climatology

    fields = {
        "horizon": horizon,
        "lead_times": lead_times,
        "selection": selection,
        "climatology_path": climatology_path,
        "bootstrap": bootstrap,
        "checkpoint": sanitize_tree({
            "label": global_best.get("label"),
            "epoch": global_best.get("epoch"),
            "train_rollout_steps": global_best.get("train_rollout_steps"),
            "params_m": global_best.get("params_m"),
        }),
        "variables": list(series.keys()),
        "series": series,
        "baselines": baselines,
        "units": {v: DECLARED_UNITS.get(v, "") for v in series.keys()},
        "_fatal": fatal,
    }
    return fields, problems


def has_error(problems: list) -> bool:
    return any(p.severity == "error" for p in problems)
