"""Read-only RunRecord builder for gw-run/1.

Walks run directories under a runs root, assembles RunRecords from the
artifacts the training/eval pipeline already writes, and validates them
against the contract. Never writes into a run directory; never imports
from src/ (invariant I2).

CLI:  python -m dashboard.ingest --once --json [--root PATH]
"""
from __future__ import annotations

import argparse
import csv
import math
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import yaml

from .contract import (
    Architecture,
    DiagRecord,
    EvalRecord,
    EVAL_DIR_PRIORITY,
    Problem,
    QualRecord,
    RunRecord,
    SCHEMA_VERSION,
    has_error,
    sanitize_tree,
    utc_now_iso,
    validate_eval_payload,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"

# A run with no DONE marker whose newest artifact is older than this is
# surfaced with a stale_partial warning (training crashed or paused).
FRESH_SECONDS = 30 * 60

_DONE_MARKER = b"DONE rank"
_MAP_PNG_RE = re.compile(r"^(yearmean|sample)_(.+)_rollout_maps\.png$")
_DIAG_STEM_RE = re.compile(r"^epoch_(\d{4})_")
_FIXED_JSON_RE = re.compile(r"^fixed(\d+)_global_best_metrics\.json$")


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def discover_run_dirs(runs_root: Path) -> list[Path]:
    """A run = a direct child directory containing config_resolved.yaml.

    Everything else (stray files at the root, wandb/, the empty nested
    runs/<name>/<name>/ dirs) is ignored silently by construction.
    """
    if not runs_root.is_dir():
        return []
    return sorted(
        p for p in runs_root.iterdir()
        # skip symlinks: back-compat old->new run-name symlinks (P4.3) point at a
        # real run dir; ingesting both would double-count the same run.
        if p.is_dir() and not p.is_symlink() and (p / "config_resolved.yaml").is_file()
    )


# ---------------------------------------------------------------------------
# architecture (config_resolved.yaml + model_summary.txt)
# ---------------------------------------------------------------------------

def _read_params_millions(run_dir: Path) -> Optional[float]:
    path = run_dir / "model_summary.txt"
    if not path.is_file():
        return None
    m = re.search(r"trainable_parameters:\s*(\d+)", path.read_text(errors="replace"))
    return round(int(m.group(1)) / 1e6, 3) if m else None


def build_architecture(run_dir: Path, run_id: str) -> tuple[Architecture, list]:
    problems: list = []
    cfg_path = run_dir / "config_resolved.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text(errors="replace")) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"top-level YAML is {type(cfg).__name__}, expected mapping")
    except Exception as exc:  # noqa: BLE001 - any parse failure is the same Problem
        problems.append(Problem(
            run_id=run_id, eval_id=None, artifact="config_resolved_yaml",
            path=str(cfg_path), expected="parseable YAML mapping",
            found=f"{type(exc).__name__}: {exc}", reason="parse_error", severity="error",
        ))
        return Architecture(), problems

    th = cfg.get("target_handling") or {}
    model = cfg.get("model") or {}

    def pick(*keys, src=None):
        d = src if src is not None else cfg
        for k in keys:
            if isinstance(d, dict) and d.get(k) is not None:
                return d.get(k)
        return None

    arch = Architecture(
        resolution_mode=cfg.get("resolution_mode"),
        grid_shape=cfg.get("grid_shape"),
        level_shapes=cfg.get("level_shapes"),
        node_counts=cfg.get("node_counts"),
        edge_counts=cfg.get("edge_counts"),
        hidden_dim=pick("hidden_dim"),
        num_heads=pick("num_heads"),
        level_k_neighbors=cfg.get("level_k_neighbors"),
        graph_connectivity_strategy=cfg.get("graph_connectivity_strategy"),
        blocks={
            "encoder": pick("encoder_blocks"),
            "decoder": pick("decoder_blocks"),
            "l0": pick("l0_blocks", src=model) or cfg.get("l0_blocks"),
            "l1": pick("l1_blocks", src=model) or cfg.get("l1_blocks"),
            "l2": pick("l2_blocks", src=model) or cfg.get("l2_blocks"),
            "l3": pick("l3_blocks", src=model) or cfg.get("l3_blocks"),
            "l0_refine": pick("l0_refine_blocks", src=model) or cfg.get("l0_refine_blocks"),
            "l1_refine": pick("l1_refine_blocks", src=model) or cfg.get("l1_refine_blocks"),
            "l2_refine": pick("l2_refine_after_l3_blocks", src=model) or cfg.get("l2_refine_after_l3_blocks"),
        },
        params_millions=_read_params_millions(run_dir),
        lr_schedule={
            "type": cfg.get("lr_schedule_type"),
            "lr": cfg.get("lr"),
            "min_lr": cfg.get("min_lr"),
            "warmup_epochs": cfg.get("warmup_epochs"),
        },
        rollout={
            "schedule": cfg.get("rollout_schedule"),
            "stage_epochs": cfg.get("rollout_stage_epochs"),
            "max_epochs": cfg.get("max_epochs"),
        },
        forcings={
            "known_future_variables": th.get("known_future_variables") or [],
            "copy_variables": th.get("copy_variables") or [],
        },
        init_from_checkpoint=cfg.get("init_from_checkpoint"),
        tags=list((cfg.get("wandb") or {}).get("tags") or []),
    )
    return arch, problems


# ---------------------------------------------------------------------------
# evaluations
# ---------------------------------------------------------------------------

def _relpath(path: Path, run_dir: Path) -> str:
    return path.relative_to(run_dir).as_posix()


def _eval_artifacts(eval_dir: Path, run_dir: Path, horizon: Optional[int],
                    prefix_n: Optional[int], variables: list) -> dict:
    """Relpaths of the secondary artifacts next to the metrics JSON."""
    arts: dict = {"summary_csv": None, "summary_txt": None, "plots": {}, "s_dir": None, "log": None}
    n = prefix_n if prefix_n is not None else horizon
    if n is not None:
        for key, name in (("summary_csv", f"fixed{n}_global_best_summary.csv"),
                          ("summary_txt", f"fixed{n}_global_best_summary.txt")):
            p = eval_dir / name
            if p.is_file():
                arts[key] = _relpath(p, run_dir)
        s_dir = eval_dir / f"S{n}"
        if s_dir.is_dir():
            arts["s_dir"] = _relpath(s_dir, run_dir)
        plots_dir = eval_dir / "plots"
        if plots_dir.is_dir():
            for var in variables:
                for p in sorted(plots_dir.glob(f"fixed{n}_global_best_{var}_rmse_acc.*")):
                    arts["plots"][var] = _relpath(p, run_dir)
                    break
    log = eval_dir / "evaluation.log"
    if log.is_file():
        arts["log"] = _relpath(log, run_dir)
    return arts


def _stub_eval() -> dict:
    return {
        "horizon": None, "lead_times": [], "selection": {}, "climatology_path": None,
        "bootstrap": {}, "checkpoint": {}, "variables": [], "series": {},
        "baselines": {}, "units": {},
        "artifacts": {"summary_csv": None, "summary_txt": None, "plots": {},
                      "s_dir": None, "log": None},
    }


def build_eval_records(run_dir: Path, run_id: str) -> tuple[list, list]:
    problems: list = []
    records: list = []

    eval_dirs = sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("evaluation")
    )

    for eval_dir in eval_dirs:
        eval_id = eval_dir.name
        jsons = sorted(p for p in eval_dir.iterdir()
                       if p.is_file() and _FIXED_JSON_RE.match(p.name))
        if not jsons:
            # Eval dir without the metrics JSON: eval incomplete / in progress.
            # Absence is never invalidity (docs/03, "partial runs").
            problems.append(Problem(
                run_id=run_id, eval_id=eval_id, artifact="fixed_metrics_json",
                path=str(eval_dir),
                expected="fixed{N}_global_best_metrics.json in the evaluation dir",
                found="absent (evaluation incomplete or still running)",
                reason="missing_file", severity="warning",
            ))
            continue
        if len(jsons) > 1:
            problems.append(Problem(
                run_id=run_id, eval_id=eval_id, artifact="fixed_metrics_json",
                path=str(eval_dir),
                expected="exactly one fixed{N}_global_best_metrics.json",
                found=f"{len(jsons)}: {[p.name for p in jsons]}",
                reason="unexpected_value", severity="warning",
            ))
        json_path = jsons[0]
        prefix_n = int(_FIXED_JSON_RE.match(json_path.name).group(1))

        try:
            payload = json.loads(json_path.read_text(errors="replace"))
            if not isinstance(payload, dict):
                raise ValueError(f"top-level JSON is {type(payload).__name__}, expected object")
        except Exception as exc:  # noqa: BLE001
            problems.append(Problem(
                run_id=run_id, eval_id=eval_id, artifact="fixed_metrics_json",
                path=str(json_path), expected="parseable JSON object",
                found=f"{type(exc).__name__}: {exc}", reason="parse_error",
                severity="error",
            ))
            records.append(EvalRecord(eval_id=eval_id, is_primary=False, is_valid=False,
                                      **_stub_eval()))
            continue

        fields, eval_problems = validate_eval_payload(
            payload, run_id=run_id, eval_id=eval_id, json_path=str(json_path))
        problems.extend(eval_problems)

        if fields is None:
            records.append(EvalRecord(eval_id=eval_id, is_primary=False, is_valid=False,
                                      **_stub_eval()))
            continue

        fields.pop("_fatal", None)
        horizon = fields["horizon"]
        if prefix_n != horizon:
            problems.append(Problem(
                run_id=run_id, eval_id=eval_id, artifact="fixed_metrics_json",
                path=str(json_path),
                expected=f"filename prefix fixed{horizon} to match eval_fixed_rollout_steps",
                found=f"fixed{prefix_n}", reason="unexpected_value", severity="warning",
            ))
        s_dir = eval_dir / f"S{horizon}"
        if not s_dir.is_dir():
            problems.append(Problem(
                run_id=run_id, eval_id=eval_id, artifact="s_dir",
                path=str(s_dir), expected=f"S{horizon}/ consistent with horizon {horizon}",
                found="absent", reason="unexpected_value", severity="warning",
            ))

        is_valid = not has_error([p for p in eval_problems])
        fields["artifacts"] = _eval_artifacts(eval_dir, run_dir, horizon, prefix_n,
                                              fields["variables"])
        records.append(EvalRecord(eval_id=eval_id, is_primary=False,
                                  is_valid=is_valid, **fields))

    # exactly one primary among the records we actually built (Q1)
    if records:
        by_id = {r.eval_id: r for r in records}
        primary = next((by_id[name] for name in EVAL_DIR_PRIORITY if name in by_id), records[0])
        primary.is_primary = True

    return records, problems


# ---------------------------------------------------------------------------
# diagnostics (all sub-blocks optional; absence of conditional files is fine)
# ---------------------------------------------------------------------------

def _read_csv_rows(path: Path) -> list:
    with path.open(newline="", errors="replace") as f:
        return list(csv.DictReader(f))


def _maybe_float(v):
    """Coerce a CSV cell to float when it IS a number; keep strings verbatim.

    Diagnostics CSVs mix identifier columns (phase, layer_name, backend, …)
    with numeric ones — a robust reader must never assume a column's type.
    Non-finite numbers become None (never plotted as fake values).
    """
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return f if math.isfinite(f) else None


def _diag_problem(run_id: str, path: Path, exc: Exception) -> Problem:
    return Problem(
        run_id=run_id, eval_id=None, artifact="diagnostics",
        path=str(path), expected="parseable diagnostics CSV",
        found=f"{type(exc).__name__}: {exc}", reason="parse_error", severity="warning",
    )


def build_diagnostics(run_dir: Path, run_id: str) -> tuple[Optional[DiagRecord], list]:
    diag_dir = run_dir / "diagnostics_full_eval"
    if not diag_dir.is_dir():
        return None, []
    problems: list = []

    stems = set()
    for p in diag_dir.iterdir():
        m = _DIAG_STEM_RE.match(p.name)
        if m:
            stems.add(int(m.group(1)))
    epoch = max(stems) if stems else None
    stem = f"epoch_{epoch:04d}" if epoch is not None else None

    def find(name: str) -> Optional[Path]:
        if stem is None:
            return None
        for candidate in (diag_dir / f"{stem}_{name}", diag_dir / "tables" / f"{stem}_{name}"):
            if candidate.is_file():
                return candidate
        return None

    rollout_curve = None
    rc_path = find("rollout_curve.csv")
    if rc_path is not None:
        try:
            rows = [r for r in _read_csv_rows(rc_path) if r.get("phase") == "valid"]
            if rows:
                def _loss(r) -> Optional[float]:
                    v = _maybe_float(r.get("loss"))
                    return v if isinstance(v, float) else None

                by_h: dict = {}
                for r in rows:
                    h = _maybe_float(r.get("horizon"))
                    step = _maybe_float(r.get("step"))
                    if not isinstance(h, float) or not isinstance(step, float):
                        continue      # malformed row: skip it, keep the rest
                    by_h.setdefault(int(h), []).append((int(step), _loss(r)))
                if not by_h:
                    raise ValueError("no parseable valid-phase rows")
                # Prefer the largest horizon whose losses are all finite (an
                # S1-only run logs NaN for steps it never validated); this
                # selects real data, it never fills it in.
                finite_hs = [h for h, pts in by_h.items()
                             if all(l is not None for _, l in pts)]
                h = max(finite_hs) if finite_hs else max(by_h)
                pts = sorted(by_h[h])
                rollout_curve = {"steps": [s for s, _ in pts],
                                 "losses": [l for _, l in pts]}
        except Exception as exc:  # noqa: BLE001
            problems.append(_diag_problem(run_id, rc_path, exc))

    variance_ratio = None
    vr_path = find("variance_ratio.csv")
    if vr_path is not None:
        try:
            by_var: dict = {}
            for r in _read_csv_rows(vr_path):
                if r.get("phase") not in (None, "valid"):
                    continue
                lead = _maybe_float(r.get("lead"))
                val = _maybe_float(r.get("variance_ratio"))
                var = r.get("variable")
                if var is None or not isinstance(lead, float):
                    continue          # malformed row: skip it, keep the rest
                by_var.setdefault(var, {})[int(lead)] = val if isinstance(val, float) else None
            variance_ratio = {v: [d[k] for k in sorted(d)] for v, d in by_var.items()} or None
        except Exception as exc:  # noqa: BLE001
            problems.append(_diag_problem(run_id, vr_path, exc))

    power_spectrum = None
    ps_path = find("power_spectrum.csv")
    if ps_path is not None:
        try:
            rows = []
            for r in _read_csv_rows(ps_path):
                if r.get("phase") not in (None, "valid"):
                    continue
                if not isinstance(_maybe_float(r.get("lead")), float) or \
                        not isinstance(_maybe_float(r.get("wavenumber_bin")), float):
                    continue          # malformed row: skip it, keep the rest
                rows.append(r)
            by_var: dict = {}
            for r in rows:
                if r.get("variable"):
                    by_var.setdefault(r["variable"], []).append(r)
            spectra: dict = {}
            for var, rws in by_var.items():
                leads = sorted({int(r["lead"]) for r in rws})
                base_lead = leads[0]
                base = sorted((r for r in rws if int(r["lead"]) == base_lead),
                              key=lambda r: int(r["wavenumber_bin"]))
                ks = [int(float(r["wavenumber_bin"])) for r in base]
                truth = [_maybe_float(r.get("power_truth")) for r in base]
                truth = [t if isinstance(t, float) else None for t in truth]
                pred_by_lead: dict = {}
                for lead in leads:
                    pred = {}
                    for r in rws:
                        if int(float(r["lead"])) == lead:
                            v = _maybe_float(r.get("power_pred"))
                            pred[int(float(r["wavenumber_bin"]))] = v if isinstance(v, float) else None
                    pred_by_lead[str(lead)] = [pred.get(k) for k in ks]
                spectra[var] = {"wavenumbers": ks, "truth": truth,
                                "pred_by_lead": pred_by_lead}
            power_spectrum = spectra or None
        except Exception as exc:  # noqa: BLE001
            problems.append(_diag_problem(run_id, ps_path, exc))

    attention_table = None
    at_path = find("attention_metrics.csv")
    if at_path is not None:
        try:
            rows = _read_csv_rows(at_path)
            attention_table = [
                {k: _maybe_float(v) for k, v in r.items() if k}
                for r in rows
            ] or None
        except Exception as exc:  # noqa: BLE001
            problems.append(_diag_problem(run_id, at_path, exc))

    spectral_rmse = None
    if power_spectrum is None:
        sc_path = find("spectral_curve.csv")
        if sc_path is not None:
            try:
                rows = [r for r in _read_csv_rows(sc_path)
                        if r.get("phase") in (None, "valid")]
                if rows:
                    hmax = max(int(r.get("horizon", 0) or 0) for r in rows)
                    by_var: dict = {}
                    for r in rows:
                        if int(r.get("horizon", 0) or 0) != hmax:
                            continue
                        by_var.setdefault(r["variable"], {})[int(r["wavenumber_bin"])] =                             float(r["spectral_rmse"])
                    spectral_rmse = {
                        v: {"wavenumbers": sorted(d), "rmse": [d[k] for k in sorted(d)],
                            "horizon": hmax}
                        for v, d in by_var.items()
                    } or None
            except Exception as exc:  # noqa: BLE001
                problems.append(_diag_problem(run_id, sc_path, exc))

    def rel_if_exists(path: Optional[Path]) -> Optional[str]:
        return _relpath(path, run_dir) if path is not None and path.is_file() else None

    record = DiagRecord(
        epoch=epoch,
        rollout_curve=rollout_curve,
        variance_ratio=variance_ratio,
        power_spectrum=power_spectrum,
        attention=rel_if_exists(find("attention_metrics.csv")),
        scalars=rel_if_exists(find("scalars.json")),
        final_json=rel_if_exists(diag_dir / "final_full_diagnostics.json"
                                 if (diag_dir / "final_full_diagnostics.json").is_file() else None),
        attention_table=attention_table,
        spectral_rmse=spectral_rmse,
    )
    return record, problems


# ---------------------------------------------------------------------------
# qualitative maps
# ---------------------------------------------------------------------------

def build_qualitative(run_dir: Path, run_id: str) -> tuple[Optional[QualRecord], list]:
    qual_dir = run_dir / "visualizations_test_biasmaps"
    if not qual_dir.is_dir() and not (run_dir / "visualizations_dashboard").is_dir():
        return None, []
    problems: list = []

    metadata: dict = {}
    meta_path = qual_dir / "visualization_metadata.json"
    if qual_dir.is_dir() and meta_path.is_file():
        try:
            metadata = json.loads(meta_path.read_text(errors="replace"))
        except Exception as exc:  # noqa: BLE001
            problems.append(Problem(
                run_id=run_id, eval_id=None, artifact="visualization_metadata",
                path=str(meta_path), expected="parseable JSON",
                found=f"{type(exc).__name__}: {exc}", reason="parse_error",
                severity="warning",
            ))

    maps: dict = {}
    for p in sorted(qual_dir.glob("*.png")) if qual_dir.is_dir() else []:
        m = _MAP_PNG_RE.match(p.name)
        if m:
            var = m.group(2)
            # prefer yearmean over sample panels when both exist
            if var not in maps or m.group(1) == "yearmean":
                maps[var] = _relpath(p, run_dir)

    cm = qual_dir / "variable_channel_mapping.json"

    # per-(variable, day) maps generated offline by dashboard/generate_maps.py
    days: list = []
    by_day: dict = {}
    generated_at = None
    gen_dir = run_dir / "visualizations_dashboard"
    index_path = gen_dir / "maps_index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(errors="replace"))
            generated_at = index.get("generated_at")
            days = [int(d) for d in index.get("days", [])]
            for var, day_files in (index.get("files") or {}).items():
                for day, name in day_files.items():
                    p = gen_dir / name
                    if p.is_file():
                        by_day.setdefault(var, {})[str(int(day))] = _relpath(p, run_dir)
        except Exception as exc:  # noqa: BLE001
            problems.append(Problem(
                run_id=run_id, eval_id=None, artifact="maps_index",
                path=str(index_path), expected="parseable maps_index.json",
                found=f"{type(exc).__name__}: {exc}", reason="parse_error",
                severity="warning",
            ))

    record = QualRecord(
        metadata=sanitize_tree(metadata) if isinstance(metadata, dict) else {},
        maps=maps,
        channel_mapping=_relpath(cm, run_dir) if cm.is_file() else None,
        days=days,
        by_day=by_day,
        generated_at=generated_at,
    )
    return record, problems


# ---------------------------------------------------------------------------
# lifecycle + fingerprint
# ---------------------------------------------------------------------------

def _training_done(run_dir: Path) -> bool:
    out_log = run_dir / "out.log"
    if not out_log.is_file():
        return False
    try:
        return _DONE_MARKER in out_log.read_bytes()
    except OSError:
        return False


def _latest_activity(run_dir: Path) -> Optional[float]:
    times = []
    for name in ("last_ckpt.tar", "ckpt.tar", "out.log"):
        p = run_dir / name
        if p.is_file():
            try:
                times.append(p.stat().st_mtime)
            except OSError:
                pass
    return max(times) if times else None


def build_fingerprint(run_dir: Path) -> dict:
    fp: dict = {}

    def add(path: Path) -> None:
        try:
            st = path.stat()
        except OSError:
            return
        fp[_relpath(path, run_dir)] = [st.st_mtime_ns, st.st_size]

    for name in ("config_resolved.yaml", "out.log", "model_summary.txt"):
        p = run_dir / name
        if p.exists():
            add(p)
    for child in sorted(run_dir.iterdir()):
        if child.is_dir() and child.name.startswith("evaluation"):
            for p in child.iterdir():
                if p.is_file() and _FIXED_JSON_RE.match(p.name):
                    add(p)
        elif child.is_dir() and child.name in ("diagnostics_full_eval",
                                               "visualizations_test_biasmaps",
                                               "visualizations_dashboard"):
            add(child)
    return fp


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def compute_status(run_dir: Path, run_id: str, evaluations: list, problems: list,
                   *, now: Optional[float] = None,
                   fresh_seconds: int = FRESH_SECONDS) -> str:
    """Status per docs/02: evaluated > invalid > trained / in_progress.

    May append a stale_partial warning to ``problems`` (the contract's status
    enum has no "crashed"; the honest signal is in_progress + that warning).
    Reused by the scanner when the parse-error grace window rewrites a record.
    """
    now = time.time() if now is None else now
    if any(e.is_valid for e in evaluations):
        return "evaluated"
    if evaluations or has_error(problems):
        return "invalid"
    if _training_done(run_dir):
        return "trained"
    latest = _latest_activity(run_dir)
    if latest is not None and (now - latest) > fresh_seconds:
        problems.append(Problem(
            run_id=run_id, eval_id=None, artifact="out_log",
            path=str(run_dir / "out.log"),
            expected=f"'DONE rank 0' in out.log, or activity within {fresh_seconds}s",
            found=f"no DONE marker; last artifact activity {int(now - latest)}s ago",
            reason="stale_partial", severity="warning",
        ))
    return "in_progress"


def build_run_record(run_dir: Path, *, now: Optional[float] = None,
                     fresh_seconds: int = FRESH_SECONDS) -> RunRecord:
    run_id = run_dir.name
    now = time.time() if now is None else now
    problems: list = []

    architecture, arch_problems = build_architecture(run_dir, run_id)
    problems.extend(arch_problems)

    evaluations, eval_problems = build_eval_records(run_dir, run_id)
    problems.extend(eval_problems)

    diagnostics, diag_problems = build_diagnostics(run_dir, run_id)
    problems.extend(diag_problems)

    qualitative, qual_problems = build_qualitative(run_dir, run_id)
    problems.extend(qual_problems)

    status = compute_status(run_dir, run_id, evaluations, problems,
                            now=now, fresh_seconds=fresh_seconds)

    ts = utc_now_iso()
    return RunRecord(
        run_id=run_id,
        path=str(run_dir),
        status=status,
        discovered_at=ts,
        last_scanned_at=ts,
        fingerprint=build_fingerprint(run_dir),
        architecture=architecture,
        evaluations=evaluations,
        diagnostics=diagnostics,
        qualitative=qualitative,
        problems=problems,
    )


def ingest_all(runs_root: Path) -> list:
    return [build_run_record(d) for d in discover_run_dirs(runs_root)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="gw-dashboard read-only run ingest")
    parser.add_argument("--root", default=str(DEFAULT_RUNS_ROOT),
                        help="runs root to scan (default: repo runs/)")
    parser.add_argument("--once", action="store_true",
                        help="single sweep (the CLI never loops; flag kept for symmetry)")
    parser.add_argument("--json", action="store_true", help="dump records as JSON")
    parser.add_argument("--pretty", action="store_true", help="indent the JSON output")
    args = parser.parse_args(argv)

    records = ingest_all(Path(args.root))
    if args.json:
        payload = [r.to_dict() for r in records]
        json.dump(payload, sys.stdout, indent=2 if args.pretty else None, allow_nan=False)
        sys.stdout.write("\n")
    else:
        for r in records:
            evals = ",".join(f"{e.eval_id}{'*' if e.is_primary else ''}"
                             f"({'ok' if e.is_valid else 'INVALID'},H={e.horizon})"
                             for e in r.evaluations) or "-"
            errs = sum(1 for p in r.problems if p.severity == "error")
            warns = sum(1 for p in r.problems if p.severity == "warning")
            print(f"{r.run_id:60s} {r.status:12s} evals: {evals}  "
                  f"problems: {errs}E/{warns}W")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
