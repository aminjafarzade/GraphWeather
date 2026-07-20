"""Backend comparison logic: overlay, best-per-variable, ranking, deltas.

This module is the ONLY place such logic exists (invariant I1) and it operates
exclusively on values already stored in the run artifacts (invariant I2).
Derived quantities (order, deltas, means over stored leads, threshold
crossings, lead intersections) are labeled ``derived`` in the responses.
Nothing here recomputes a metric, interpolates, or guesses a value: a run that
cannot be compared under a criterion is flagged ``not_comparable``.
"""
from __future__ import annotations

from typing import Optional

from .contract import EvalRecord, MetricSeries, RunRecord

METRICS = ("rmse", "acc")


# ---------------------------------------------------------------------------
# selection helpers
# ---------------------------------------------------------------------------

def select_eval(run: RunRecord, eval_choice: str = "primary") -> Optional[EvalRecord]:
    """Pick one *valid* evaluation of a run ('primary' or a specific eval_id)."""
    valid = [e for e in run.evaluations if e.is_valid]
    if not valid:
        return None
    if eval_choice in (None, "", "primary"):
        for e in valid:
            if e.is_primary:
                return e
        return valid[0]
    return next((e for e in valid if e.eval_id == eval_choice), None)


def _metric_series(ev: EvalRecord, variable: str, metric: str) -> Optional[MetricSeries]:
    vs = ev.series.get(variable)
    if vs is None:
        return None
    return getattr(vs, metric, None)


def _values_at(leads_wanted: list, lead_times: list, values: Optional[list]) -> Optional[list]:
    """Reindex stored values onto a lead subset. Only exact lead matches — a
    lead absent from the run's axis yields None, never an interpolation."""
    if values is None:
        return None
    index = {lt: i for i, lt in enumerate(lead_times)}
    return [values[index[l]] if l in index else None for l in leads_wanted]


def common_lead_times(lead_lists: list) -> tuple[list, Optional[dict]]:
    """Intersection of declared lead axes (invariant I4).

    Returns (common leads, horizon_truncated warning | None).
    """
    sets = [set(l) for l in lead_lists if l]
    if not sets:
        return [], None
    common = sorted(set.intersection(*sets))
    horizons = sorted({max(l) for l in lead_lists if l})
    if len(horizons) > 1:
        return common, {
            "type": "horizon_truncated",
            "detail": "compared series have different horizons; showing the "
                      "intersection of their declared lead times",
            "per_series_horizons": horizons,
            "common_max": common[-1] if common else None,
            "derived": True,
        }
    return common, None


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------

def overlay(runs: list, run_ids: list, variable: str, metric: str,
            *, eval_choice: str = "primary", include: Optional[list] = None,
            externals: Optional[list] = None) -> dict:
    include = include or []
    externals = {x.id: x for x in (externals or [])}
    by_id = {r.run_id: r for r in runs}
    warnings: list = []
    picked: list = []          # (run, eval)

    for rid in run_ids:
        run = by_id.get(rid)
        if run is None:
            warnings.append({"type": "run_skipped", "run_id": rid,
                             "detail": "unknown run id"})
            continue
        ev = select_eval(run, eval_choice)
        if ev is None:
            warnings.append({"type": "run_skipped", "run_id": rid,
                             "detail": f"no valid evaluation ({eval_choice!r})"})
            continue
        if _metric_series(ev, variable, metric) is None:
            warnings.append({"type": "run_skipped", "run_id": rid,
                             "detail": f"variable {variable!r} not in evaluation "
                                       f"{ev.eval_id!r}"})
            continue
        picked.append((run, ev))

    wanted_externals = []
    for token in include:
        if token.startswith("external:"):
            ext = externals.get(token.split(":", 1)[1])
            if ext is None:
                warnings.append({"type": "run_skipped", "run_id": token,
                                 "detail": "unknown external baseline"})
            elif variable not in ext.series:
                warnings.append({"type": "run_skipped", "run_id": ext.id,
                                 "detail": f"variable {variable!r} not in external "
                                           f"baseline"})
            else:
                wanted_externals.append(ext)

    lead_lists = [ev.lead_times for _, ev in picked] + \
                 [ext.lead_times for ext in wanted_externals]
    common, trunc = common_lead_times(lead_lists)
    if trunc:
        warnings.append(trunc)

    series = []
    for run, ev in picked:
        ms = _metric_series(ev, variable, metric)
        series.append({
            "run_id": run.run_id,
            "eval_id": ev.eval_id,
            "label": run.run_id,
            "lead_times": ev.lead_times,
            "values": _values_at(common, ev.lead_times, ms.mean),
            "ci_lower": _values_at(common, ev.lead_times, ms.ci_lower),
            "ci_upper": _values_at(common, ev.lead_times, ms.ci_upper),
        })

    baselines = []
    if "persistence" in include:
        seen_resolutions = set()
        for run, ev in picked:
            res = run.architecture.resolution_mode or "?"
            pers = ev.baselines.get("persistence", {}).get(variable)
            if pers is None or res in seen_resolutions:
                continue
            seen_resolutions.add(res)
            ms = pers[metric] if isinstance(pers, dict) else getattr(pers, metric)
            baselines.append({
                "id": "persistence",
                "label": f"persistence ({res})" if len(picked) > 1 else "persistence",
                "source_run": run.run_id,   # verbatim source, stated openly
                "values": _values_at(common, ev.lead_times, ms.mean),
            })

    if "persistence" in include and picked and \
            not any(b.get("id") == "persistence" for b in baselines):
        warnings.append({
            "type": "baseline_missing",
            "detail": "persistence was requested but is not stored in the "
                      "selected evaluation(s) — omitted, not fabricated",
        })

    run_resolutions = {run.architecture.resolution_mode for run, _ in picked
                       if run.architecture.resolution_mode}
    for ext in wanted_externals:
        mismatched = sorted(run_resolutions - {ext.resolution})
        if mismatched:
            warnings.append({
                "type": "resolution_mismatch",
                "detail": f"external baseline {ext.id!r} is {ext.resolution}; "
                          f"selection includes resolutions {mismatched}",
                "external": ext.id,
            })
        baselines.append({
            "id": ext.id,
            "label": ext.label,
            "external": True,
            "values": _values_at(common, ext.lead_times, ext.series[variable][metric]),
        })

    return {
        "variable": variable,
        "metric": metric,
        "lead_times": common,
        "series": series,
        "baselines": baselines,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# best per variable
# ---------------------------------------------------------------------------

def _ci_at(ms: MetricSeries, idx: int) -> Optional[list]:
    if ms.ci_lower is None or ms.ci_upper is None:
        return None
    lo, hi = ms.ci_lower[idx], ms.ci_upper[idx]
    if lo is None or hi is None:
        return None
    return [lo, hi]


def best_per_variable(runs: list, run_ids: list, *, lead: int, metric: str,
                      eval_choice: str = "primary",
                      externals: Optional[list] = None,
                      include: Optional[list] = None) -> dict:
    """argmin (rmse) / argmax (acc) over STORED values at one lead.

    External baselines named in ``include`` (tokens "external:<id>") compete as
    candidates too — a selected reference like kai-2p5 can win a variable.
    """
    by_id = {r.run_id: r for r in runs}
    lower_better = metric == "rmse"
    warnings: list = []
    out: dict = {"lead": lead, "metric": metric, "derived": True,
                 "variables": {}, "warnings": warnings}

    candidates: dict = {}
    run_resolutions: set = set()
    for rid in run_ids:
        run = by_id.get(rid)
        ev = select_eval(run, eval_choice) if run else None
        if ev is None or lead not in ev.lead_times:
            continue
        if run.architecture.resolution_mode:
            run_resolutions.add(run.architecture.resolution_mode)
        idx = ev.lead_times.index(lead)
        for var in ev.variables:
            ms = _metric_series(ev, var, metric)
            if ms is None or ms.mean[idx] is None:
                continue
            candidates.setdefault(var, []).append(
                {"value": ms.mean[idx], "id": rid, "ci": _ci_at(ms, idx),
                 "external": False})

    wanted = {t.split(":", 1)[1] for t in (include or []) if t.startswith("external:")}
    for ext in externals or []:
        if ext.id not in wanted or lead not in ext.lead_times:
            continue
        mismatched = sorted(run_resolutions - {ext.resolution})
        if mismatched:
            warnings.append({"type": "resolution_mismatch", "external": ext.id,
                             "detail": f"{ext.id} is {ext.resolution}; selection "
                                       f"includes {mismatched}"})
        idx = ext.lead_times.index(lead)
        for var, series in ext.series.items():
            v = (series.get(metric) or [None] * len(ext.lead_times))[idx]
            if v is None:
                continue
            candidates.setdefault(var, []).append(
                {"value": v, "id": ext.id, "ci": None, "external": True})

    for var, entries in candidates.items():
        entries.sort(key=lambda e: e["value"], reverse=not lower_better)
        best = entries[0]
        within = None
        if len(entries) > 1 and best["ci"] is not None and entries[1]["ci"] is not None:
            ci, ci2 = best["ci"], entries[1]["ci"]
            within = not (ci[1] < ci2[0] or ci2[1] < ci[0])  # stored CI overlap
        out["variables"][var] = {
            "run_id": best["id"], "value": best["value"], "ci": best["ci"],
            "external": best["external"],
            "within_ci_of_runner_up": within,
            "runner_up": entries[1]["id"] if len(entries) > 1 else None,
            "n_candidates": len(entries),
        }
    return out


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------

def _criterion_value(ev: EvalRecord, variable: str, metric: str,
                     lead: dict, common_leads: Optional[list] = None,
                     ) -> tuple[Optional[float], list]:
    """Value of one run under the criterion + flags. None => not comparable."""
    flags: list = []
    ms = _metric_series(ev, variable, metric)
    if ms is None:
        return None, ["not_comparable", "variable_missing"]

    kind = lead.get("type")
    if kind == "day":
        k = int(lead.get("k"))
        if k not in ev.lead_times:
            return None, ["not_comparable", "lead_outside_horizon"]
        v = ms.mean[ev.lead_times.index(k)]
        if v is None:
            return None, ["not_comparable", "stored_value_null"]
        return v, flags

    if kind == "mean_leads":
        # I4/audit rule: the mean runs over the COMMON leads of the compared
        # series, never each run's full axis (means over different horizons
        # would not be comparable).
        leads = common_leads if common_leads is not None else ev.lead_times
        idx = {lt: i for i, lt in enumerate(ev.lead_times)}
        vals = [ms.mean[idx[l]] for l in leads if l in idx]
        nn = [v for v in vals if v is not None]
        if not nn:
            return None, ["not_comparable", "stored_value_null"]
        flags.append("derived")
        if len(nn) < len(vals):
            flags.append("nulls_excluded")
        return sum(nn) / len(nn), flags

    if kind == "acc_crossing":
        threshold = float(lead.get("threshold", 0.6))
        acc = _metric_series(ev, variable, "acc")
        if acc is None:
            return None, ["not_comparable", "variable_missing"]
        flags.append("derived")
        for lt, v in zip(ev.lead_times, acc.mean):
            if v is not None and v < threshold:
                return float(lt), flags
        flags.append("never_crossed")
        return None, flags

    return None, ["not_comparable", "unknown_lead_type"]


def rank(runs: list, criterion: dict, *, externals: Optional[list] = None) -> dict:
    """The ONE ranking table. Order/deltas are derived; values are stored."""
    run_ids = criterion.get("runs") or []
    variable = criterion["variable"]
    metric = criterion.get("metric", "rmse")
    lead = criterion.get("lead") or {"type": "day", "k": None}
    eval_choice = criterion.get("eval", "primary")
    include_external = bool(criterion.get("include_external"))
    kind = lead.get("type")
    if kind == "acc_crossing":
        metric = "acc"
    lower_better = metric == "rmse" and kind != "acc_crossing"

    by_id = {r.run_id: r for r in runs}
    rows: list = []
    warnings: list = []
    lead_axes: list = []

    # resolve evaluations first so mean_leads can use the common lead axis
    resolved: dict = {}
    for rid in run_ids:
        run = by_id.get(rid)
        resolved[rid] = select_eval(run, eval_choice) if run else None
    axes_for_common = [e.lead_times for e in resolved.values() if e is not None]
    if include_external and kind == "mean_leads":
        axes_for_common += [x.lead_times for x in (externals or [])
                            if variable in x.series]
    common_ml, _trunc_ml = common_lead_times(axes_for_common)

    for rid in run_ids:
        ev = resolved.get(rid)
        if ev is None:
            rows.append({"run_id": rid, "value": None, "ci": None,
                         "flags": ["not_comparable", "no_valid_evaluation"]})
            continue
        lead_axes.append(ev.lead_times)
        value, flags = _criterion_value(ev, variable, metric, lead,
                                        common_leads=common_ml if kind == "mean_leads" else None)
        ci = None
        if kind == "day" and value is not None:
            ms = _metric_series(ev, variable, metric)
            ci = _ci_at(ms, ev.lead_times.index(int(lead["k"])))
        rows.append({"run_id": rid, "eval_id": ev.eval_id, "value": value,
                     "ci": ci, "flags": flags})

    if include_external:
        run_resolutions = {by_id[r].architecture.resolution_mode for r in run_ids
                           if r in by_id and by_id[r].architecture.resolution_mode}
        for ext in externals or []:
            flags = ["external"]
            value = None
            if variable in ext.series and kind in ("day", "mean_leads", "acc_crossing"):
                vals = ext.series[variable].get(metric)
                if kind == "day":
                    k = int(lead.get("k"))
                    if k in ext.lead_times:
                        value = vals[ext.lead_times.index(k)]
                    else:
                        flags += ["not_comparable", "lead_outside_horizon"]
                elif kind == "mean_leads":
                    idx = {lt: i for i, lt in enumerate(ext.lead_times)}
                    sub = [vals[idx[l]] for l in common_ml if l in idx]
                    nn = [v for v in sub if v is not None]
                    value = sum(nn) / len(nn) if nn else None
                    flags.append("derived")
                else:  # acc_crossing
                    threshold = float(lead.get("threshold", 0.6))
                    accs = ext.series[variable].get("acc") or []
                    flags.append("derived")
                    for lt, v in zip(ext.lead_times, accs):
                        if v is not None and v < threshold:
                            value = float(lt)
                            break
                    else:
                        flags.append("never_crossed")
            else:
                flags += ["not_comparable", "variable_missing"]
            mismatched = sorted(run_resolutions - {ext.resolution})
            if mismatched:
                flags.append("resolution_mismatch")
                warnings.append({"type": "resolution_mismatch", "external": ext.id,
                                 "detail": f"{ext.id} is {ext.resolution}; selection "
                                           f"includes {mismatched}"})
            rows.append({"run_id": ext.id, "label": ext.label, "value": value,
                         "ci": None, "flags": flags})

    lead_axes_common, trunc = common_lead_times(lead_axes)
    if trunc and kind == "mean_leads":
        warnings.append(trunc)  # means run over different horizons — say so

    def sort_key(row: dict):
        v = row["value"]
        never = "never_crossed" in row["flags"]
        if kind == "acc_crossing":
            # never crossing the threshold within the horizon is the best
            # outcome; then later crossings beat earlier ones.
            return (0, 0.0) if never else ((1, -v) if v is not None else (2, 0.0))
        if v is None:
            return (2, 0.0)
        return (1, v if lower_better else -v)

    rows.sort(key=sort_key)
    comparable = [r for r in rows if r["value"] is not None
                  or "never_crossed" in r["flags"]]
    best_value = next((r["value"] for r in comparable if r["value"] is not None), None)
    rank_n = 0
    for row in rows:
        if row["value"] is None and "never_crossed" not in row["flags"]:
            row["rank"] = None
            row["delta_vs_best_pct"] = None
            continue
        rank_n += 1
        row["rank"] = rank_n
        if row["value"] is None or best_value in (None, 0):
            row["delta_vs_best_pct"] = None
        else:
            row["delta_vs_best_pct"] = round(
                (row["value"] - best_value) / abs(best_value) * 100.0, 3)

    return {
        "criterion": {"runs": run_ids, "variable": variable, "metric": metric,
                      "lead": lead, "eval": eval_choice,
                      "include_external": include_external,
                      **({"leads_used": common_ml} if kind == "mean_leads" else {})},
        "derived": ["rank", "delta_vs_best_pct"] +
                   (["value"] if kind in ("mean_leads", "acc_crossing") else []),
        "rows": rows,
        "warnings": warnings,
    }
