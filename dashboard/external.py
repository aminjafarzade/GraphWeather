"""External reference baselines (e.g. the KAI CSVs).

An external baseline is a long-format CSV. Two column layouts are accepted:

* curated   — ``variable,timestep,rmse,acc``      (experiments/kai_2.5.csv)
* raw eval  — ``lead_time,variable_idx,original_channel_idx,variable_name,rmse,acc``
              (experiments/kai_1.5.csv — a full per-channel evaluation dump)

Both are normalised to (variable, timestep, rmse, acc); the raw layout's
timestep-0 identity rows are dropped. A baseline carries a declared resolution
tag; overlaying it on a run of another resolution is allowed with a
``resolution_mismatch`` warning (decided, Q3 in docs/00).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
KAI_CSV = REPO_ROOT / "experiments" / "kai_2.5.csv"

# The curated 2.5° reference tracks these six headline variables; the 1.5°
# reference is restricted to the same set so the two baselines stay symmetric
# (and the raw dump's 60+ extra channels don't flood the per-variable tables).
REFERENCE_VARIABLES = ("z500", "t850", "t2m", "msl", "q700", "u850")


@dataclass
class ExternalBaseline:
    id: str
    label: str
    resolution: str            # resolution_mode tag, e.g. "2p5"
    path: str
    lead_times: list
    series: dict               # {var: {"rmse": [f x L], "acc": [f x L]}}
    variables: list = field(default_factory=list)

    def to_meta(self) -> dict:
        return {"id": self.id, "label": self.label, "resolution": self.resolution,
                "leads": self.lead_times, "variables": self.variables}


def _row_reader(reader: csv.DictReader):
    """Map either accepted CSV layout to (variable, timestep, rmse, acc) tuples.

    Yields nothing if the header matches neither layout, which makes
    ``load_external_csv`` return None (a format problem, never invented data).
    Timestep-0 rows (the identity step in raw eval dumps) are skipped.
    """
    fields = set(reader.fieldnames or [])
    if {"variable", "timestep", "rmse", "acc"} <= fields:
        var_key, t_key = "variable", "timestep"
    elif {"lead_time", "variable_name", "rmse", "acc"} <= fields:
        var_key, t_key = "variable_name", "lead_time"
    else:
        return
    for row in reader:
        t = int(row[t_key])
        if t <= 0:
            continue
        yield row[var_key].strip(), t, float(row["rmse"]), float(row["acc"])


def load_external_csv(path: Path, *, id: str, label: str, resolution: str,
                      only_variables: Optional[tuple] = None) -> Optional[ExternalBaseline]:
    """Parse a curated or raw-eval baseline CSV. Returns None if unreadable.

    ``only_variables`` restricts the baseline to that variable set (used to keep
    the raw per-channel dumps to the headline variables).
    """
    only = set(only_variables) if only_variables else None
    try:
        rmse: dict = {}
        acc: dict = {}
        with path.open(newline="", errors="replace") as f:
            for var, t, r, a in _row_reader(csv.DictReader(f)):
                if only is not None and var not in only:
                    continue
                rmse.setdefault(var, {})[t] = r if math.isfinite(r) else None
                acc.setdefault(var, {})[t] = a if math.isfinite(a) else None
    except (OSError, ValueError, KeyError):
        return None
    if not rmse:
        return None
    leads = sorted({t for d in rmse.values() for t in d})
    series = {
        var: {
            "rmse": [rmse[var].get(t) for t in leads],
            "acc": [acc.get(var, {}).get(t) for t in leads],
        }
        for var in rmse
    }
    return ExternalBaseline(id=id, label=label, resolution=resolution,
                            path=str(path), lead_times=leads, series=series,
                            variables=sorted(series.keys()))


def default_externals(repo_root: Optional[Path] = None) -> list:
    """The built-in registry. Extend via create_app(externals=[...]) (Q2/P4)."""
    root = repo_root or REPO_ROOT
    out = []
    kai = root / "experiments" / "kai_2.5.csv"
    if kai.is_file():
        ext = load_external_csv(kai, id="kai-2p5", label="KAI (2.5° reference)",
                                resolution="2p5")
        if ext is not None:
            out.append(ext)
    kai_1p5 = root / "experiments" / "kai_1.5.csv"
    if kai_1p5.is_file():
        ext = load_external_csv(kai_1p5, id="kai-1p5", label="KAI (1.5° reference)",
                                resolution="1p5", only_variables=REFERENCE_VARIABLES)
        if ext is not None:
            out.append(ext)
    return out
