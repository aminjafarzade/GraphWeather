from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Path(os.environ.setdefault("XDG_CACHE_HOME", "/tmp")).mkdir(parents=True, exist_ok=True)
Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_EXPERIMENTS = [
    PROJECT_ROOT / "experiments" / "main_raw_2p5_b4_acc3_bf16_delta_l3_stage_warmup_cosine",
    PROJECT_ROOT / "experiments" / "main_raw_2p5_b4_acc3_bf16_delta_warmup_cosine_l3",
    PROJECT_ROOT / "experiments" / "main_raw_2p5_b4_acc3_bf16_delta_warmup_cosine_static_forcing",
]
DEFAULT_LABELS = [
    "L3 stage warmup cosine",
    "L3 warmup cosine",
    "Static forcing",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "comparison_l3_stage_static_rmse_acc"
DEFAULT_KAI_CSV = PROJECT_ROOT / "data" / "baselines" / "kai_2p5.csv"
DEFAULT_VARIABLES = ["z500", "t2m", "t850", "msl", "q700", "u850"]
EVAL_DIR_NAMES = [
    "evaluation_test_weekly52",
    "evaluation_test_with_kai",
    "evaluation_test_allstarts",
    "evaluation_test",
    "evaluation",
]


@dataclass(frozen=True)
class EvaluationSource:
    label: str
    requested_path: Path
    eval_dir: Path
    rollout_dir: Path
    data: dict[str, Any]


def _safe_stem(value: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return stem.strip("_") or "variable"


def _parse_items(values: list[str] | None) -> list[str]:
    if not values:
        return []
    items: list[str] = []
    for value in values:
        items.extend(chunk.strip() for chunk in value.replace(",", " ").split() if chunk.strip())
    return items


def _has_rollout_files(path: Path) -> bool:
    return (path / "rollout_rmse.csv").is_file() and (path / "rollout_acc.csv").is_file()


def _has_weatherbench2_rollout_files(path: Path) -> bool:
    return (path / "weatherbench2_rollout_rmse.csv").is_file() and (
        path / "weatherbench2_rollout_acc.csv"
    ).is_file()


def _rollout_dir_from_eval_dir(path: Path, stage_dir: str) -> Path | None:
    if _has_weatherbench2_rollout_files(path):
        return path
    if _has_rollout_files(path):
        return path
    candidate = path / stage_dir
    if _has_rollout_files(candidate):
        return candidate
    return None


def _resolve_evaluation_path(path: Path, stage_dir: str) -> tuple[Path, Path]:
    path = path.expanduser().resolve()
    direct_rollout = _rollout_dir_from_eval_dir(path, stage_dir)
    if direct_rollout is not None:
        if direct_rollout != path:
            eval_dir = path
        elif _has_weatherbench2_rollout_files(path):
            eval_dir = path
        else:
            eval_dir = path.parent
        return eval_dir, direct_rollout

    candidates = [path / name for name in EVAL_DIR_NAMES]
    candidates.extend(sorted(path.glob("evaluation*")))

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    matches: list[tuple[Path, Path]] = []
    for candidate in unique_candidates:
        rollout_dir = _rollout_dir_from_eval_dir(candidate, stage_dir)
        if rollout_dir is not None:
            matches.append((candidate, rollout_dir))

    if matches:
        name_rank = {name: idx for idx, name in enumerate(EVAL_DIR_NAMES)}

        def sort_key(item: tuple[Path, Path]) -> tuple[int, float, str]:
            eval_dir, _ = item
            rank = name_rank.get(eval_dir.name, len(name_rank))
            try:
                mtime = -eval_dir.stat().st_mtime
            except FileNotFoundError:
                mtime = 0.0
            return rank, mtime, str(eval_dir)

        return sorted(matches, key=sort_key)[0]

    searched = "\n  ".join(str(candidate / stage_dir) for candidate in unique_candidates[:20])
    raise FileNotFoundError(
        f"Could not find rollout_rmse.csv and rollout_acc.csv for {path}.\n"
        f"Expected them in an evaluation directory with a {stage_dir}/ child, for example:\n"
        f"  {path}/evaluation_test_weekly52/{stage_dir}/rollout_rmse.csv\n"
        f"Searched:\n  {searched}"
    )


def _read_rollout_csv(path: Path) -> tuple[list[int], dict[str, list[float]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing rollout CSV: {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"{path} has no header row.")

        lead_column = "lead_time" if "lead_time" in fieldnames else fieldnames[0]
        variables = [name for name in fieldnames if name != lead_column]
        if not variables:
            raise ValueError(f"{path} does not contain variable columns.")

        leads: list[int] = []
        series: dict[str, list[float]] = {variable: [] for variable in variables}
        for row_index, row in enumerate(reader, start=1):
            lead_raw = row.get(lead_column, row_index)
            leads.append(int(float(lead_raw)))
            for variable in variables:
                raw_value = row.get(variable, "")
                series[variable].append(float(raw_value) if raw_value not in {"", None} else np.nan)

    return leads, series


def _read_eval_data(eval_dir: Path, rollout_dir: Path) -> dict[str, Any]:
    if _has_weatherbench2_rollout_files(rollout_dir):
        rmse_path = rollout_dir / "weatherbench2_rollout_rmse.csv"
        acc_path = rollout_dir / "weatherbench2_rollout_acc.csv"
        metadata_path = rollout_dir / "weatherbench2_summary.json"
    else:
        rmse_path = rollout_dir / "rollout_rmse.csv"
        acc_path = rollout_dir / "rollout_acc.csv"
        metadata_path = eval_dir / "fixed10_global_best_metrics.json"

    rmse_leads, rmse = _read_rollout_csv(rmse_path)
    acc_leads, acc = _read_rollout_csv(acc_path)
    if rmse_leads != acc_leads:
        raise ValueError(f"Lead times differ between RMSE and ACC CSVs in {rollout_dir}")

    metadata = None
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    return {
        "leads": rmse_leads,
        "rmse": rmse,
        "acc": acc,
        "metadata": metadata,
    }


def _read_kai_csv(path: Path, keep_variables: list[str] | None = None) -> dict[str, Any]:
    """Read a KAI baseline CSV in either accepted layout.

    * curated  -- ``variable,timestep,rmse,acc``      (data/baselines/kai_2p5.csv)
    * raw eval -- ``lead_time,variable_idx,original_channel_idx,variable_name,rmse,acc``
                  (data/baselines/kai_1p5.csv, a full 67-channel evaluation dump)

    Both normalise to (variable, timestep, rmse, acc). The raw layout's
    timestep-0 identity rows are dropped, and ``keep_variables`` restricts the
    per-channel dump to the headline variables so it does not flood the plots.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing KAI CSV: {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if {"variable", "timestep", "rmse", "acc"} <= fields:
            var_key, lead_key = "variable", "timestep"
        elif {"lead_time", "variable_name", "rmse", "acc"} <= fields:
            var_key, lead_key = "variable_name", "lead_time"
        else:
            raise ValueError(
                f"{path} matches neither KAI layout: expected columns "
                f"'variable,timestep,rmse,acc' or 'lead_time,variable_name,rmse,acc', got {sorted(fields)}"
            )
        rows = [dict(row) for row in reader]

    keep = {str(v).strip() for v in keep_variables} if keep_variables else None

    def _row_variable(row: dict[str, str]) -> str:
        return str(row[var_key]).strip()

    # Lead 0 is the identity step in raw dumps (rmse 0 / acc 1) and would drag
    # every curve to the origin -- and make the lead grid disagree with the model
    # evaluations, which start at lead 1. The curated layout starts at 1 anyway.
    kept_rows = [
        row for row in rows
        if _row_variable(row) and int(float(row[lead_key])) > 0
    ]
    if keep is not None:
        filtered = [row for row in kept_rows if _row_variable(row) in keep]
        # Only narrow when the baseline actually covers some requested variable;
        # otherwise keep everything and let _select_variables report the gap.
        if filtered:
            kept_rows = filtered
    if not kept_rows:
        raise ValueError(f"{path} yielded no usable rows after dropping the lead-0 identity step")

    leads = sorted({int(float(row[lead_key])) for row in kept_rows})
    variables = sorted({_row_variable(row) for row in kept_rows})
    rmse = {variable: [np.nan] * len(leads) for variable in variables}
    acc = {variable: [np.nan] * len(leads) for variable in variables}
    lead_to_index = {lead: index for index, lead in enumerate(leads)}

    for row in kept_rows:
        variable = _row_variable(row)
        index = lead_to_index[int(float(row[lead_key]))]
        rmse[variable][index] = float(row["rmse"])
        acc[variable][index] = float(row["acc"])

    return {
        "leads": leads,
        "rmse": rmse,
        "acc": acc,
        "metadata": {"source_csv": str(path), "kind": "kai"},
    }


def _available_variables(data: dict[str, Any]) -> set[str]:
    return set(data["rmse"]).intersection(data["acc"])


def _select_variables(sources: list[EvaluationSource], requested: list[str]) -> list[str]:
    common = set.intersection(*(_available_variables(source.data) for source in sources))
    if requested:
        available_any = set.union(*(_available_variables(source.data) for source in sources))
        missing_everywhere = sorted(set(requested) - available_any)
        if missing_everywhere:
            raise ValueError(f"Requested variables are missing from every evaluation: {missing_everywhere}")
        for source in sources:
            missing = sorted(set(requested) - _available_variables(source.data))
            if missing:
                print(f"WARNING: {source.label} is missing variables and will be skipped for them: {missing}")
        return requested

    variables = [variable for variable in DEFAULT_VARIABLES if variable in common]
    if variables:
        return variables
    if common:
        return sorted(common)
    raise ValueError("No common variables found across all evaluations.")


def _validate_leads(sources: list[EvaluationSource]) -> list[int]:
    leads = list(sources[0].data["leads"])
    mismatched = [source.label for source in sources[1:] if list(source.data["leads"]) != leads]
    if mismatched:
        raise ValueError(f"Lead-time grids differ from {sources[0].label}: {mismatched}")
    return leads


def _line_kwargs(index: int) -> dict[str, Any]:
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    linestyles = ["-", "--", "-.", ":"]
    return {
        "marker": markers[index % len(markers)],
        "linestyle": linestyles[(index // len(markers)) % len(linestyles)],
        "linewidth": 2.0,
        "markersize": 4.5,
    }


def _plot_overview(
    output_dir: Path,
    variables: list[str],
    sources: list[EvaluationSource],
    plot_format: str,
) -> Path:
    fig_height = max(5.0, 2.7 * len(variables))
    fig, axes = plt.subplots(len(variables), 2, figsize=(14.0, fig_height), squeeze=False)
    for row, variable in enumerate(variables):
        for source_index, source in enumerate(sources):
            if variable not in _available_variables(source.data):
                continue
            leads = np.asarray(source.data["leads"], dtype=np.int64)
            kwargs = _line_kwargs(source_index)
            axes[row, 0].plot(leads, source.data["rmse"][variable], label=source.label, **kwargs)
            axes[row, 1].plot(leads, source.data["acc"][variable], label=source.label, **kwargs)

        axes[row, 0].set_title(f"{variable} RMSE")
        axes[row, 0].set_ylabel("RMSE")
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].set_title(f"{variable} ACC")
        axes[row, 1].set_ylabel("ACC")
        axes[row, 1].set_ylim(-1.05, 1.05)
        axes[row, 1].grid(True, alpha=0.3)

        if row == len(variables) - 1:
            axes[row, 0].set_xlabel("Lead time")
            axes[row, 1].set_xlabel("Lead time")

    handles_by_label: dict[str, Any] = {}
    for axis in axes.ravel():
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)
    if handles_by_label:
        fig.legend(
            list(handles_by_label.values()),
            list(handles_by_label.keys()),
            loc="upper center",
            ncol=min(len(handles_by_label), 3),
            fontsize=9,
        )
    fig.suptitle("Experiment RMSE and ACC comparison", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    path = output_dir / f"combined_all_variables_rmse_acc.{plot_format}"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_variable(
    output_dir: Path,
    variable: str,
    sources: list[EvaluationSource],
    plot_format: str,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for source_index, source in enumerate(sources):
        if variable not in _available_variables(source.data):
            continue
        leads = np.asarray(source.data["leads"], dtype=np.int64)
        kwargs = _line_kwargs(source_index)
        axes[0].plot(leads, source.data["rmse"][variable], label=source.label, **kwargs)
        axes[1].plot(leads, source.data["acc"][variable], label=source.label, **kwargs)

    axes[0].set_title(f"{variable} RMSE by lead time")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title(f"{variable} ACC by lead time")
    axes[1].set_xlabel("Lead time")
    axes[1].set_ylabel("ACC")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_xticks(sources[0].data["leads"])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    path = output_dir / f"combined_{_safe_stem(variable)}_rmse_acc.{plot_format}"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_summary_csv(output_dir: Path, variables: list[str], sources: list[EvaluationSource]) -> Path:
    rows: list[dict[str, Any]] = []
    for variable in variables:
        for source in sources:
            has_variable = variable in _available_variables(source.data)
            leads = np.asarray(source.data["leads"], dtype=np.int64)
            rmse = np.asarray(source.data["rmse"][variable], dtype=np.float64) if has_variable else np.asarray([])
            acc = np.asarray(source.data["acc"][variable], dtype=np.float64) if has_variable else np.asarray([])
            rows.append(
                {
                    "variable": variable,
                    "experiment": source.label,
                    "requested_path": str(source.requested_path),
                    "eval_dir": str(source.eval_dir),
                    "rollout_dir": str(source.rollout_dir),
                    "avg_rmse": float(np.nanmean(rmse)) if has_variable else np.nan,
                    "last_lead": int(leads[-1]),
                    "last_lead_rmse": float(rmse[-1]) if has_variable else np.nan,
                    "avg_acc": float(np.nanmean(acc)) if has_variable else np.nan,
                    "last_lead_acc": float(acc[-1]) if has_variable else np.nan,
                }
            )

    path = output_dir / "combined_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_metadata(output_dir: Path, variables: list[str], sources: list[EvaluationSource]) -> Path:
    payload = {
        "variables": variables,
        "sources": [
            {
                "label": source.label,
                "requested_path": str(source.requested_path),
                "eval_dir": str(source.eval_dir),
                "rollout_dir": str(source.rollout_dir),
                "leads": source.data["leads"],
                "selection": None
                if source.data.get("metadata") is None
                else source.data["metadata"].get("selection"),
            }
            for source in sources
        ],
    }
    path = output_dir / "combined_metadata.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot RMSE and ACC rollout curves for multiple GraphWeather experiment evaluations. "
            "Inputs may be experiment roots, evaluation dirs, or S10 rollout dirs."
        )
    )
    parser.add_argument("--experiments", nargs="*", default=[str(path) for path in DEFAULT_EXPERIMENTS])
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    parser.add_argument("--kai_csv", default=str(DEFAULT_KAI_CSV), type=str, help="Optional KAI baseline CSV, either 'variable,timestep,rmse,acc' (kai_2.5.csv) or the raw per-channel dump 'lead_time,variable_name,rmse,acc' (kai_1.5.csv). Use '' to disable.")
    parser.add_argument("--kai_label", default="KAI 2.5", type=str)
    parser.add_argument("--stage_dir", default="S10", help="Rollout subdirectory to read, default S10.")
    parser.add_argument("--variables", nargs="*", default=None, help="Variable names, comma or space separated.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), type=str)
    parser.add_argument("--plot_format", default="png", type=str)
    parser.add_argument("--skip_per_variable_plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    experiment_paths = [Path(path) for path in args.experiments]
    labels = [str(label) for label in args.labels]
    if len(experiment_paths) != len(labels):
        raise ValueError(f"--experiments and --labels must have the same length, got {len(experiment_paths)} and {len(labels)}")
    if not experiment_paths:
        raise ValueError("At least one experiment path is required.")

    requested_variables = _parse_items(args.variables)

    sources: list[EvaluationSource] = []
    for label, requested_path in zip(labels, experiment_paths):
        eval_dir, rollout_dir = _resolve_evaluation_path(requested_path, str(args.stage_dir))
        sources.append(
            EvaluationSource(
                label=label,
                requested_path=requested_path.expanduser().resolve(),
                eval_dir=eval_dir,
                rollout_dir=rollout_dir,
                data=_read_eval_data(eval_dir, rollout_dir),
            )
        )
    if str(args.kai_csv).strip():
        kai_csv = Path(args.kai_csv).expanduser().resolve()
        sources.append(
            EvaluationSource(
                label=str(args.kai_label),
                requested_path=kai_csv,
                eval_dir=kai_csv.parent,
                rollout_dir=kai_csv,
                data=_read_kai_csv(kai_csv, requested_variables or DEFAULT_VARIABLES),
            )
        )

    _validate_leads(sources)
    variables = _select_variables(sources, requested_variables)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_format = str(args.plot_format).lstrip(".")

    plot_paths = [_plot_overview(output_dir, variables, sources, plot_format)]
    if not args.skip_per_variable_plots:
        plot_paths.extend(_plot_variable(output_dir, variable, sources, plot_format) for variable in variables)
    summary_path = _write_summary_csv(output_dir, variables, sources)
    metadata_path = _write_metadata(output_dir, variables, sources)

    print(f"Saved comparison plots to: {output_dir}")
    for path in plot_paths:
        print(f"  {path}")
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved metadata JSON: {metadata_path}")
    print("Sources:")
    for source in sources:
        print(f"  {source.label}: {source.rollout_dir}")


if __name__ == "__main__":
    main()
