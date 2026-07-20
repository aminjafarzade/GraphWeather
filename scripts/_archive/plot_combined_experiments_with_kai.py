from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

Path(os.environ.setdefault("XDG_CACHE_HOME", "/tmp")).mkdir(parents=True, exist_ok=True)
Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_EXPERIMENT = PROJECT_ROOT / "experiments" / "main_raw_2p5_b4_acc3_bf16_delta_warmup_cosine"
STATIC_EXPERIMENT = PROJECT_ROOT / "experiments" / "main_raw_2p5_b4_acc3_bf16_delta_warmup_cosine_static_forcing"

DEFAULT_BASE_EVAL = BASE_EXPERIMENT / "evaluation_test_weekly52"
DEFAULT_STATIC_EVAL = STATIC_EXPERIMENT / "evaluation_test_with_kai"
DEFAULT_KAI_CSV = DEFAULT_BASE_EVAL / "kai_2.5.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "comparison_static_forcing_vs_base_with_kai"
DEFAULT_VARIABLES = ["z500", "t2m", "t850", "msl", "q700", "u850"]


def _safe_stem(value: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return stem.strip("_") or "variable"


def _read_rollout_csv(path: Path, variables: list[str]) -> dict[str, list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing rollout CSV: {path}")
    series = {variable: [] for variable in variables}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [variable for variable in variables if variable not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing requested variable columns: {missing}")
        for row in reader:
            for variable in variables:
                series[variable].append(float(row[variable]))
    return series


def _read_graphweather_eval(eval_dir: Path, variables: list[str]) -> dict[str, Any]:
    s10_dir = eval_dir / "S10"
    rmse = _read_rollout_csv(s10_dir / "rollout_rmse.csv", variables)
    acc = _read_rollout_csv(s10_dir / "rollout_acc.csv", variables)
    lead_count = len(next(iter(rmse.values())))
    return {
        "rmse": rmse,
        "acc": acc,
        "leads": list(range(1, lead_count + 1)),
        "metadata": _read_optional_json(eval_dir / "fixed10_global_best_metrics.json"),
        "source_dir": str(eval_dir),
    }


def _read_kai_csv(path: Path, variables: list[str], lead_count: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing KAI CSV: {path}")
    rmse = {variable: [np.nan] * lead_count for variable in variables}
    acc = {variable: [np.nan] * lead_count for variable in variables}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"variable", "timestep", "rmse", "acc"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row in reader:
            variable = str(row["variable"]).strip()
            if variable not in rmse:
                continue
            lead = int(float(row["timestep"]))
            if 1 <= lead <= lead_count:
                rmse[variable][lead - 1] = float(row["rmse"])
                acc[variable][lead - 1] = float(row["acc"])
    return {
        "rmse": rmse,
        "acc": acc,
        "leads": list(range(1, lead_count + 1)),
        "source_csv": str(path),
    }


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _plot_variable(
    output_dir: Path,
    variable: str,
    curves: list[dict[str, Any]],
    plot_format: str,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for curve in curves:
        leads = np.asarray(curve["data"]["leads"], dtype=np.int64)
        label = str(curve["label"])
        style = str(curve.get("style", "model"))
        if style == "kai":
            marker = "s"
            linestyle = "-."
            linewidth = 1.9
        else:
            marker = "o"
            linestyle = "-"
            linewidth = 2.0
        axes[0].plot(
            leads,
            curve["data"]["rmse"][variable],
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
        axes[1].plot(
            leads,
            curve["data"]["acc"][variable],
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )

    axes[0].set_title(f"{variable} RMSE by lead time")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title(f"{variable} ACC by lead time")
    axes[1].set_xlabel("Lead time (days)")
    axes[1].set_ylabel("ACC")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_xticks(curves[0]["data"]["leads"])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle("2.5-degree GraphWeather experiments vs KAI", fontsize=13)
    fig.tight_layout()
    path = output_dir / f"combined_{_safe_stem(variable)}_rmse_acc.{plot_format}"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_summary_csv(output_dir: Path, variables: list[str], curves: list[dict[str, Any]]) -> Path:
    rows: list[dict[str, Any]] = []
    for variable in variables:
        for curve in curves:
            rmse = np.asarray(curve["data"]["rmse"][variable], dtype=np.float64)
            acc = np.asarray(curve["data"]["acc"][variable], dtype=np.float64)
            rows.append(
                {
                    "variable": variable,
                    "model": curve["label"],
                    "source": curve["source"],
                    "avg_rmse_1_10": float(np.nanmean(rmse)),
                    "day10_rmse": float(rmse[-1]),
                    "avg_acc_1_10": float(np.nanmean(acc)),
                    "day10_acc": float(acc[-1]),
                }
            )
    path = output_dir / "combined_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_metadata(output_dir: Path, variables: list[str], curves: list[dict[str, Any]]) -> Path:
    payload = {
        "variables": variables,
        "curves": [
            {
                "label": curve["label"],
                "source": curve["source"],
                "style": curve.get("style", "model"),
                "selection": None
                if curve["data"].get("metadata") is None
                else curve["data"]["metadata"].get("selection"),
            }
            for curve in curves
        ],
    }
    path = output_dir / "combined_metadata.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot two GraphWeather experiment evaluations against KAI.")
    parser.add_argument("--base_eval_dir", default=str(DEFAULT_BASE_EVAL), type=str)
    parser.add_argument("--static_eval_dir", default=str(DEFAULT_STATIC_EVAL), type=str)
    parser.add_argument("--kai_csv", default=str(DEFAULT_KAI_CSV), type=str)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), type=str)
    parser.add_argument("--base_label", default=BASE_EXPERIMENT.name, type=str)
    parser.add_argument("--static_label", default=STATIC_EXPERIMENT.name, type=str)
    parser.add_argument("--kai_label", default="Kai 7M", type=str)
    parser.add_argument("--variables", nargs="*", default=DEFAULT_VARIABLES)
    parser.add_argument("--plot_format", default="png", type=str)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    variables = [str(item).strip() for item in args.variables if str(item).strip()]
    if not variables:
        raise ValueError("At least one variable is required.")

    base_eval_dir = Path(args.base_eval_dir).expanduser().resolve()
    static_eval_dir = Path(args.static_eval_dir).expanduser().resolve()
    kai_csv = Path(args.kai_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_data = _read_graphweather_eval(base_eval_dir, variables)
    static_data = _read_graphweather_eval(static_eval_dir, variables)
    if base_data["leads"] != static_data["leads"]:
        raise ValueError("The two GraphWeather evaluations have different lead-time grids.")
    kai_data = _read_kai_csv(kai_csv, variables, lead_count=len(base_data["leads"]))

    curves = [
        {"label": str(args.base_label), "source": str(base_eval_dir), "data": base_data, "style": "model"},
        {"label": str(args.static_label), "source": str(static_eval_dir), "data": static_data, "style": "model"},
        {"label": str(args.kai_label), "source": str(kai_csv), "data": kai_data, "style": "kai"},
    ]

    plot_paths = [_plot_variable(output_dir, variable, curves, str(args.plot_format).lstrip(".")) for variable in variables]
    summary_path = _write_summary_csv(output_dir, variables, curves)
    metadata_path = _write_metadata(output_dir, variables, curves)

    print(f"Saved combined plots to: {output_dir}")
    for path in plot_paths:
        print(f"  {path}")
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved metadata JSON: {metadata_path}")


if __name__ == "__main__":
    main()
