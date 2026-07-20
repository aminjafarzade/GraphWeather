from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

Path(os.environ.setdefault("XDG_CACHE_HOME", "/tmp")).mkdir(parents=True, exist_ok=True)
Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")).mkdir(parents=True, exist_ok=True)

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.config import YParams, setup_logging
from src.evaluator import _canonical_for_name, run_evaluation_from_params


DEFAULT_VARIABLES = ["z500", "t2m", "t850", "msl", "q700", "u850"]


def _looks_like_yaml(value: str) -> bool:
    return value.endswith((".yaml", ".yml")) or Path(value).expanduser().exists()


def _resolve_config_args(args: argparse.Namespace) -> tuple[str, str]:
    yaml_path = args.yaml_config
    config_name = args.config_name
    if args.config:
        if _looks_like_yaml(args.config):
            yaml_path = args.config
            if config_name is None:
                config_map = {
                    "weather_dual_resolution.yaml": "raw",
                    "weather_dual_resolution_l3.yaml": "raw_l3",
                    "weather_dual_resolution_l3_stage_warmup_cosine.yaml": "raw_l3_stage_warmup_cosine",
                    "weather_dual_resolution_l3_blocks3.yaml": "raw_l3_blocks3",
                    "weather_dual_resolution_l3_heavy_unet.yaml": "raw_l3_heavy_unet",
                    "weather_dual_resolution_l3_full_rollout.yaml": "raw_l3_full_rollout",
                    "weather_dual_resolution_l3_hidden128.yaml": "raw_l3_hidden128",
                    "weather_dual_resolution_l3_hidden160.yaml": "raw_l3_hidden160",
                    "weather_dual_resolution_l3_orog_tisr_fixed.yaml": "raw_l3_orog_tisr_fixed",
                    "weather_dual_resolution_l3_hidden128_orog_tisr_fixed.yaml": "raw_l3_hidden128_orog_tisr_fixed",
                    "weather_dual_resolution_l3_hidden160_orog_tisr_fixed.yaml": "raw_l3_hidden160_orog_tisr_fixed",
                }
                config_name = config_map.get(Path(args.config).name)
        elif config_name is None:
            config_name = args.config
    yaml_path = yaml_path or str(project_root / "configs" / "gnn_5p625.yaml")
    config_name = config_name or "raw_5p625"
    return str(Path(yaml_path).expanduser().resolve()), config_name


def _parse_strings(values: list[str] | None, default: list[str] | None = None) -> list[str]:
    if not values:
        return list(default or [])
    parsed: list[str] = []
    for value in values:
        parsed.extend(chunk.strip() for chunk in str(value).replace(",", " ").split() if chunk.strip())
    return parsed or list(default or [])


def _parse_optional_strings(values: list[str] | None, default: list[str]) -> list[str]:
    if values is None:
        return list(default)
    return _parse_strings(values, [])


def _split_data_path(params: Any, split: str) -> str:
    if split == "train":
        return str(params.train_data_path)
    if split == "valid":
        return str(params.valid_data_path)
    if split == "test":
        return str(params.get("test_dataset_path", params.valid_data_path))
    raise ValueError(f"Unsupported split: {split}")


def _apply_common_eval_params(params: Any, args: argparse.Namespace, output_dir: Path) -> None:
    params["eval_split"] = str(args.split)
    params["test_dataset_path"] = _split_data_path(params, str(args.split))
    params["eval_output_dir"] = str(output_dir)
    params["eval_checkpoint_path"] = str(args.checkpoint)
    params["eval_fixed_rollout_steps"] = int(args.fixed_rollout_steps)
    params["eval_forecast_steps"] = [int(args.fixed_rollout_steps)]
    params["eval_compare_stage_checkpoints"] = False
    params["eval_selection"] = str(args.selection)
    params["eval_ic_stride"] = int(args.stride)
    params["eval_start_offset"] = int(args.start_offset)
    params["eval_plot_variables"] = _parse_strings(args.variables, DEFAULT_VARIABLES)
    params["plot_persistence"] = bool(args.include_persistence)
    params["bootstrap_samples"] = int(args.bootstrap_samples)
    params["confidence_level"] = float(args.confidence_level)
    params["bootstrap_seed"] = int(args.bootstrap_seed)
    params["plot_confidence_intervals"] = bool(args.plot_confidence_intervals)
    if args.experiment_dir:
        params["experiment_dir"] = str(Path(args.experiment_dir).expanduser().resolve())
    if args.climatology_path:
        params["climatology_path"] = str(args.climatology_path)
    if args.build_climatology_if_missing:
        params["build_climatology_if_missing"] = True
        params["compute_climatology"] = True
    if args.start_date:
        params["eval_start_date"] = str(args.start_date)
    if args.end_date:
        params["eval_end_date"] = str(args.end_date)
    if args.n_initial_conditions is not None:
        params["n_initial_conditions"] = int(args.n_initial_conditions)
    if args.max_initial_conditions is not None:
        params["max_initial_conditions"] = int(args.max_initial_conditions)
    if args.device:
        params["eval_device"] = str(args.device)
    if args.external_baseline_csv:
        params["external_baseline_csv"] = args.external_baseline_csv
    if args.external_baseline_label:
        params["external_baseline_label"] = args.external_baseline_label
    if args.external_baseline_params_m:
        params["external_baseline_params_m"] = args.external_baseline_params_m


def _run_eval(args: argparse.Namespace, output_dir: Path, *, override: bool) -> Path:
    yaml_path, config_name = _resolve_config_args(args)
    params = YParams(yaml_path, config_name, resolution_mode=args.resolution_mode)
    _apply_common_eval_params(params, args, output_dir)
    if override:
        params["eval_target_override"] = True
        params["eval_copy_variables"] = _parse_optional_strings(args.eval_copy_variables, ["orog"])
        params["eval_known_future_variables"] = _parse_optional_strings(args.eval_known_future_variables, ["tisr"])
        params["eval_exclude_loss_variables"] = _parse_optional_strings(args.eval_exclude_loss_variables, ["orog", "tisr"])
        params["debug_eval_target_override"] = bool(args.debug_eval_target_override)
    else:
        params["eval_target_override"] = False
        params["debug_eval_target_override"] = False

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(rank=0, log_file=str(output_dir / "evaluation.log"))
    params.log()
    run_evaluation_from_params(params, logger=logging)
    return output_dir


def _load_metrics(eval_dir: Path, fixed_steps: int) -> dict[str, Any]:
    path = eval_dir / f"fixed{int(fixed_steps)}_global_best_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation metrics JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["checkpoints"]["global_best"]["metrics"]


def _metric_series(metrics: dict[str, Any], variable: str, metric_name: str) -> list[float | None]:
    if variable not in metrics:
        available = ", ".join(sorted(metrics))
        raise ValueError(f"Variable {variable!r} is missing from evaluation JSON. Available: {available}")
    return metrics[variable][metric_name]["mean"]


def _finite_percent(delta: float, baseline: float | None) -> float | None:
    if baseline is None:
        return None
    if not np.isfinite(float(baseline)) or float(baseline) == 0.0:
        return None
    return float(100.0 * delta / float(baseline))


def _safe_stem(value: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return stem.strip("_") or "variable"


def _override_label(args: argparse.Namespace) -> str:
    copy_vars = _parse_optional_strings(args.eval_copy_variables, ["orog"])
    known_future = _parse_optional_strings(args.eval_known_future_variables, ["tisr"])
    parts: list[str] = []
    if copy_vars:
        parts.append("copy " + ",".join(copy_vars))
    if known_future:
        parts.append("future " + ",".join(known_future))
    return "override: " + "; ".join(parts) if parts else "override"


def _series_array(metrics: dict[str, Any], variable: str, metric_name: str) -> np.ndarray:
    values = _metric_series(metrics, variable, metric_name)
    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float64)


def _plot_comparison(
    output_dir: Path,
    normal_metrics: dict[str, Any],
    override_metrics: dict[str, Any],
    variables: list[str],
    fixed_steps: int,
    args: argparse.Namespace,
) -> list[Path]:
    if bool(args.skip_plots):
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required for comparison plots. Install it or pass --skip_plots.") from exc

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_format = str(args.plot_format).lstrip(".").lower()
    leads = np.arange(1, int(fixed_steps) + 1)
    override_label = _override_label(args)
    plot_paths: list[Path] = []

    fig_height = max(5.2, 2.7 * len(variables))
    fig, axes = plt.subplots(len(variables), 2, figsize=(14.0, fig_height), squeeze=False)
    for row_idx, variable in enumerate(variables):
        key = _canonical_for_name(variable) or variable
        normal_rmse = _series_array(normal_metrics, key, "rmse")
        override_rmse = _series_array(override_metrics, key, "rmse")
        normal_acc = _series_array(normal_metrics, key, "acc")
        override_acc = _series_array(override_metrics, key, "acc")

        axes[row_idx, 0].plot(leads, normal_rmse, marker="o", linewidth=2.0, label="normal")
        axes[row_idx, 0].plot(leads, override_rmse, marker="s", linewidth=2.0, label=override_label)
        axes[row_idx, 0].set_title(f"{key} RMSE")
        axes[row_idx, 0].set_ylabel("RMSE")
        axes[row_idx, 0].grid(True, alpha=0.3)

        axes[row_idx, 1].plot(leads, normal_acc, marker="o", linewidth=2.0, label="normal")
        axes[row_idx, 1].plot(leads, override_acc, marker="s", linewidth=2.0, label=override_label)
        axes[row_idx, 1].set_title(f"{key} ACC")
        axes[row_idx, 1].set_ylabel("ACC")
        axes[row_idx, 1].set_ylim(-1.05, 1.05)
        axes[row_idx, 1].grid(True, alpha=0.3)
        if row_idx == len(variables) - 1:
            axes[row_idx, 0].set_xlabel("Lead time")
            axes[row_idx, 1].set_xlabel("Lead time")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), fontsize=9)
    fig.suptitle("Normal vs evaluation target override", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    overview_path = plot_dir / f"comparison_all_variables_rmse_acc.{plot_format}"
    fig.savefig(overview_path, dpi=170)
    plt.close(fig)
    plot_paths.append(overview_path)

    for variable in variables:
        key = _canonical_for_name(variable) or variable
        normal_rmse = _series_array(normal_metrics, key, "rmse")
        override_rmse = _series_array(override_metrics, key, "rmse")
        normal_acc = _series_array(normal_metrics, key, "acc")
        override_acc = _series_array(override_metrics, key, "acc")

        fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
        axes[0].plot(leads, normal_rmse, marker="o", linewidth=2.0, label="normal")
        axes[0].plot(leads, override_rmse, marker="s", linewidth=2.0, label=override_label)
        axes[0].set_title(f"{key} RMSE by lead time")
        axes[0].set_ylabel("RMSE")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=9)

        axes[1].plot(leads, normal_acc, marker="o", linewidth=2.0, label="normal")
        axes[1].plot(leads, override_acc, marker="s", linewidth=2.0, label=override_label)
        axes[1].set_title(f"{key} ACC by lead time")
        axes[1].set_xlabel("Lead time")
        axes[1].set_ylabel("ACC")
        axes[1].set_ylim(-1.05, 1.05)
        axes[1].set_xticks(leads)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=9)

        fig.tight_layout()
        path = plot_dir / f"comparison_{_safe_stem(key)}_rmse_acc.{plot_format}"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        plot_paths.append(path)
    return plot_paths


def _comparison_rows(
    normal_metrics: dict[str, Any],
    override_metrics: dict[str, Any],
    variables: list[str],
    fixed_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variable in variables:
        key = _canonical_for_name(variable) or variable
        normal_rmse = _metric_series(normal_metrics, key, "rmse")
        override_rmse = _metric_series(override_metrics, key, "rmse")
        normal_acc = _metric_series(normal_metrics, key, "acc")
        override_acc = _metric_series(override_metrics, key, "acc")
        for lead_idx in range(int(fixed_steps)):
            n_rmse = None if normal_rmse[lead_idx] is None else float(normal_rmse[lead_idx])
            o_rmse = None if override_rmse[lead_idx] is None else float(override_rmse[lead_idx])
            n_acc = None if normal_acc[lead_idx] is None else float(normal_acc[lead_idx])
            o_acc = None if override_acc[lead_idx] is None else float(override_acc[lead_idx])
            rmse_delta = None if n_rmse is None or o_rmse is None else float(o_rmse - n_rmse)
            acc_delta = None if n_acc is None or o_acc is None else float(o_acc - n_acc)
            rows.append(
                {
                    "variable": key,
                    "lead_time": int(lead_idx + 1),
                    "normal_rmse": n_rmse,
                    "override_rmse": o_rmse,
                    "rmse_delta": rmse_delta,
                    "rmse_percent_change": None if rmse_delta is None else _finite_percent(rmse_delta, n_rmse),
                    "normal_acc": n_acc,
                    "override_acc": o_acc,
                    "acc_delta": acc_delta,
                }
            )
    return rows


def _write_comparison(
    output_dir: Path,
    rows: list[dict[str, Any]],
    plot_paths: list[Path],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    final_lead = int(args.fixed_rollout_steps)
    day10_summary = [row for row in rows if int(row["lead_time"]) == final_lead]
    payload = {
        "checkpoint": str(args.checkpoint),
        "fixed_rollout_steps": final_lead,
        "variables": sorted({str(row["variable"]) for row in rows}),
        "normal_dir": str(output_dir / "normal"),
        "override_dir": str(output_dir / "orog_tisr_override"),
        "plots": [str(path) for path in plot_paths],
        "rows": rows,
        "day_10_summary": day10_summary,
    }
    json_path = output_dir / "comparison_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    title = "Day-10 summary" if final_lead == 10 else f"Day-{final_lead} summary"
    lines = [
        "Evaluation target override comparison:",
        f"- checkpoint: {args.checkpoint}",
        f"- normal: {output_dir / 'normal'}",
        f"- override: {output_dir / 'orog_tisr_override'}",
        f"- plots: {output_dir / 'plots'}" if plot_paths else "- plots: skipped",
        "",
        f"{title}:",
    ]
    for row in day10_summary:
        percent = row["rmse_percent_change"]
        percent_text = "n/a" if percent is None else f"{float(percent):+.3f}%"
        lines.append(
            f"- {row['variable']}: RMSE {row['normal_rmse']:.6g} -> {row['override_rmse']:.6g} "
            f"({row['rmse_delta']:+.6g}, {percent_text}); "
            f"ACC {row['normal_acc']:.6g} -> {row['override_acc']:.6g} ({row['acc_delta']:+.6g})"
        )
    txt_path = output_dir / "comparison_report.txt"
    txt_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison JSON: {json_path}")
    print(f"Saved comparison report: {txt_path}")
    if plot_paths:
        print(f"Saved comparison plots under: {output_dir / 'plots'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare normal evaluation with evaluation-only orog/tisr target override.")
    parser.add_argument("--yaml_config", default=str(project_root / "configs" / "gnn_5p625.yaml"), type=str)
    parser.add_argument("--config", default="raw_5p625", type=str)
    parser.add_argument("--config_name", default=None, type=str)
    parser.add_argument("--resolution_mode", default=None, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--experiment_dir", default=None, type=str)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--fixed_rollout_steps", default=10, type=int)
    parser.add_argument("--selection", default="stride", choices=["first_n", "stride", "all"])
    parser.add_argument("--stride", default=7, type=int)
    parser.add_argument("--start_offset", default=0, type=int)
    parser.add_argument("--start_date", default=None, type=str)
    parser.add_argument("--end_date", default=None, type=str)
    parser.add_argument("--n_initial_conditions", default=None, type=int)
    parser.add_argument("--max_initial_conditions", default=None, type=int)
    parser.add_argument("--variables", nargs="*", default=None)
    parser.add_argument("--include_persistence", action="store_true")
    parser.add_argument("--climatology_path", default=None, type=str)
    parser.add_argument("--build_climatology_if_missing", action="store_true")
    parser.add_argument("--bootstrap_samples", default=0, type=int)
    parser.add_argument("--confidence_level", default=0.95, type=float)
    parser.add_argument("--bootstrap_seed", default=42, type=int)
    parser.add_argument("--plot_confidence_intervals", action="store_true")
    parser.add_argument("--external_baseline_csv", nargs="*", default=None)
    parser.add_argument("--external_baseline_label", nargs="*", default=None)
    parser.add_argument("--external_baseline_params_m", nargs="*", default=None, type=float)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--eval_copy_variables", nargs="*", default=None)
    parser.add_argument("--eval_known_future_variables", nargs="*", default=None)
    parser.add_argument("--eval_exclude_loss_variables", nargs="*", default=None)
    parser.add_argument("--debug_eval_target_override", action="store_true")
    parser.add_argument("--plot_format", default="png", type=str)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--output_dir", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fixed_steps = int(args.fixed_rollout_steps)
    if fixed_steps <= 0:
        raise ValueError("--fixed_rollout_steps must be positive.")
    output_dir = Path(args.output_dir).expanduser().resolve()
    normal_dir = output_dir / "normal"
    override_dir = output_dir / "orog_tisr_override"

    _run_eval(args, normal_dir, override=False)
    _run_eval(args, override_dir, override=True)

    variables = _parse_strings(args.variables, DEFAULT_VARIABLES)
    normal_metrics = _load_metrics(normal_dir, fixed_steps)
    override_metrics = _load_metrics(override_dir, fixed_steps)
    rows = _comparison_rows(normal_metrics, override_metrics, variables, fixed_steps)
    plot_paths = _plot_comparison(output_dir, normal_metrics, override_metrics, variables, fixed_steps, args)
    _write_comparison(output_dir, rows, plot_paths, args)


if __name__ == "__main__":
    main()
