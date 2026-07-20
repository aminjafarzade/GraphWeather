from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

Path(os.environ.setdefault("XDG_CACHE_HOME", "/tmp")).mkdir(parents=True, exist_ok=True)
Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")).mkdir(parents=True, exist_ok=True)

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.config import YParams, setup_logging
from src.evaluator import SENSITIVITY_MODES, _canonical_for_name, run_evaluation_from_params


DEFAULT_VARIABLES = ["z500", "t2m", "t850", "msl", "q700", "u850"]
DEFAULT_PLOT_MODES = [
    "normal",
    "override_orog_tisr",
    "tisr_zero",
    "tisr_random",
    "orog_zero",
    "orog_random",
    "orog_tisr_random",
]
MODE_GROUPS = {
    "tisr_modes": ["normal", "override_orog_tisr", "tisr_zero", "tisr_random", "tisr_shuffle"],
    "orog_modes": ["normal", "override_orog_tisr", "orog_zero", "orog_random", "orog_shuffle"],
    "combined_modes": ["normal", "override_orog_tisr", "orog_tisr_zero", "orog_tisr_random", "orog_tisr_shuffle"],
}


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


def _split_data_path(params: Any, split: str) -> str:
    if split == "train":
        return str(params.train_data_path)
    if split == "valid":
        return str(params.valid_data_path)
    if split == "test":
        return str(params.get("test_dataset_path", params.valid_data_path))
    raise ValueError(f"Unsupported split: {split}")


def _resolve_modes(args: argparse.Namespace) -> list[str]:
    requested = _parse_strings(args.modes, list(SENSITIVITY_MODES))
    invalid = [mode for mode in requested if mode not in SENSITIVITY_MODES]
    if invalid:
        raise ValueError(f"Unsupported modes: {invalid}. Expected one of: {', '.join(SENSITIVITY_MODES)}")
    modes: list[str] = []
    if "normal" not in requested:
        modes.append("normal")
    for mode in requested:
        if mode not in modes:
            modes.append(mode)
    return modes


def _apply_common_eval_params(params: Any, args: argparse.Namespace, output_dir: Path, mode: str) -> None:
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
    params["eval_sensitivity_mode"] = str(mode)
    params["eval_sensitivity_noise_seed"] = int(args.noise_seed)
    params["debug_eval_sensitivity"] = True
    params["eval_sensitivity_debug_samples"] = int(args.debug_samples)
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
    if args.max_eval_initializations is not None:
        params["max_initial_conditions"] = int(args.max_eval_initializations)
    if args.device:
        params["eval_device"] = str(args.device)
    if args.external_baseline_csv:
        params["external_baseline_csv"] = args.external_baseline_csv
    if args.external_baseline_label:
        params["external_baseline_label"] = args.external_baseline_label
    if args.external_baseline_params_m:
        params["external_baseline_params_m"] = args.external_baseline_params_m


def _run_mode(args: argparse.Namespace, output_dir: Path, mode: str) -> None:
    yaml_path, config_name = _resolve_config_args(args)
    params = YParams(yaml_path, config_name, resolution_mode=args.resolution_mode)
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    _apply_common_eval_params(params, args, mode_dir, mode)
    setup_logging(rank=0, log_file=str(mode_dir / "evaluation.log"))
    logging.info("Running OROG/TISR sensitivity mode: %s", mode)
    print(f"Running OROG/TISR sensitivity mode: {mode}")
    params.log()
    run_evaluation_from_params(params, logger=logging)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_mode_metrics(output_dir: Path, mode: str, fixed_steps: int) -> dict[str, Any]:
    path = output_dir / mode / f"fixed{int(fixed_steps)}_global_best_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics JSON for mode {mode}: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["checkpoints"]["global_best"]["metrics"]


def _metric_series(metrics: dict[str, Any], variable: str, metric_name: str) -> list[float | None]:
    if variable not in metrics:
        available = ", ".join(sorted(metrics))
        raise ValueError(f"Variable {variable!r} is missing from evaluation JSON. Available: {available}")
    return metrics[variable][metric_name]["mean"]


def _series_array(metrics: dict[str, Any], variable: str, metric_name: str) -> np.ndarray:
    values = _metric_series(metrics, variable, metric_name)
    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float64)


def _json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _safe_stem(value: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return stem.strip("_") or "variable"


def _percent_change(value: float | None, normal: float | None) -> float | None:
    if value is None or normal is None:
        return None
    if not np.isfinite(float(normal)) or float(normal) == 0.0:
        return None
    return float(100.0 * (float(value) - float(normal)) / float(normal))


def _comparison_rows(
    metrics_by_mode: dict[str, dict[str, Any]],
    modes: list[str],
    variables: list[str],
    fixed_steps: int,
) -> list[dict[str, Any]]:
    normal_metrics = metrics_by_mode["normal"]
    rows: list[dict[str, Any]] = []
    for mode in modes:
        mode_metrics = metrics_by_mode[mode]
        for variable in variables:
            key = _canonical_for_name(variable) or variable
            normal_rmse = _series_array(normal_metrics, key, "rmse")
            normal_acc = _series_array(normal_metrics, key, "acc")
            mode_rmse = _series_array(mode_metrics, key, "rmse")
            mode_acc = _series_array(mode_metrics, key, "acc")
            for lead_idx in range(int(fixed_steps)):
                n_rmse = _json_number(normal_rmse[lead_idx])
                n_acc = _json_number(normal_acc[lead_idx])
                rmse = _json_number(mode_rmse[lead_idx])
                acc = _json_number(mode_acc[lead_idx])
                rmse_delta = None if rmse is None or n_rmse is None else float(rmse - n_rmse)
                acc_delta = None if acc is None or n_acc is None else float(acc - n_acc)
                rows.append(
                    {
                        "mode": mode,
                        "variable": key,
                        "lead": int(lead_idx + 1),
                        "rmse": rmse,
                        "acc": acc,
                        "normal_rmse": n_rmse,
                        "normal_acc": n_acc,
                        "rmse_delta": rmse_delta,
                        "rmse_percent_change": None if rmse_delta is None else _percent_change(rmse, n_rmse),
                        "acc_delta": acc_delta,
                    }
                )
    return rows


def _day10_rows(rows: list[dict[str, Any]], final_lead: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if int(row["lead"]) != int(final_lead):
            continue
        out.append(
            {
                "mode": row["mode"],
                "variable": row["variable"],
                "rmse_day10": row["rmse"],
                "normal_rmse_day10": row["normal_rmse"],
                "rmse_delta_day10": row["rmse_delta"],
                "rmse_percent_change_day10": row["rmse_percent_change"],
                "acc_day10": row["acc"],
                "normal_acc_day10": row["normal_acc"],
                "acc_delta_day10": row["acc_delta"],
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    day10: list[dict[str, Any]],
    modes: list[str],
    variables: list[str],
    args: argparse.Namespace,
) -> None:
    _write_csv(output_dir / "comparison_summary.csv", rows)
    _write_csv(output_dir / "comparison_day10.csv", day10)
    summary_payload = {
        "checkpoint": str(args.checkpoint),
        "fixed_rollout_steps": int(args.fixed_rollout_steps),
        "modes": modes,
        "variables": variables,
        "state_space": "normalized",
        "rows": rows,
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    (output_dir / "comparison_day10.json").write_text(json.dumps({"rows": day10}, indent=2), encoding="utf-8")

    lines = [
        "Orog/TISR sensitivity comparison:",
        f"- checkpoint: {args.checkpoint}",
        f"- fixed rollout steps: {int(args.fixed_rollout_steps)}",
        f"- modes: {', '.join(modes)}",
        f"- variables: {', '.join(variables)}",
        "",
        "Day-10 summary:",
    ]
    for row in day10:
        if row["mode"] == "normal":
            continue
        pct = row["rmse_percent_change_day10"]
        pct_text = "n/a" if pct is None else f"{float(pct):+.3f}%"
        lines.append(
            f"- {row['mode']} {row['variable']}: RMSE {row['normal_rmse_day10']:.6g} -> {row['rmse_day10']:.6g} "
            f"({pct_text}); ACC {row['normal_acc_day10']:.6g} -> {row['acc_day10']:.6g} "
            f"({row['acc_delta_day10']:+.6g})"
        )
    (output_dir / "comparison_report.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required for sensitivity plots.") from exc
    return plt


def _plot_day10_bars(
    plot_dir: Path,
    day10: list[dict[str, Any]],
    modes: list[str],
    variables: list[str],
    metric_key: str,
    title: str,
    ylabel: str,
    filename: str,
    plot_format: str,
) -> Path:
    plt = _load_matplotlib()
    mode_labels = [mode for mode in modes if mode != "normal"]
    ncols = 2
    nrows = int(np.ceil(len(variables) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(11.0, 1.2 * len(mode_labels)), 3.4 * nrows), squeeze=False)
    lookup = {(row["mode"], row["variable"]): row for row in day10}
    for idx, variable in enumerate(variables):
        ax = axes[idx // ncols][idx % ncols]
        key = _canonical_for_name(variable) or variable
        values = [
            np.nan if lookup.get((mode, key), {}).get(metric_key) is None else float(lookup[(mode, key)][metric_key])
            for mode in mode_labels
        ]
        ax.bar(np.arange(len(mode_labels)), values)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(key)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(len(mode_labels)))
        ax.set_xticklabels(mode_labels, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    for idx in range(len(variables), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path = plot_dir / f"{filename}.{plot_format}"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_heatmap(plot_dir: Path, day10: list[dict[str, Any]], modes: list[str], variables: list[str], plot_format: str) -> Path:
    plt = _load_matplotlib()
    mode_labels = [mode for mode in modes if mode != "normal"]
    variable_keys = [_canonical_for_name(variable) or variable for variable in variables]
    lookup = {(row["mode"], row["variable"]): row for row in day10}
    matrix = np.full((len(mode_labels), len(variable_keys)), np.nan, dtype=np.float64)
    for i, mode in enumerate(mode_labels):
        for j, variable in enumerate(variable_keys):
            value = lookup.get((mode, variable), {}).get("rmse_percent_change_day10")
            matrix[i, j] = np.nan if value is None else float(value)
    fig, ax = plt.subplots(figsize=(max(8.5, 0.8 * len(variable_keys)), max(5.0, 0.38 * len(mode_labels))))
    finite = matrix[np.isfinite(matrix)]
    vmax = max(1.0, float(np.nanmax(np.abs(finite)))) if finite.size else 1.0
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(variable_keys)))
    ax.set_xticklabels(variable_keys)
    ax.set_yticks(np.arange(len(mode_labels)))
    ax.set_yticklabels(mode_labels)
    ax.set_title("Day-10 RMSE percent change vs normal")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "nan" if not np.isfinite(value) else f"{value:+.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="RMSE percent change")
    fig.tight_layout()
    path = plot_dir / f"day10_rmse_percent_change_heatmap.{plot_format}"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_lead_group(
    plot_dir: Path,
    metrics_by_mode: dict[str, dict[str, Any]],
    variable: str,
    modes: list[str],
    metric_name: str,
    filename: str,
    plot_format: str,
    fixed_steps: int,
) -> Path:
    plt = _load_matplotlib()
    key = _canonical_for_name(variable) or variable
    leads = np.arange(1, int(fixed_steps) + 1)
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    for mode in modes:
        if mode not in metrics_by_mode:
            continue
        series = _series_array(metrics_by_mode[mode], key, metric_name)
        ax.plot(leads, series[: len(leads)], marker="o", linewidth=1.8, label=mode)
    ax.set_title(f"{key} {metric_name.upper()} vs lead")
    ax.set_xlabel("Lead time")
    ax.set_ylabel(metric_name.upper())
    if metric_name == "acc":
        ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(leads)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = plot_dir / f"{filename}.{plot_format}"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_plots(
    output_dir: Path,
    metrics_by_mode: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    day10: list[dict[str, Any]],
    modes: list[str],
    variables: list[str],
    fixed_steps: int,
    args: argparse.Namespace,
) -> list[Path]:
    if bool(args.skip_plots):
        return []
    plot_format = str(args.plot_format).lstrip(".").lower()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _plot_day10_bars(
            plot_dir,
            day10,
            modes,
            variables,
            "rmse_percent_change_day10",
            "Day-10 RMSE percent change vs normal",
            "RMSE percent change",
            "day10_rmse_percent_change_by_mode",
            plot_format,
        ),
        _plot_day10_bars(
            plot_dir,
            day10,
            modes,
            variables,
            "acc_delta_day10",
            "Day-10 ACC delta vs normal",
            "ACC delta",
            "day10_acc_delta_by_mode",
            plot_format,
        ),
        _plot_heatmap(plot_dir, day10, modes, variables, plot_format),
    ]
    for variable in variables:
        key = _canonical_for_name(variable) or variable
        representative = [mode for mode in DEFAULT_PLOT_MODES if mode in modes]
        paths.append(
            _plot_lead_group(
                plot_dir,
                metrics_by_mode,
                key,
                representative,
                "rmse",
                f"rmse_vs_lead_{_safe_stem(key)}",
                plot_format,
                fixed_steps,
            )
        )
        paths.append(
            _plot_lead_group(
                plot_dir,
                metrics_by_mode,
                key,
                representative,
                "acc",
                f"acc_vs_lead_{_safe_stem(key)}",
                plot_format,
                fixed_steps,
            )
        )
        for group_name, group_modes in MODE_GROUPS.items():
            present = [mode for mode in group_modes if mode in modes]
            if len(present) <= 1:
                continue
            paths.append(
                _plot_lead_group(
                    plot_dir,
                    metrics_by_mode,
                    key,
                    present,
                    "rmse",
                    f"rmse_vs_lead_{_safe_stem(key)}_{group_name}",
                    plot_format,
                    fixed_steps,
                )
            )
            paths.append(
                _plot_lead_group(
                    plot_dir,
                    metrics_by_mode,
                    key,
                    present,
                    "acc",
                    f"acc_vs_lead_{_safe_stem(key)}_{group_name}",
                    plot_format,
                    fixed_steps,
                )
            )
    return paths


def _effect_class(rmse_pct: float, acc_delta: float) -> str:
    rmse_abs = abs(float(rmse_pct))
    acc_abs = abs(float(acc_delta))
    if rmse_abs > 2.0 or acc_abs > 0.02:
        return "meaningful"
    if rmse_abs >= 0.5 or acc_abs >= 0.005:
        return "small"
    return "negligible"


def _worst_row(day10: list[dict[str, Any]], modes: list[str]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in day10
        if row["mode"] in modes
        and row.get("rmse_percent_change_day10") is not None
        and row.get("acc_delta_day10") is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: max(abs(float(row["rmse_percent_change_day10"])) / 2.0, abs(float(row["acc_delta_day10"])) / 0.02),
    )


def _write_interpretation_report(output_dir: Path, day10: list[dict[str, Any]], modes: list[str], variables: list[str]) -> None:
    override = _worst_row(day10, ["override_orog_tisr"])
    tisr = _worst_row(day10, [mode for mode in modes if mode.startswith("tisr_")])
    orog = _worst_row(day10, [mode for mode in modes if mode.startswith("orog_") and not mode.startswith("orog_tisr_")])
    combined = _worst_row(day10, [mode for mode in modes if mode.startswith("orog_tisr_")])

    def describe(title: str, row: dict[str, Any] | None) -> tuple[str, str]:
        if row is None:
            return title, f"{title}:\n  Not evaluated or no finite day-10 comparison was available.\n"
        rmse_pct = float(row["rmse_percent_change_day10"])
        acc_delta = float(row["acc_delta_day10"])
        level = _effect_class(rmse_pct, acc_delta)
        text = (
            f"{title}:\n"
            f"  Worst day-10 change: {row['mode']} / {row['variable']} changed RMSE by {rmse_pct:+.3f}% "
            f"and ACC by {acc_delta:+.6g}.\n"
            f"  This is {level}.\n"
        )
        return level, text

    override_level, override_text = describe("Override sensitivity", override)
    tisr_level, tisr_text = describe("TISR sensitivity", tisr)
    orog_level, orog_text = describe("OROG sensitivity", orog)
    combined_level, combined_text = describe("Combined OROG/TISR sensitivity", combined)

    if override_level == "negligible":
        override_conclusion = "Clean orog/tisr correction is not an immediate evaluation bottleneck."
    else:
        override_conclusion = "Clean orog/tisr correction changes metrics, so target handling may affect evaluation."

    corruption_levels = {tisr_level, orog_level, combined_level}
    if "meaningful" in corruption_levels:
        usage_conclusion = "At least one corruption mode has meaningful impact, suggesting the model uses these channels."
    elif "small" in corruption_levels:
        usage_conclusion = "Corruption has small impact, suggesting weak but nonzero channel usage."
    else:
        usage_conclusion = "Corruption is negligible, suggesting the model mostly ignores these channels as autoregressive inputs."

    if override_level != "negligible":
        priority = "high"
    elif "meaningful" in corruption_levels:
        priority = "medium: channels matter, but clean evaluation override did not change metrics."
    else:
        priority = "low"

    lines = [
        "Orog/TISR sensitivity interpretation report",
        "",
        "Thresholds:",
        "- negligible: |RMSE percent change| < 0.5% and |ACC delta| < 0.005",
        "- small: 0.5%-2% RMSE change or 0.005-0.02 ACC change",
        "- meaningful: >2% RMSE change or >0.02 ACC change",
        "",
        override_text,
        tisr_text,
        orog_text,
        combined_text,
        "Final conclusion:",
        f"  {override_conclusion}",
        f"  {usage_conclusion}",
        f"  Target-handling retraining priority: {priority}.",
    ]
    (output_dir / "sensitivity_interpretation_report.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate orog/tisr autoregressive sensitivity modes for one checkpoint.")
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
    parser.add_argument("--max_eval_initializations", default=None, type=int)
    parser.add_argument("--variables", nargs="*", default=None)
    parser.add_argument("--modes", nargs="*", default=None, help="Sensitivity modes to evaluate. Normal is added automatically if omitted.")
    parser.add_argument("--include_persistence", action="store_true")
    parser.add_argument("--climatology_path", default=None, type=str)
    parser.add_argument("--build_climatology_if_missing", action="store_true")
    parser.add_argument("--bootstrap_samples", default=0, type=int)
    parser.add_argument("--confidence_level", default=0.95, type=float)
    parser.add_argument("--bootstrap_seed", default=42, type=int)
    parser.add_argument("--noise_seed", default=123, type=int)
    parser.add_argument("--plot_confidence_intervals", action="store_true")
    parser.add_argument("--external_baseline_csv", nargs="*", default=None)
    parser.add_argument("--external_baseline_label", nargs="*", default=None)
    parser.add_argument("--external_baseline_params_m", nargs="*", default=None, type=float)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--debug_samples", default=3, type=int)
    parser.add_argument("--plot_format", default="png", type=str)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--output_dir", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if int(args.fixed_rollout_steps) <= 0:
        raise ValueError("--fixed_rollout_steps must be positive.")
    if args.max_eval_initializations is not None and int(args.max_eval_initializations) <= 0:
        raise ValueError("--max_eval_initializations must be positive when provided.")
    modes = _resolve_modes(args)
    variables = _parse_strings(args.variables, DEFAULT_VARIABLES)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Orog/TISR sensitivity test:")
    print(f"  modes: {modes}")
    print(f"  noise_seed: {int(args.noise_seed)}")
    print("  variable channels are resolved inside each evaluation mode.")

    for mode in modes:
        _run_mode(args, output_dir, mode)

    fixed_steps = int(args.fixed_rollout_steps)
    metrics_by_mode = {mode: _load_mode_metrics(output_dir, mode, fixed_steps) for mode in modes}
    rows = _comparison_rows(metrics_by_mode, modes, variables, fixed_steps)
    day10 = _day10_rows(rows, fixed_steps)
    _write_comparison_outputs(output_dir, rows, day10, modes, variables, args)
    plot_paths = _write_plots(output_dir, metrics_by_mode, rows, day10, modes, variables, fixed_steps, args)
    _write_interpretation_report(output_dir, day10, modes, variables)

    print(f"Saved sensitivity outputs to: {output_dir}")
    print(f"Saved comparison CSV: {output_dir / 'comparison_summary.csv'}")
    print(f"Saved day-10 CSV: {output_dir / 'comparison_day10.csv'}")
    print(f"Saved interpretation report: {output_dir / 'sensitivity_interpretation_report.txt'}")
    if plot_paths:
        print(f"Saved plots under: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
