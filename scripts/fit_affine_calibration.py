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

from src.affine_calibration import (  # noqa: E402
    DEFAULT_CALIBRATION_VARIABLES,
    VALID_CALIBRATION_APPLY_MODES,
    coefficient_rows,
    finalize_affine_coefficients,
    save_affine_coefficients,
    validate_apply_mode,
    validate_fit_split,
)
from src.config import YParams, setup_logging  # noqa: E402
from src.evaluator import EvalConfig, GraphWeatherEvaluator, _canonical_for_name, run_evaluation_from_params  # noqa: E402


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
                    "weather_dual_resolution_l3_hidden128_dense_l3k24_fixed_orog.yaml": "raw_l3_hidden128_dense_l3k24_fixed_orog",
                    "weather_dual_resolution_l3_hidden128_scalar_gated_skip_fixed_orog.yaml": "raw_l3_hidden128_scalar_gated_skip_fixed_orog",
                    "weather_dual_resolution_l3_hidden128_scalar_gated_pooling_fixed_orog.yaml": "raw_l3_hidden128_scalar_gated_pooling_fixed_orog",
                    "weather_dual_resolution_l3_hidden128_lead_conditioned_fixed_orog.yaml": "raw_l3_hidden128_lead_conditioned_fixed_orog",
                }
                config_name = config_map.get(Path(args.config).name)
        elif config_name is None:
            config_name = args.config
    yaml_path = yaml_path or str(project_root / "configs" / "gnn_5p625.yaml")
    config_name = config_name or "raw_5p625"
    return str(Path(yaml_path).expanduser().resolve()), config_name


def _parse_strings(values: list[str] | None, default: list[str] | None = None) -> list[str]:
    if values is None:
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


def _apply_common_eval_params(
    params: Any,
    args: argparse.Namespace,
    *,
    split: str,
    output_dir: Path,
    calibration_path: Path | None = None,
    calibration_apply_mode: str | None = None,
) -> None:
    params["eval_split"] = str(split)
    params["test_dataset_path"] = _split_data_path(params, str(split))
    params["eval_output_dir"] = str(output_dir)
    params["eval_checkpoint_path"] = str(args.checkpoint)
    params["eval_fixed_rollout_steps"] = int(args.fixed_rollout_steps)
    params["eval_forecast_steps"] = [int(args.fixed_rollout_steps)]
    params["eval_compare_stage_checkpoints"] = False
    params["eval_selection"] = str(args.selection)
    params["eval_ic_stride"] = int(args.stride)
    params["eval_start_timestep"] = int(args.start_timestep)
    params["eval_plot_variables"] = _parse_strings(args.variables, DEFAULT_CALIBRATION_VARIABLES)
    params["plot_persistence"] = bool(args.include_persistence)
    params["bootstrap_samples"] = int(args.bootstrap_samples)
    params["confidence_level"] = float(args.confidence_level)
    params["bootstrap_seed"] = int(args.bootstrap_seed)
    params["plot_confidence_intervals"] = bool(args.plot_confidence_intervals)
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
    if args.experiment_dir:
        params["experiment_dir"] = str(Path(args.experiment_dir).expanduser().resolve())
    if calibration_path is not None:
        params["affine_calibration_path"] = str(calibration_path)
        params["calibration_apply_mode"] = validate_apply_mode(calibration_apply_mode or "output_only")
    else:
        params["affine_calibration_path"] = None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fit_coefficients(args: argparse.Namespace, yaml_path: str, config_name: str, output_dir: Path) -> Path:
    validate_fit_split(str(args.fit_split), allow_fit_on_test=bool(args.allow_fit_on_test))
    fit_log_dir = output_dir / "fit_validation"
    fit_log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(rank=0, log_file=str(fit_log_dir / "fit_affine_calibration.log"))
    logging.info("Fitting affine calibration on split=%s", args.fit_split)
    params = YParams(yaml_path, config_name, resolution_mode=args.resolution_mode)
    _apply_common_eval_params(params, args, split=str(args.fit_split), output_dir=fit_log_dir)
    params["plot_persistence"] = False
    params["bootstrap_samples"] = 0
    params.log()

    cfg = EvalConfig.from_params(params)
    evaluator = GraphWeatherEvaluator(cfg, logger=logging)
    ds, fields, total_t, height, width, _climatology_norm, lat_weights, _lat_weights_np = evaluator._prepare_context()
    try:
        checkpoint_path = evaluator._resolve_checkpoint_path(cfg.checkpoint_path)
        evaluator.model = evaluator._load_model(height, width, checkpoint_path)
        variables = evaluator._resolve_output_variables(
            _parse_strings(args.variables, DEFAULT_CALIBRATION_VARIABLES),
            calibrate_all_dynamic_variables=bool(args.calibrate_all_dynamic_variables),
        )
        if not variables:
            raise ValueError("No variables were resolved for affine calibration.")
        accumulator = evaluator.accumulate_affine_calibration_statistics(
            fields,
            total_t,
            height,
            width,
            int(args.fixed_rollout_steps),
            lat_weights,
            variables,
            label=f"{args.fit_split} fit",
        )
        metadata = {
            "fit_split": str(args.fit_split),
            "eval_split": str(args.eval_split),
            "variables": [str(v["canonical_name"]) for v in variables],
            "fixed_rollout_steps": int(args.fixed_rollout_steps),
            "calibration_space": "normalized",
            "created_from_checkpoint": str(Path(args.checkpoint).expanduser()),
            "created_from_config": str(yaml_path),
            "config_name": str(config_name),
            "resolution_mode": str(args.resolution_mode),
            "selection": dict(evaluator._selection_metadata),
            "selected_initial_conditions": int(len(accumulator["initial_condition_indices"])),
            "initial_condition_indices": [int(x) for x in accumulator["initial_condition_indices"].tolist()],
        }
        payload = finalize_affine_coefficients(accumulator, variables, metadata, eps=float(args.eps))
    finally:
        ds.close()

    json_path = output_dir / "affine_coefficients.json"
    csv_path = output_dir / "affine_coefficients.csv"
    save_affine_coefficients(payload, json_path, csv_path)
    rows = coefficient_rows(payload)
    validation_rows: list[dict[str, Any]] = []
    for row in rows:
        before = row["fit_rmse_before"]
        after = row["fit_rmse_after"]
        delta = None if before is None or after is None else float(after) - float(before)
        percent = None if delta is None or float(before) == 0.0 else 100.0 * delta / float(before)
        validation_rows.append({**row, "fit_rmse_delta": delta, "fit_rmse_percent_change": percent})
    _write_csv(output_dir / "validation_fit_summary.csv", validation_rows)
    (output_dir / "validation_fit_summary.json").write_text(
        json.dumps({"metadata": payload["metadata"], "rows": validation_rows}, indent=2),
        encoding="utf-8",
    )
    logging.info("Saved affine coefficients: %s", json_path)
    logging.info("Saved affine coefficient CSV: %s", csv_path)
    return json_path


def _run_eval(
    args: argparse.Namespace,
    yaml_path: str,
    config_name: str,
    output_dir: Path,
    *,
    split: str,
    calibration_path: Path | None,
    calibration_apply_mode: str | None,
) -> None:
    params = YParams(yaml_path, config_name, resolution_mode=args.resolution_mode)
    _apply_common_eval_params(
        params,
        args,
        split=split,
        output_dir=output_dir,
        calibration_path=calibration_path,
        calibration_apply_mode=calibration_apply_mode,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(rank=0, log_file=str(output_dir / "evaluation.log"))
    if calibration_path is None:
        logging.info("Running normal evaluation without affine calibration.")
        print(f"Running normal evaluation: {output_dir}")
    else:
        logging.info("Running affine-calibrated evaluation mode=%s", calibration_apply_mode)
        print(f"Running affine-calibrated evaluation ({calibration_apply_mode}): {output_dir}")
    params.log()
    run_evaluation_from_params(params, logger=logging)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_metrics(eval_dir: Path, fixed_steps: int) -> dict[str, Any]:
    path = eval_dir / f"fixed{int(fixed_steps)}_global_best_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation metrics JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["checkpoints"]["global_best"]["metrics"]


def _metric_series(metrics: dict[str, Any], variable: str, metric_name: str) -> np.ndarray:
    key = _canonical_for_name(variable) or variable
    if key not in metrics:
        available = ", ".join(sorted(metrics))
        raise ValueError(f"Variable {key!r} is missing from evaluation JSON. Available: {available}")
    values = metrics[key][metric_name]["mean"]
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


def _comparison_rows(
    normal_metrics: dict[str, Any],
    calibrated_metrics_by_mode: dict[str, dict[str, Any]],
    variables: list[str],
    fixed_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, calibrated_metrics in calibrated_metrics_by_mode.items():
        for variable in variables:
            key = _canonical_for_name(variable) or variable
            normal_rmse = _metric_series(normal_metrics, key, "rmse")
            normal_acc = _metric_series(normal_metrics, key, "acc")
            calibrated_rmse = _metric_series(calibrated_metrics, key, "rmse")
            calibrated_acc = _metric_series(calibrated_metrics, key, "acc")
            for lead_idx in range(int(fixed_steps)):
                n_rmse = _json_number(normal_rmse[lead_idx])
                c_rmse = _json_number(calibrated_rmse[lead_idx])
                n_acc = _json_number(normal_acc[lead_idx])
                c_acc = _json_number(calibrated_acc[lead_idx])
                rmse_delta = None if n_rmse is None or c_rmse is None else float(c_rmse - n_rmse)
                acc_delta = None if n_acc is None or c_acc is None else float(c_acc - n_acc)
                rows.append(
                    {
                        "calibration_apply_mode": mode,
                        "variable": key,
                        "lead_time": int(lead_idx + 1),
                        "normal_rmse": n_rmse,
                        "calibrated_rmse": c_rmse,
                        "rmse_delta": rmse_delta,
                        "rmse_percent_change": (
                            None if rmse_delta is None or n_rmse is None or n_rmse == 0.0 else 100.0 * rmse_delta / n_rmse
                        ),
                        "normal_acc": n_acc,
                        "calibrated_acc": c_acc,
                        "acc_delta": acc_delta,
                    }
                )
    return rows


def _day_rows(rows: list[dict[str, Any]], lead: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["lead_time"]) == int(lead)]


def _summary_by_mode(day_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for mode in sorted({str(row["calibration_apply_mode"]) for row in day_rows}):
        mode_rows = [row for row in day_rows if str(row["calibration_apply_mode"]) == mode]
        rmse_pct = [float(row["rmse_percent_change"]) for row in mode_rows if row["rmse_percent_change"] is not None]
        acc_delta = [float(row["acc_delta"]) for row in mode_rows if row["acc_delta"] is not None]
        summary[mode] = {
            "mean_day10_rmse_percent_change": None if not rmse_pct else float(np.nanmean(rmse_pct)),
            "mean_day10_acc_delta": None if not acc_delta else float(np.nanmean(acc_delta)),
        }
    return summary


def _plot_comparison(
    output_dir: Path,
    normal_metrics: dict[str, Any],
    calibrated_metrics_by_mode: dict[str, dict[str, Any]],
    coeff_payload: dict[str, Any],
    variables: list[str],
    fixed_steps: int,
    *,
    plot_format: str,
    skip_plots: bool,
) -> list[Path]:
    if skip_plots:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fmt = plot_format.lstrip(".").lower()
    leads = np.arange(1, int(fixed_steps) + 1)
    modes = list(calibrated_metrics_by_mode)
    plot_paths: list[Path] = []

    def save(fig: Any, name: str) -> None:
        path = plot_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        plot_paths.append(path)

    x = np.arange(len(variables))
    width = min(0.22, 0.8 / max(1, len(modes) + 1))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    normal_day = [_metric_series(normal_metrics, _canonical_for_name(v) or v, "rmse")[fixed_steps - 1] for v in variables]
    ax.bar(x - width * len(modes) / 2, normal_day, width=width, label="normal")
    for idx, mode in enumerate(modes):
        values = [_metric_series(calibrated_metrics_by_mode[mode], _canonical_for_name(v) or v, "rmse")[fixed_steps - 1] for v in variables]
        ax.bar(x - width * len(modes) / 2 + width * (idx + 1), values, width=width, label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels([_canonical_for_name(v) or v for v in variables], rotation=25, ha="right")
    ax.set_ylabel(f"Day-{fixed_steps} RMSE")
    ax.set_title(f"Day-{fixed_steps} RMSE before/after affine calibration")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "day10_rmse_before_after")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for idx, mode in enumerate(modes):
        pct_values = []
        for variable in variables:
            n = _metric_series(normal_metrics, _canonical_for_name(variable) or variable, "rmse")[fixed_steps - 1]
            c = _metric_series(calibrated_metrics_by_mode[mode], _canonical_for_name(variable) or variable, "rmse")[fixed_steps - 1]
            pct_values.append(np.nan if n == 0 else 100.0 * (c - n) / n)
        ax.bar(x + width * (idx - (len(modes) - 1) / 2), pct_values, width=width, label=mode)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([_canonical_for_name(v) or v for v in variables], rotation=25, ha="right")
    ax.set_ylabel("RMSE percent change")
    ax.set_title(f"Day-{fixed_steps} RMSE percent change")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "day10_rmse_percent_change")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    normal_day_acc = [_metric_series(normal_metrics, _canonical_for_name(v) or v, "acc")[fixed_steps - 1] for v in variables]
    ax.bar(x - width * len(modes) / 2, normal_day_acc, width=width, label="normal")
    for idx, mode in enumerate(modes):
        values = [_metric_series(calibrated_metrics_by_mode[mode], _canonical_for_name(v) or v, "acc")[fixed_steps - 1] for v in variables]
        ax.bar(x - width * len(modes) / 2 + width * (idx + 1), values, width=width, label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels([_canonical_for_name(v) or v for v in variables], rotation=25, ha="right")
    ax.set_ylabel(f"Day-{fixed_steps} ACC")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"Day-{fixed_steps} ACC before/after affine calibration")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "day10_acc_before_after")

    for variable in variables:
        key = _canonical_for_name(variable) or variable
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        ax.plot(leads, _metric_series(normal_metrics, key, "rmse")[:fixed_steps], marker="o", label="normal")
        for mode in modes:
            ax.plot(leads, _metric_series(calibrated_metrics_by_mode[mode], key, "rmse")[:fixed_steps], marker="s", label=mode)
        ax.set_title(f"{key} RMSE vs lead")
        ax.set_xlabel("Lead time")
        ax.set_ylabel("RMSE")
        ax.set_xticks(leads)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        save(fig, f"rmse_vs_lead_{_safe_stem(key)}")

        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        ax.plot(leads, _metric_series(normal_metrics, key, "acc")[:fixed_steps], marker="o", label="normal")
        for mode in modes:
            ax.plot(leads, _metric_series(calibrated_metrics_by_mode[mode], key, "acc")[:fixed_steps], marker="s", label=mode)
        ax.set_title(f"{key} ACC vs lead")
        ax.set_xlabel("Lead time")
        ax.set_ylabel("ACC")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks(leads)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        save(fig, f"acc_vs_lead_{_safe_stem(key)}")

    coeffs = dict(coeff_payload.get("coefficients", {}))
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for variable in variables:
        key = _canonical_for_name(variable) or variable
        if key not in coeffs:
            continue
        ax.plot(leads, [coeffs[key][str(int(lead))]["a"] for lead in leads], marker="o", label=key)
    ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
    ax.set_title("Affine scale a by lead")
    ax.set_xlabel("Lead time")
    ax.set_ylabel("a")
    ax.set_xticks(leads)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "affine_scale_by_lead")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for variable in variables:
        key = _canonical_for_name(variable) or variable
        if key not in coeffs:
            continue
        ax.plot(leads, [coeffs[key][str(int(lead))]["b"] for lead in leads], marker="o", label=key)
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax.set_title("Affine bias b by lead")
    ax.set_xlabel("Lead time")
    ax.set_ylabel("b (normalized units)")
    ax.set_xticks(leads)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "affine_bias_by_lead")

    return plot_paths


def _magnitude_label_rmse(improvement_percent: float) -> str:
    value = abs(float(improvement_percent))
    if value < 0.5:
        return "negligible"
    if value <= 2.0:
        return "small"
    return "meaningful"


def _magnitude_label_acc(delta: float) -> str:
    value = abs(float(delta))
    if value < 0.005:
        return "negligible"
    if value <= 0.02:
        return "small"
    return "meaningful"


def _reference_lines(args: argparse.Namespace, day_rows: list[dict[str, Any]]) -> list[str]:
    if not args.reference_comparison_dir:
        return ["Hidden128/hidden160 cross-model comparison: not available; no --reference_comparison_dir was provided."]
    ref_path = Path(args.reference_comparison_dir).expanduser() / "comparison_day10.csv"
    if not ref_path.is_file():
        return [f"Hidden128/hidden160 cross-model comparison: unavailable; missing {ref_path}."]
    with ref_path.open("r", encoding="utf-8") as f:
        ref_rows = list(csv.DictReader(f))
    lines = [f"Reference comparison: {ref_path}"]
    for mode in sorted({str(row["calibration_apply_mode"]) for row in day_rows}):
        current = {str(row["variable"]): row for row in day_rows if str(row["calibration_apply_mode"]) == mode}
        reference = {str(row["variable"]): row for row in ref_rows if str(row.get("calibration_apply_mode")) == mode}
        shared = sorted(set(current) & set(reference))
        if not shared:
            lines.append(f"- {mode}: no shared variables with reference.")
            continue
        worse = []
        better = []
        for variable in shared:
            c = float(current[variable]["calibrated_rmse"])
            r = float(reference[variable]["calibrated_rmse"])
            (worse if c > r else better).append(variable)
        lines.append(
            f"- {mode}: current calibrated RMSE is higher than reference for {len(worse)}/{len(shared)} variables "
            f"({', '.join(worse) if worse else 'none'})."
        )
        if better:
            lines.append(f"  lower than reference for: {', '.join(better)}")
    return lines


def _write_comparison_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    rows: list[dict[str, Any]],
    plot_paths: list[Path],
) -> None:
    _write_csv(output_dir / "comparison_summary.csv", rows)
    day_rows = _day_rows(rows, int(args.fixed_rollout_steps))
    _write_csv(output_dir / "comparison_day10.csv", day_rows)
    summary = _summary_by_mode(day_rows)
    payload = {
        "checkpoint": str(args.checkpoint),
        "fixed_rollout_steps": int(args.fixed_rollout_steps),
        "fit_split": str(args.fit_split),
        "eval_split": str(args.eval_split),
        "summary_by_mode": summary,
        "comparison_rows": rows,
        "day10_rows": day_rows,
        "plots": [str(path) for path in plot_paths],
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "Affine calibration verdict:",
    ]
    best_mode = None
    best_rmse = None
    for mode, mode_summary in summary.items():
        value = mode_summary.get("mean_day10_rmse_percent_change")
        if value is None:
            continue
        if best_rmse is None or float(value) < float(best_rmse):
            best_rmse = float(value)
            best_mode = mode
    if best_rmse is None:
        verdict = "[MIXED]"
    elif best_rmse < -2.0:
        verdict = "[USEFUL]"
    elif best_rmse > 0.5:
        verdict = "[NOT USEFUL]"
    else:
        verdict = "[MIXED]"
    lines[0] += f"\n{verdict}"
    lines.extend(["", "Evidence:"])
    for mode, mode_summary in summary.items():
        rmse_pct = mode_summary.get("mean_day10_rmse_percent_change")
        acc_delta = mode_summary.get("mean_day10_acc_delta")
        if rmse_pct is None or acc_delta is None:
            lines.append(f"- {mode}: insufficient finite metrics.")
            continue
        improvement = -float(rmse_pct)
        lines.append(
            f"- {mode}: mean day-10 RMSE changed by {float(rmse_pct):.3f}% "
            f"({_magnitude_label_rmse(improvement)} {'improvement' if improvement > 0 else 'regression'}); "
            f"mean day-10 ACC changed by {float(acc_delta):+.5f} ({_magnitude_label_acc(float(acc_delta))})."
        )
    if day_rows:
        best = min(day_rows, key=lambda row: float(row["rmse_percent_change"]) if row["rmse_percent_change"] is not None else float("inf"))
        worst = max(day_rows, key=lambda row: float(row["rmse_percent_change"]) if row["rmse_percent_change"] is not None else float("-inf"))
        lines.append(
            f"- Best day-10 RMSE change: {best['variable']} {best['calibration_apply_mode']} "
            f"{float(best['rmse_percent_change']):.3f}%."
        )
        lines.append(
            f"- Worst day-10 RMSE change: {worst['variable']} {worst['calibration_apply_mode']} "
            f"{float(worst['rmse_percent_change']):.3f}%."
        )
    for mode in sorted({str(row["calibration_apply_mode"]) for row in rows}):
        mode_rows = [row for row in rows if str(row["calibration_apply_mode"]) == mode]
        by_lead = []
        for lead in range(1, int(args.fixed_rollout_steps) + 1):
            values = [float(row["rmse_percent_change"]) for row in mode_rows if int(row["lead_time"]) == lead and row["rmse_percent_change"] is not None]
            if values:
                by_lead.append((lead, float(np.nanmean(values))))
        if by_lead:
            lead, value = min(by_lead, key=lambda item: item[1])
            lines.append(f"- {mode}: strongest average RMSE improvement occurs at lead {lead} ({value:.3f}%).")
    if "output_only" in summary and "autoregressive_state" in summary:
        out = summary["output_only"].get("mean_day10_rmse_percent_change")
        ar = summary["autoregressive_state"].get("mean_day10_rmse_percent_change")
        if out is not None and ar is not None:
            better = "output_only" if float(out) <= float(ar) else "autoregressive_state"
            lines.append(f"- Output-only vs autoregressive-state: {better} has the better mean day-10 RMSE change.")
    lines.extend(_reference_lines(args, day_rows))
    lines.extend(["", "Recommendation:"])
    if verdict == "[USEFUL]":
        lines.append("Keep affine calibration as a cheap post-hoc evaluation baseline; do not treat it as a replacement for retraining.")
    elif verdict == "[NOT USEFUL]":
        lines.append("Do not prioritize this calibration path for future experiments unless diagnostics show variable-specific value.")
    else:
        lines.append("Use the per-variable curves to decide; the aggregate result is mixed or small.")
    (output_dir / "comparison_report.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default=str(project_root / "configs" / "gnn_5p625.yaml"), type=str)
    parser.add_argument("--config", default="raw_5p625", type=str)
    parser.add_argument("--config_name", default=None, type=str)
    parser.add_argument("--resolution_mode", default=None, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--experiment_dir", default=None, type=str)
    parser.add_argument("--fit_split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--eval_split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--allow_fit_on_test", action="store_true")
    parser.add_argument("--fixed_rollout_steps", default=10, type=int)
    parser.add_argument("--selection", default="stride", choices=["first_n", "stride", "all"])
    parser.add_argument("--stride", default=7, type=int)
    parser.add_argument("--start_timestep", default=1, type=int)
    parser.add_argument("--n_initial_conditions", default=None, type=int)
    parser.add_argument("--max_initial_conditions", default=None, type=int)
    parser.add_argument("--max_eval_initializations", default=None, type=int)
    parser.add_argument("--start_date", default=None, type=str)
    parser.add_argument("--end_date", default=None, type=str)
    parser.add_argument("--variables", nargs="*", default=None)
    parser.add_argument("--calibrate_all_dynamic_variables", action="store_true")
    parser.add_argument("--climatology_path", default=None, type=str)
    parser.add_argument("--build_climatology_if_missing", action="store_true")
    parser.add_argument("--include_persistence", action="store_true")
    parser.add_argument("--bootstrap_samples", default=0, type=int)
    parser.add_argument("--confidence_level", default=0.95, type=float)
    parser.add_argument("--bootstrap_seed", default=42, type=int)
    parser.add_argument("--plot_confidence_intervals", action="store_true")
    parser.add_argument("--calibration_apply_modes", nargs="*", default=list(VALID_CALIBRATION_APPLY_MODES))
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--eps", default=1.0e-12, type=float)
    parser.add_argument("--plot_format", default="png", type=str)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--reference_comparison_dir", default=None, type=str)
    args = parser.parse_args()

    modes = [validate_apply_mode(mode) for mode in _parse_strings(args.calibration_apply_modes, list(VALID_CALIBRATION_APPLY_MODES))]
    modes = list(dict.fromkeys(modes))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path, config_name = _resolve_config_args(args)
    variables = _parse_strings(args.variables, DEFAULT_CALIBRATION_VARIABLES)

    coeff_path = _fit_coefficients(args, yaml_path, config_name, output_dir)

    _run_eval(
        args,
        yaml_path,
        config_name,
        output_dir / "test_normal",
        split=str(args.eval_split),
        calibration_path=None,
        calibration_apply_mode=None,
    )
    calibrated_metrics_by_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        mode_dir = output_dir / f"test_calibrated_{mode}"
        _run_eval(
            args,
            yaml_path,
            config_name,
            mode_dir,
            split=str(args.eval_split),
            calibration_path=coeff_path,
            calibration_apply_mode=mode,
        )
        calibrated_metrics_by_mode[mode] = _load_metrics(mode_dir, int(args.fixed_rollout_steps))

    normal_metrics = _load_metrics(output_dir / "test_normal", int(args.fixed_rollout_steps))
    rows = _comparison_rows(normal_metrics, calibrated_metrics_by_mode, variables, int(args.fixed_rollout_steps))
    coeff_payload = json.loads(coeff_path.read_text(encoding="utf-8"))
    plot_paths = _plot_comparison(
        output_dir,
        normal_metrics,
        calibrated_metrics_by_mode,
        coeff_payload,
        variables,
        int(args.fixed_rollout_steps),
        plot_format=str(args.plot_format),
        skip_plots=bool(args.skip_plots),
    )
    _write_comparison_outputs(args, output_dir, rows, plot_paths)


if __name__ == "__main__":
    main()
