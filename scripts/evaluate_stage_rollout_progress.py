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
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.config import YParams, setup_logging
from src.evaluator import (
    EXTERNAL_BASELINE_FAIRNESS_NOTE,
    EvalConfig,
    ExternalBaseline,
    GraphWeatherEvaluator,
)


VARIABLE_ALIASES: dict[str, list[str]] = {
    "msl": ["msl", "mslp", "mean_sea_level_pressure", "mean_sea_level_pressure_surface"],
    "t2m": ["t2m", "2m_temperature", "temperature_2m"],
    "t850": ["t850", "t_850", "temperature_850", "t@850"],
    "z500": ["z500", "z_500", "z@500", "geopotential_500"],
    "tisr": ["tisr", "top_incoming_solar_radiation"],
    "orog": ["orog", "orography", "geopotential_at_surface", "surface_geopotential"],
    "lsm": ["lsm", "land_sea_mask", "land_mask"],
    "q700": ["q700", "q_700", "specific_humidity_700"],
    "u850": ["u850", "u_850", "u_component_wind_850"],
}


def _parse_ints(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return default
    parsed: list[int] = []
    for value in values:
        for chunk in str(value).replace(",", " ").split():
            if chunk:
                parsed.append(int(chunk))
    return parsed or default


def _parse_strings(values: list[str] | None, default: list[str]) -> list[str]:
    if not values:
        return default
    parsed: list[str] = []
    for value in values:
        parsed.extend(chunk.strip() for chunk in str(value).replace(",", " ").split() if chunk.strip())
    return parsed or default


def _norm_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _canonical_for_name(name: str) -> str | None:
    norm = _norm_name(name)
    for canonical, aliases in VARIABLE_ALIASES.items():
        if norm == _norm_name(canonical) or norm in {_norm_name(alias) for alias in aliases}:
            return canonical
    return None


def _resolve_config_args(args: argparse.Namespace) -> tuple[str, str]:
    default_yaml = project_root / "configs" / "gnn_5p625.yaml"
    yaml_path = args.yaml_config
    config_name = args.config_name
    if args.config:
        looks_like_yaml = args.config.endswith((".yaml", ".yml")) or Path(args.config).exists()
        if looks_like_yaml:
            yaml_path = args.config
            if config_name is None:
                if Path(args.config).name == "weather_dual_resolution.yaml":
                    config_name = "raw"
                elif Path(args.config).name == "weather_dual_resolution_l3.yaml":
                    config_name = "raw_l3"
                elif Path(args.config).name == "weather_dual_resolution_l3_stage_warmup_cosine.yaml":
                    config_name = "raw_l3_stage_warmup_cosine"
                elif Path(args.config).name == "weather_dual_resolution_l3_blocks3.yaml":
                    config_name = "raw_l3_blocks3"
                elif Path(args.config).name == "weather_dual_resolution_l3_heavy_unet.yaml":
                    config_name = "raw_l3_heavy_unet"
                elif Path(args.config).name == "weather_dual_resolution_l3_full_rollout.yaml":
                    config_name = "raw_l3_full_rollout"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128.yaml":
                    config_name = "raw_l3_hidden128"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden160.yaml":
                    config_name = "raw_l3_hidden160"
                elif Path(args.config).name == "weather_dual_resolution_l3_orog_tisr_fixed.yaml":
                    config_name = "raw_l3_orog_tisr_fixed"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128_orog_tisr_fixed.yaml":
                    config_name = "raw_l3_hidden128_orog_tisr_fixed"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden160_orog_tisr_fixed.yaml":
                    config_name = "raw_l3_hidden160_orog_tisr_fixed"
        elif config_name is None:
            config_name = args.config
    yaml_path = yaml_path or str(default_yaml)
    config_name = config_name or "raw_5p625"
    return str(Path(yaml_path).expanduser().resolve()), config_name


def _split_data_path(params: Any, split: str) -> str:
    if split == "train":
        return str(params.train_data_path)
    if split == "valid":
        return str(params.valid_data_path)
    if split == "test":
        return str(params.get("test_dataset_path", params.valid_data_path))
    raise ValueError(f"Unsupported split: {split}")


def _checkpoint_metadata(path: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    metadata = dict(checkpoint.get("metadata", {}))
    if "epoch" not in metadata:
        metadata["epoch"] = checkpoint.get("epoch", None)
    return metadata


def _build_eval_config(
    params: Any,
    args: argparse.Namespace,
    output_dir: Path,
    checkpoint_path: str,
) -> EvalConfig:
    params["experiment_dir"] = str(Path(args.experiment_dir).expanduser().resolve())
    params["test_dataset_path"] = _split_data_path(params, str(args.split))
    params["eval_output_dir"] = str(output_dir)
    params["eval_checkpoint_path"] = checkpoint_path
    params["eval_fixed_rollout_steps"] = int(args.fixed_rollout_steps)
    params["eval_forecast_steps"] = [int(args.fixed_rollout_steps)]
    params["eval_compare_stage_checkpoints"] = False
    params["plot_persistence"] = bool(args.include_persistence)
    params["eval_split"] = str(args.split)
    params["eval_selection"] = str(args.selection)
    params["eval_ic_stride"] = int(args.stride)
    params["eval_start_offset"] = int(args.start_offset)
    params["eval_plot_variables"] = _parse_strings(args.variables, ["z500", "t2m", "t850", "msl"])
    params["bootstrap_samples"] = int(args.bootstrap_samples)
    params["confidence_level"] = float(args.confidence_level)
    params["bootstrap_seed"] = int(args.bootstrap_seed)
    params["plot_confidence_intervals"] = bool(args.plot_confidence_intervals)
    if args.external_baseline_csv:
        params["external_baseline_csv"] = args.external_baseline_csv
    if args.external_baseline_label:
        params["external_baseline_label"] = args.external_baseline_label
    if args.external_baseline_params_m:
        params["external_baseline_params_m"] = args.external_baseline_params_m
    if args.start_date:
        params["eval_start_date"] = str(args.start_date)
    if args.end_date:
        params["eval_end_date"] = str(args.end_date)
    if args.n_initial_conditions is not None:
        params["n_initial_conditions"] = int(args.n_initial_conditions)
    if args.max_initial_conditions is not None:
        params["max_initial_conditions"] = int(args.max_initial_conditions)
    if args.climatology_path:
        params["climatology_path"] = str(args.climatology_path)
    if args.build_climatology_if_missing:
        params["build_climatology_if_missing"] = True
        params["compute_climatology"] = True
    if args.max_batches is not None:
        params["max_initial_conditions"] = int(args.max_batches)
    if args.device:
        params["eval_device"] = str(args.device)
    return EvalConfig.from_params(params)


def _discover_checkpoints(
    experiment_dir: Path,
    stages: list[int],
    include_global_best: bool,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for stage in stages:
        path = experiment_dir / f"best_ckpt_S{int(stage)}.tar"
        if not path.exists():
            print(f"WARNING: missing {path.name}, skipping trained S={stage} curve.")
            logging.warning("WARNING: missing %s, skipping trained S=%s curve.", path.name, stage)
            continue
        items.append(
            {
                "key": f"trained_S{int(stage)}",
                "label": f"trained S={int(stage)}",
                "stage": int(stage),
                "checkpoint_path": str(path),
            }
        )
    if include_global_best:
        path = experiment_dir / "best_ckpt.tar"
        if path.exists():
            items.append(
                {
                    "key": "global_best",
                    "label": "global best",
                    "stage": None,
                    "checkpoint_path": str(path),
                }
            )
        else:
            print(f"WARNING: missing global best checkpoint: {path.name}. Skipping.")
            logging.warning("WARNING: missing global best checkpoint: %s. Skipping.", path.name)
    return items


def _warn_if_identical_paths(items: list[dict[str, Any]]) -> None:
    paths = [str(Path(item["checkpoint_path"]).resolve()) for item in items if item.get("stage") is not None]
    if len(paths) > 1 and len(set(paths)) == 1:
        logging.warning("WARNING: all checkpoint paths are identical. Check stage checkpoint configuration.")
        print("WARNING: all checkpoint paths are identical. Check stage checkpoint configuration.")


def _resolve_variables(requested: list[str], variable_names: list[str], out_channels: list[int]) -> list[dict[str, Any]]:
    lower_names = {_norm_name(name): idx for idx, name in enumerate(variable_names)}
    resolved: list[dict[str, Any]] = []
    seen: set[int] = set()
    for token in requested:
        var_idx: int | None = None
        stripped = str(token).strip()
        if not stripped:
            continue
        if stripped.lower() == "all":
            for idx, name in enumerate(variable_names):
                if idx not in seen:
                    resolved.append(
                        {
                            "requested": stripped,
                            "name": name,
                            "canonical_name": _canonical_for_name(name) or name,
                            "variable_idx": idx,
                            "channel": int(out_channels[idx]),
                        }
                    )
                    seen.add(idx)
            continue
        if stripped.lstrip("-").isdigit():
            value = int(stripped)
            if 0 <= value < len(variable_names):
                var_idx = value
            elif value in out_channels:
                var_idx = out_channels.index(value)
        else:
            candidates = [_norm_name(stripped)]
            canonical = _canonical_for_name(stripped)
            if canonical:
                candidates.extend(_norm_name(alias) for alias in VARIABLE_ALIASES.get(canonical, []))
            for candidate in candidates:
                if candidate in lower_names:
                    var_idx = lower_names[candidate]
                    break
        if var_idx is None:
            raise ValueError(
                f"Could not resolve variable '{token}'. Available variables include: "
                f"{', '.join(variable_names[:12])}..."
            )
        if var_idx in seen:
            continue
        name = variable_names[var_idx]
        resolved.append(
            {
                "requested": stripped,
                "name": name,
                "canonical_name": _canonical_for_name(stripped) or _canonical_for_name(name) or name,
                "variable_idx": int(var_idx),
                "channel": int(out_channels[var_idx]),
            }
        )
        seen.add(var_idx)
    return resolved


def _metric_payload(metrics: dict[str, np.ndarray], var_idx: int, fixed_steps: int) -> dict[str, Any]:
    rmse = np.asarray(metrics["rmse"][:fixed_steps, var_idx], dtype=np.float64)
    acc = np.asarray(metrics["acc"][:fixed_steps, var_idx], dtype=np.float64)
    payload = {
        "rmse_by_lead": [float(x) for x in rmse],
        "acc_by_lead": [float(x) for x in acc],
        "rmse_avg_1_10": float(np.nanmean(rmse)),
        "rmse_day10": float(rmse[-1]),
        "rmse_final_lead": float(rmse[-1]),
        "acc_avg_1_10": float(np.nanmean(acc)),
        "acc_day10": float(acc[-1]),
        "acc_final_lead": float(acc[-1]),
        "final_lead": int(fixed_steps),
    }
    if "rmse_ci_lower" in metrics:
        rmse_lower = np.asarray(metrics["rmse_ci_lower"][:fixed_steps, var_idx], dtype=np.float64)
        rmse_upper = np.asarray(metrics["rmse_ci_upper"][:fixed_steps, var_idx], dtype=np.float64)
        acc_lower = np.asarray(metrics["acc_ci_lower"][:fixed_steps, var_idx], dtype=np.float64)
        acc_upper = np.asarray(metrics["acc_ci_upper"][:fixed_steps, var_idx], dtype=np.float64)
        payload.update(
            {
                "rmse_ci_lower_by_lead": [float(x) for x in rmse_lower],
                "rmse_ci_upper_by_lead": [float(x) for x in rmse_upper],
                "acc_ci_lower_by_lead": [float(x) for x in acc_lower],
                "acc_ci_upper_by_lead": [float(x) for x in acc_upper],
                "rmse_day10_ci_lower": float(rmse_lower[-1]),
                "rmse_day10_ci_upper": float(rmse_upper[-1]),
                "acc_day10_ci_lower": float(acc_lower[-1]),
                "acc_day10_ci_upper": float(acc_upper[-1]),
            }
        )
    return payload


def _check_identical_metric_curves(
    checkpoint_payloads: dict[str, dict[str, Any]],
    variables: list[dict[str, Any]],
) -> None:
    stage_payloads = [
        payload for key, payload in checkpoint_payloads.items() if key.startswith("trained_S")
    ]
    if len(stage_payloads) <= 1:
        return
    for var in variables:
        name = str(var["canonical_name"])
        first_rmse = stage_payloads[0]["metrics"][name]["rmse_by_lead"]
        first_acc = stage_payloads[0]["metrics"][name]["acc_by_lead"]
        if all(
            payload["metrics"][name]["rmse_by_lead"] == first_rmse
            and payload["metrics"][name]["acc_by_lead"] == first_acc
            for payload in stage_payloads[1:]
        ):
            general_message = "WARNING: all stage curves are identical. Check whether the same checkpoint or cached rollout result was reused."
            message = (
                f"WARNING: all stage curves are identical for {name}. "
                "Check whether the same checkpoint or cached rollout result was reused."
            )
            logging.warning(general_message)
            logging.warning(message)
            print(general_message)
            print(message)


def _safe_stem(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))
    return safe.strip("_") or "variable"


def _plot_lead_curves(
    output_dir: Path,
    variables: list[dict[str, Any]],
    checkpoint_payloads: dict[str, dict[str, Any]],
    persistence_payload: dict[str, Any] | None,
    external_baselines: list[ExternalBaseline],
    fixed_steps: int,
    plot_confidence_intervals: bool = False,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    leads = np.arange(1, fixed_steps + 1)
    for var in variables:
        name = str(var["canonical_name"])
        display = str(var["name"])
        safe = _safe_stem(name)
        for metric_name, y_label, filename in [
            ("rmse_by_lead", "RMSE", f"stage_eval_{safe}_rmse_by_lead.png"),
            ("acc_by_lead", "ACC", f"stage_eval_{safe}_acc_by_lead.png"),
        ]:
            fig, ax = plt.subplots(figsize=(9.5, 6.0))
            for payload in checkpoint_payloads.values():
                values = payload["metrics"][name][metric_name]
                ax.plot(leads, values, marker="o", linewidth=1.9, label=payload["label"])
                if plot_confidence_intervals:
                    lower_key = metric_name.replace("_by_lead", "_ci_lower_by_lead")
                    upper_key = metric_name.replace("_by_lead", "_ci_upper_by_lead")
                    lower = payload["metrics"][name].get(lower_key)
                    upper = payload["metrics"][name].get(upper_key)
                    if lower is not None and upper is not None:
                        ax.fill_between(leads, lower, upper, alpha=0.14)
            if persistence_payload and name in persistence_payload:
                values = persistence_payload[name].get(metric_name)
                if values is not None:
                    ax.plot(leads, values, marker="x", linewidth=1.7, linestyle="--", label="persistence")
                    if plot_confidence_intervals:
                        lower_key = metric_name.replace("_by_lead", "_ci_lower_by_lead")
                        upper_key = metric_name.replace("_by_lead", "_ci_upper_by_lead")
                        lower = persistence_payload[name].get(lower_key)
                        upper = persistence_payload[name].get(upper_key)
                        if lower is not None and upper is not None:
                            ax.fill_between(leads, lower, upper, alpha=0.12)
            for baseline in external_baselines:
                curve = baseline.curves.get(name)
                if curve is None:
                    logging.warning(
                        "WARNING: external baseline %s does not contain variable %s. Skipping %s external curve.",
                        baseline.label,
                        name,
                        name,
                    )
                    print(f"WARNING: external baseline {baseline.label} does not contain variable {name}. Skipping {name} external curve.")
                    continue
                curve_key = "rmse" if metric_name.startswith("rmse") else "acc"
                values = curve[curve_key][:fixed_steps]
                ax.plot(leads, values, marker="s", linewidth=1.7, linestyle="-.", label=baseline.label)
                lower_key = f"{curve_key}_ci_lower"
                upper_key = f"{curve_key}_ci_upper"
                if lower_key in curve and upper_key in curve:
                    ax.fill_between(leads, curve[lower_key][:fixed_steps], curve[upper_key][:fixed_steps], alpha=0.12)
            ax.set_title(f"{display} {y_label} | fixed {fixed_steps}-day evaluation by training rollout stage", fontsize=13)
            ax.set_xlabel("Lead time (days)", fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_xticks(leads)
            if metric_name.startswith("acc"):
                ax.set_ylim(-1.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            fig.tight_layout()
            fig.savefig(plot_dir / filename, dpi=160)
            plt.close(fig)


def _stage_payloads_sorted(checkpoint_payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    payloads = [payload for key, payload in checkpoint_payloads.items() if key.startswith("trained_S")]
    return sorted(payloads, key=lambda payload: int(payload["stage"]))


def _plot_summary_curves(
    output_dir: Path,
    variables: list[dict[str, Any]],
    checkpoint_payloads: dict[str, dict[str, Any]],
    fixed_steps: int,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    stage_payloads = _stage_payloads_sorted(checkpoint_payloads)
    if not stage_payloads:
        return
    stages = np.asarray([int(payload["stage"]) for payload in stage_payloads], dtype=np.int64)
    final_label = "day-10" if int(fixed_steps) == 10 else f"day-{int(fixed_steps)}"
    final_suffix = "day10" if int(fixed_steps) == 10 else f"day{int(fixed_steps)}"
    metric_specs = [
        ("rmse_day10", f"{final_label} RMSE", f"{final_suffix}_rmse_vs_stage"),
        ("acc_day10", f"{final_label} ACC", f"{final_suffix}_acc_vs_stage"),
        ("rmse_avg_1_10", "avg RMSE days 1-10", "avg_rmse_vs_stage"),
        ("acc_avg_1_10", "avg ACC days 1-10", "avg_acc_vs_stage"),
    ]
    for var in variables:
        name = str(var["canonical_name"])
        display = str(var["name"])
        safe = _safe_stem(name)
        for metric_key, y_label, suffix in metric_specs:
            values = [payload["metrics"][name][metric_key] for payload in stage_payloads]
            fig, ax = plt.subplots(figsize=(8.0, 5.2))
            ax.plot(stages, values, marker="o", linewidth=2.0)
            ax.set_title(f"{display} {y_label} vs training rollout stage", fontsize=13)
            ax.set_xlabel("training rollout stage", fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_xticks(stages)
            if metric_key.startswith("acc"):
                ax.set_ylim(-1.05, 1.05)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / f"summary_{safe}_{suffix}.png", dpi=160)
            plt.close(fig)

    for metric_key, y_label, suffix in metric_specs:
        values = []
        for payload in stage_payloads:
            values.append(float(np.nanmean([payload["metrics"][str(var["canonical_name"])][metric_key] for var in variables])))
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        ax.plot(stages, values, marker="o", linewidth=2.0)
        ax.set_title(f"All selected variables {y_label} vs training rollout stage", fontsize=13)
        ax.set_xlabel("training rollout stage", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xticks(stages)
        if metric_key.startswith("acc"):
            ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"summary_allvars_{suffix}.png", dpi=160)
        plt.close(fig)


def _write_summary_csv(
    output_dir: Path,
    variables: list[dict[str, Any]],
    checkpoint_payloads: dict[str, dict[str, Any]],
    persistence_payload: dict[str, Any] | None,
) -> None:
    rows: list[dict[str, Any]] = []
    for payload in checkpoint_payloads.values():
        for var in variables:
            name = str(var["canonical_name"])
            metric = payload["metrics"][name]
            persistence_metric = persistence_payload.get(name, {}) if persistence_payload else {}
            row = {
                "stage": "" if payload.get("stage") is None else int(payload["stage"]),
                "checkpoint_path": payload["checkpoint_path"],
                "checkpoint_epoch": payload.get("checkpoint_epoch"),
                "variable": name,
                "rmse_avg_1_10": metric["rmse_avg_1_10"],
                "rmse_day10": metric["rmse_day10"],
                "acc_avg_1_10": metric["acc_avg_1_10"],
                "acc_day10": metric["acc_day10"],
                "persistence_rmse_day10": persistence_metric.get("rmse_day10"),
                "persistence_acc_day10": persistence_metric.get("acc_day10"),
            }
            for key in ("rmse_day10_ci_lower", "rmse_day10_ci_upper", "acc_day10_ci_lower", "acc_day10_ci_upper"):
                if key in metric:
                    row[key] = metric[key]
            rows.append(row)
    path = output_dir / "stage_rollout_progress_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Saved summary CSV: %s", path)


def _json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _json_series(values: np.ndarray) -> list[float | None]:
    return [_json_number(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def _external_baselines_payload(
    external_baselines: list[ExternalBaseline],
    variables: list[dict[str, Any]],
    fixed_steps: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for baseline in external_baselines:
        item: dict[str, Any] = {
            "params_m": baseline.params_m,
            "source_csv": baseline.source_csv,
            "rmse": {},
            "acc": {},
        }
        for var in variables:
            name = str(var["canonical_name"])
            curve = baseline.curves.get(name)
            if curve is None:
                continue
            item["rmse"][name] = _json_series(curve["rmse"][:fixed_steps])
            item["acc"][name] = _json_series(curve["acc"][:fixed_steps])
            if "rmse_ci_lower" in curve:
                item.setdefault("rmse_lower", {})[name] = _json_series(curve["rmse_ci_lower"][:fixed_steps])
                item.setdefault("rmse_upper", {})[name] = _json_series(curve["rmse_ci_upper"][:fixed_steps])
            if "acc_ci_lower" in curve:
                item.setdefault("acc_lower", {})[name] = _json_series(curve["acc_ci_lower"][:fixed_steps])
                item.setdefault("acc_upper", {})[name] = _json_series(curve["acc_ci_upper"][:fixed_steps])
        payload[baseline.label] = item
    return payload


def _write_external_comparison_summary(
    output_dir: Path,
    variables: list[dict[str, Any]],
    checkpoint_payloads: dict[str, dict[str, Any]],
    external_baselines: list[ExternalBaseline],
    fixed_steps: int,
) -> None:
    if not external_baselines:
        return
    stage_payloads = _stage_payloads_sorted(checkpoint_payloads)
    if not stage_payloads:
        return
    rows: list[dict[str, Any]] = []
    lines = [EXTERNAL_BASELINE_FAIRNESS_NOTE, ""]
    for baseline in external_baselines:
        lines.append(f"Compared with {baseline.label}:")
        for var in variables:
            name = str(var["canonical_name"])
            curve = baseline.curves.get(name)
            if curve is None:
                continue
            ext_rmse = np.asarray(curve["rmse"][:fixed_steps], dtype=np.float64)
            ext_acc = np.asarray(curve["acc"][:fixed_steps], dtype=np.float64)
            if not np.isfinite(ext_rmse[-1]) or not np.isfinite(ext_acc[-1]):
                continue
            for payload in stage_payloads:
                metric = payload["metrics"][name]
                rmse = np.asarray(metric["rmse_by_lead"][:fixed_steps], dtype=np.float64)
                acc = np.asarray(metric["acc_by_lead"][:fixed_steps], dtype=np.float64)
                diff = float(rmse[-1] - ext_rmse[-1])
                percent = float(100.0 * diff / ext_rmse[-1]) if ext_rmse[-1] != 0 else None
                acc_delta = float(acc[-1] - ext_acc[-1])
                rows.append(
                    {
                        "variable": name,
                        "model_name": payload["label"],
                        "params_m": None,
                        "avg_rmse_1_10": float(np.nanmean(rmse)),
                        "day10_rmse": float(rmse[-1]),
                        "avg_acc_1_10": float(np.nanmean(acc)),
                        "day10_acc": float(acc[-1]),
                        "rmse_day10_vs_external": diff,
                        "rmse_day10_percent_vs_external": percent,
                        "acc_day10_vs_external": acc_delta,
                        "acc_day10_delta_vs_external": acc_delta,
                    }
                )
            rows.append(
                {
                    "variable": name,
                    "model_name": baseline.label,
                    "params_m": baseline.params_m,
                    "avg_rmse_1_10": float(np.nanmean(ext_rmse)),
                    "day10_rmse": float(ext_rmse[-1]),
                    "avg_acc_1_10": float(np.nanmean(ext_acc)),
                    "day10_acc": float(ext_acc[-1]),
                    "rmse_day10_vs_external": 0.0,
                    "rmse_day10_percent_vs_external": 0.0,
                    "acc_day10_vs_external": 0.0,
                    "acc_day10_delta_vs_external": 0.0,
                }
            )
            last = stage_payloads[-1]
            last_metric = last["metrics"][name]
            rmse_winner = baseline.label if ext_rmse[-1] < last_metric["rmse_day10"] else last["label"]
            acc_winner = last["label"] if last_metric["acc_day10"] > ext_acc[-1] else baseline.label
            lines.append(
                f"- {name}: {rmse_winner} has lower day-{fixed_steps} RMSE, "
                f"{acc_winner} has higher day-{fixed_steps} ACC."
            )
        lines.append("")
    if not rows:
        return
    csv_path = output_dir / "external_baseline_comparison_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    txt_path = output_dir / "external_baseline_comparison_summary.txt"
    text = "\n".join(lines).rstrip() + "\n"
    txt_path.write_text(text, encoding="utf-8")
    logging.info("Saved external baseline comparison CSV: %s", csv_path)
    logging.info("Saved external baseline comparison text: %s", txt_path)
    print(text)


def _write_metrics_json(
    output_dir: Path,
    fixed_steps: int,
    variables: list[dict[str, Any]],
    stages: list[int],
    checkpoint_payloads: dict[str, dict[str, Any]],
    persistence_payload: dict[str, Any] | None,
    external_baselines: list[ExternalBaseline],
    args: argparse.Namespace,
    selection_metadata: dict[str, Any] | None = None,
    climatology_metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "fixed_rollout_steps": int(fixed_steps),
        "variables": [str(var["canonical_name"]) for var in variables],
        "variable_channels": {
            str(var["canonical_name"]): {"name": str(var["name"]), "channel": int(var["channel"]), "variable_idx": int(var["variable_idx"])}
            for var in variables
        },
        "stages": [int(stage) for stage in stages],
        "split": str(args.split),
        "selection": selection_metadata or {},
        "climatology": climatology_metadata or {},
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "confidence_level": float(args.confidence_level),
            "seed": int(args.bootstrap_seed),
        },
        "max_batches": args.max_batches,
        "checkpoints": checkpoint_payloads,
        "persistence": persistence_payload,
        "external_baselines": _external_baselines_payload(external_baselines, variables, fixed_steps),
        "external_baseline_fairness_note": EXTERNAL_BASELINE_FAIRNESS_NOTE if external_baselines else None,
        "notes": [
            "Every checkpoint is evaluated with the same fixed_rollout_steps horizon.",
            "trained S=N labels indicate the rollout curriculum stage used for training, not the evaluation horizon.",
            "If --include_persistence is used, persistence metrics are saved in JSON/CSV and plotted on lead curves.",
            "For fixed_rollout_steps other than 10, rmse_day10/acc_day10 are compatibility fields containing the final evaluated lead; use rmse_final_lead/acc_final_lead for explicit semantics.",
            "Aggregate RMSE across variables with different physical units should be interpreted carefully; aggregate ACC is more directly comparable.",
        ],
    }
    path = output_dir / "stage_rollout_progress_metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Saved metrics JSON: %s", path)


def _best_label(
    stage_payloads: list[dict[str, Any]],
    variable: str,
    metric_key: str,
    higher_is_better: bool,
) -> str:
    values = np.asarray([payload["metrics"][variable][metric_key] for payload in stage_payloads], dtype=np.float64)
    idx = int(np.nanargmax(values) if higher_is_better else np.nanargmin(values))
    return str(stage_payloads[idx]["label"])


def _write_interpretation_summary(
    output_dir: Path,
    variables: list[dict[str, Any]],
    checkpoint_payloads: dict[str, dict[str, Any]],
    fixed_steps: int,
) -> str:
    stage_payloads = _stage_payloads_sorted(checkpoint_payloads)
    lines: list[str] = ["Stage rollout progress summary:", ""]
    final_label = "day-10" if int(fixed_steps) == 10 else f"day-{int(fixed_steps)}"
    if not stage_payloads:
        lines.append("No trained stage checkpoints were evaluated.")
    else:
        first = stage_payloads[0]
        for var in variables:
            name = str(var["canonical_name"])
            lines.append(f"{name}:")
            lines.append(f"  best {final_label} RMSE: {_best_label(stage_payloads, name, 'rmse_day10', higher_is_better=False)}")
            lines.append(f"  best {final_label} ACC:  {_best_label(stage_payloads, name, 'acc_day10', higher_is_better=True)}")
            improvements = [
                first["metrics"][name]["rmse_day10"] - payload["metrics"][name]["rmse_day10"]
                for payload in stage_payloads
            ]
            best_improvement_idx = int(np.nanargmax(np.asarray(improvements, dtype=np.float64)))
            lines.append(f"  biggest {final_label} RMSE improvement over {first['label']}: {stage_payloads[best_improvement_idx]['label']}")
            day1_values = [payload["metrics"][name]["rmse_by_lead"][0] for payload in stage_payloads]
            if day1_values[-1] > day1_values[0]:
                lines.append(f"  day-1 RMSE increased from {first['label']} to {stage_payloads[-1]['label']}")
            else:
                lines.append(f"  day-1 RMSE did not worsen from {first['label']} to {stage_payloads[-1]['label']}")
            last = stage_payloads[-1]
            if last["metrics"][name]["rmse_day10"] < first["metrics"][name]["rmse_day10"]:
                lines.append(f"  {last['label']} improves {final_label} RMSE over {first['label']}.")
            else:
                lines.append(f"  {last['label']} does not improve {final_label} RMSE over {first['label']}.")
            s8_payload = next((payload for payload in stage_payloads if int(payload["stage"]) == 8), None)
            s10_payload = next((payload for payload in stage_payloads if int(payload["stage"]) == 10), None)
            if s8_payload and s10_payload:
                delta = s8_payload["metrics"][name]["rmse_day10"] - s10_payload["metrics"][name]["rmse_day10"]
                if abs(delta) <= 0.01 * max(abs(s8_payload["metrics"][name]["rmse_day10"]), 1.0e-12):
                    lines.append("  improvement appears to saturate after S=8 by day-10 RMSE.")
                elif delta > 0:
                    lines.append("  S=10 still improves over S=8 by day-10 RMSE.")
                else:
                    lines.append("  S=10 is worse than S=8 by day-10 RMSE.")
            lines.append("")

        all_s1_acc = np.nanmean([first["metrics"][str(var["canonical_name"])]["acc_day10"] for var in variables])
        last = stage_payloads[-1]
        all_last_acc = np.nanmean([last["metrics"][str(var["canonical_name"])]["acc_day10"] for var in variables])
        all_s1_rmse = np.nanmean([first["metrics"][str(var["canonical_name"])]["rmse_day10"] for var in variables])
        best_all_rmse_payload = min(
            stage_payloads,
            key=lambda payload: np.nanmean([payload["metrics"][str(var["canonical_name"])]["rmse_day10"] for var in variables]),
        )
        lines.append("Overall:")
        if all_last_acc > all_s1_acc:
            lines.append(f"  Long-rollout checkpoints improve mean {final_label} ACC compared with {first['label']}.")
        else:
            lines.append(f"  Long-rollout checkpoints do not improve mean {final_label} ACC compared with {first['label']}.")
        best_all_rmse = np.nanmean([best_all_rmse_payload["metrics"][str(var["canonical_name"])]["rmse_day10"] for var in variables])
        lines.append(f"  Best mean {final_label} RMSE checkpoint: {best_all_rmse_payload['label']} ({best_all_rmse:.6g} vs {first['label']} {all_s1_rmse:.6g}).")
        lines.append("  Check per-variable curves for saturation and short-lead tradeoffs.")

    text = "\n".join(lines)
    path = output_dir / "stage_rollout_progress_summary.txt"
    path.write_text(text + "\n", encoding="utf-8")
    logging.info("Saved interpretation summary: %s", path)
    print(text)
    return text


def _evaluate_checkpoints(
    evaluator: GraphWeatherEvaluator,
    checkpoint_items: list[dict[str, Any]],
    variables: list[dict[str, Any]],
    fixed_steps: int,
    fields: Any,
    total_t: int,
    height: int,
    width: int,
    climatology_norm: torch.Tensor,
    lat_weights: torch.Tensor,
    lat_weights_np: np.ndarray,
) -> dict[str, dict[str, Any]]:
    checkpoint_payloads: dict[str, dict[str, Any]] = {}
    for item in checkpoint_items:
        metadata = _checkpoint_metadata(item["checkpoint_path"])
        train_rollout = metadata.get("train_rollout_steps", item.get("stage"))
        logging.info(
            "Evaluating checkpoint: %s\nCheckpoint epoch: %s\nTraining rollout stage: S=%s\nFixed evaluation rollout: %d days",
            Path(item["checkpoint_path"]).name,
            metadata.get("epoch", "unknown"),
            train_rollout,
            fixed_steps,
        )
        print(
            f"Evaluating checkpoint: {Path(item['checkpoint_path']).name}\n"
            f"Checkpoint epoch: {metadata.get('epoch', 'unknown')}\n"
            f"Training rollout stage: S={train_rollout}\n"
            f"Fixed evaluation rollout: {fixed_steps} days"
        )
        evaluator.model = evaluator._load_model(height, width, item["checkpoint_path"])
        metrics = evaluator._evaluate_loaded_model_rollout(
            fields,
            total_t,
            height,
            width,
            fixed_steps,
            climatology_norm,
            lat_weights,
            lat_weights_np,
            item["label"],
        )
        checkpoint_payloads[item["key"]] = {
            "label": item["label"],
            "stage": item.get("stage"),
            "checkpoint_path": item["checkpoint_path"],
            "checkpoint_epoch": metadata.get("epoch"),
            "train_rollout_steps": train_rollout,
            "metrics": {
                str(var["canonical_name"]): _metric_payload(metrics, int(var["variable_idx"]), fixed_steps)
                for var in variables
            },
        }
    return checkpoint_payloads


def _evaluate_persistence(
    evaluator: GraphWeatherEvaluator,
    variables: list[dict[str, Any]],
    fixed_steps: int,
    fields: Any,
    total_t: int,
    height: int,
    width: int,
    climatology_norm: torch.Tensor,
    lat_weights: torch.Tensor,
    lat_weights_np: np.ndarray,
) -> dict[str, Any]:
    logging.info("Evaluating persistence baseline | fixed eval rollout=%d", fixed_steps)
    print(f"Evaluating persistence baseline | fixed eval rollout={fixed_steps}")
    metrics = evaluator._evaluate_persistence_rollout(
        fields,
        total_t,
        height,
        width,
        fixed_steps,
        climatology_norm,
        lat_weights,
        lat_weights_np,
    )
    return {
        str(var["canonical_name"]): _metric_payload(metrics, int(var["variable_idx"]), fixed_steps)
        for var in variables
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stage checkpoints with the same fixed rollout horizon.")
    parser.add_argument("--experiment_dir", required=True, type=str)
    parser.add_argument("--config", default=str(project_root / "configs" / "gnn_5p625.yaml"), type=str)
    parser.add_argument("--yaml_config", default=None, type=str)
    parser.add_argument("--config_name", default=None, type=str)
    parser.add_argument("--resolution_mode", default=None, type=str, help="5p625 or 2p5; overrides YAML resolution_mode.")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--fixed_rollout_steps", default=10, type=int)
    parser.add_argument("--selection", default="stride", choices=["first_n", "stride", "all"])
    parser.add_argument("--n_initial_conditions", default=None, type=int)
    parser.add_argument("--max_initial_conditions", default=None, type=int)
    parser.add_argument("--stride", default=7, type=int)
    parser.add_argument("--start_offset", default=0, type=int)
    parser.add_argument("--start_date", default=None, type=str)
    parser.add_argument("--end_date", default=None, type=str)
    parser.add_argument("--stages", nargs="*", default=None)
    parser.add_argument("--variables", nargs="*", default=None)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--include_global_best", action="store_true")
    parser.add_argument("--include_persistence", dest="include_persistence", action="store_true", default=False)
    parser.add_argument("--no_include_persistence", dest="include_persistence", action="store_false")
    parser.add_argument("--climatology_path", default=None, type=str)
    parser.add_argument("--build_climatology_if_missing", action="store_true")
    parser.add_argument("--bootstrap_samples", default=0, type=int)
    parser.add_argument("--confidence_level", default=0.95, type=float)
    parser.add_argument("--bootstrap_seed", default=42, type=int)
    parser.add_argument("--plot_confidence_intervals", action="store_true")
    parser.add_argument("--external_baseline_csv", nargs="*", default=None)
    parser.add_argument("--external_baseline_label", nargs="*", default=None)
    parser.add_argument("--external_baseline_params_m", nargs="*", default=None, type=float)
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--device", default=None, type=str)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stages = _parse_ints(args.stages, [1, 2, 4, 6, 8, 10])
    requested_variables = _parse_strings(args.variables, ["z500", "t2m", "t850", "msl"])
    fixed_steps = int(args.fixed_rollout_steps)
    if fixed_steps <= 0:
        raise ValueError("--fixed_rollout_steps must be positive.")

    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else experiment_dir / "stage_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(rank=0, log_file=str(output_dir / "stage_rollout_progress.log"))

    yaml_path, config_name = _resolve_config_args(args)
    params = YParams(yaml_path, config_name, resolution_mode=args.resolution_mode)
    checkpoint_items = _discover_checkpoints(experiment_dir, stages, bool(args.include_global_best))
    stage_items = [item for item in checkpoint_items if item.get("stage") is not None]
    if not stage_items:
        raise FileNotFoundError(f"No stage checkpoints found under {experiment_dir}")
    _warn_if_identical_paths(stage_items)

    cfg = _build_eval_config(params, args, output_dir, stage_items[0]["checkpoint_path"])
    params.log()
    evaluator = GraphWeatherEvaluator(cfg, logger=logging)
    ds, fields, total_t, height, width, climatology_norm, lat_weights, lat_weights_np = evaluator._prepare_context()
    try:
        variables = _resolve_variables(requested_variables, evaluator.variable_names, cfg.out_channels)
        logging.info("Resolved variables: %s", variables)
        checkpoint_payloads = _evaluate_checkpoints(
            evaluator,
            checkpoint_items,
            variables,
            fixed_steps,
            fields,
            total_t,
            height,
            width,
            climatology_norm,
            lat_weights,
            lat_weights_np,
        )
        persistence_payload = None
        if args.include_persistence:
            persistence_payload = _evaluate_persistence(
                evaluator,
                variables,
                fixed_steps,
                fields,
                total_t,
                height,
                width,
                climatology_norm,
                lat_weights,
                lat_weights_np,
            )
        _check_identical_metric_curves(checkpoint_payloads, variables)
        _plot_lead_curves(
            output_dir,
            variables,
            checkpoint_payloads,
            persistence_payload,
            evaluator.external_baselines,
            fixed_steps,
            plot_confidence_intervals=bool(args.plot_confidence_intervals),
        )
        _plot_summary_curves(output_dir, variables, checkpoint_payloads, fixed_steps)
        _write_metrics_json(
            output_dir,
            fixed_steps,
            variables,
            stages,
            checkpoint_payloads,
            persistence_payload,
            evaluator.external_baselines,
            args,
            selection_metadata=evaluator._selection_metadata,
            climatology_metadata=evaluator._climatology_metadata,
        )
        _write_summary_csv(output_dir, variables, checkpoint_payloads, persistence_payload)
        _write_external_comparison_summary(
            output_dir,
            variables,
            checkpoint_payloads,
            evaluator.external_baselines,
            fixed_steps,
        )
        _write_interpretation_summary(output_dir, variables, checkpoint_payloads, fixed_steps)
    finally:
        ds.close()


if __name__ == "__main__":
    main()
