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

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.climatology import date_from_time, decode_time_values, find_nc_files  # noqa: E402
from src.config import YParams, setup_logging  # noqa: E402
from src.evaluator import EvalConfig, GraphWeatherEvaluator, _canonical_for_name, _latitude_weights, _norm_name  # noqa: E402


ZONE_EDGES = np.arange(-90, 91, 15, dtype=np.float64)


def _format_latitude_label(value: float) -> str:
    value_int = int(abs(value))
    if value < 0:
        return f"{value_int}S"
    if value > 0:
        return f"{value_int}N"
    return "0"


def _format_zone_label(lat_min: float, lat_max: float) -> str:
    return f"{_format_latitude_label(lat_min)}-{_format_latitude_label(lat_max)}"


ZONE_ORDER = [_format_zone_label(float(ZONE_EDGES[i]), float(ZONE_EDGES[i + 1])) for i in range(len(ZONE_EDGES) - 1)]
ZONE_BOUNDS = {
    zone: {"lat_min": float(ZONE_EDGES[i]), "lat_max": float(ZONE_EDGES[i + 1])}
    for i, zone in enumerate(ZONE_ORDER)
}
DEFAULT_VARIABLES = ["z500", "t2m", "t850", "msl", "q700", "u850"]
UNITS = {
    "z500": "m^2 s^-2",
    "t2m": "K",
    "t850": "K",
    "msl": "Pa",
    "q700": "kg kg^-1",
    "u850": "m s^-1",
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
    yaml_path = yaml_path or str(project_root / "configs" / "gnn_5p625.yaml")
    config_name = config_name or "raw_5p625"
    return os.path.abspath(os.path.expanduser(yaml_path)), config_name


def _parse_string_list(values: list[str] | None, default: list[str]) -> list[str]:
    if not values:
        return list(default)
    parsed: list[str] = []
    for value in values:
        parsed.extend(chunk.strip() for chunk in str(value).replace(",", " ").split() if chunk.strip())
    return parsed or list(default)


def _split_data_path(params: Any, split: str) -> str:
    if split == "train":
        return str(params.train_data_path)
    if split == "valid":
        return str(params.valid_data_path)
    if split == "test":
        return str(params.get("test_dataset_path", params.valid_data_path))
    raise ValueError(f"Unsupported split: {split}")


def _build_eval_config(params: Any, args: argparse.Namespace) -> EvalConfig:
    params["test_dataset_path"] = _split_data_path(params, str(args.split))
    params["eval_checkpoint_path"] = str(args.checkpoint)
    params["eval_output_dir"] = str(args.output_dir)
    params["eval_fixed_rollout_steps"] = int(args.fixed_rollout_steps)
    params["eval_forecast_steps"] = [int(args.fixed_rollout_steps)]
    params["eval_selection"] = str(args.selection)
    params["eval_ic_stride"] = int(args.stride)
    params["eval_start_offset"] = int(args.start_offset)
    params["eval_plot_variables"] = _parse_string_list(args.variables, DEFAULT_VARIABLES)
    params["plot_persistence"] = bool(args.include_persistence)
    params["compute_climatology"] = False
    params["build_climatology_if_missing"] = False
    if args.climatology_path:
        params["climatology_path"] = str(args.climatology_path)
    elif Path("data/stats/2p5_train_dayofyear_climatology.nc").exists():
        params["climatology_path"] = "data/stats/2p5_train_dayofyear_climatology.nc"
    elif Path("data/stats/5p625_train_dayofyear_climatology.nc").exists():
        params["climatology_path"] = "data/stats/5p625_train_dayofyear_climatology.nc"
    else:
        params["compute_climatology"] = True
    if args.n_initial_conditions is not None:
        params["n_initial_conditions"] = int(args.n_initial_conditions)
    if args.max_initial_conditions is not None:
        params["max_initial_conditions"] = int(args.max_initial_conditions)
    if args.start_date:
        params["eval_start_date"] = str(args.start_date)
    if args.end_date:
        params["eval_end_date"] = str(args.end_date)
    if args.device:
        params["eval_device"] = str(args.device)
    if args.external_baseline_csv:
        params["external_baseline_csv"] = [str(args.external_baseline_csv)]
    return EvalConfig.from_params(params)


def _resolve_variables(evaluator: GraphWeatherEvaluator, requested: list[str]) -> list[int]:
    lower_names = {_norm_name(name): idx for idx, name in enumerate(evaluator.variable_names)}
    resolved: list[int] = []
    for token in requested:
        token = str(token).strip()
        if not token:
            continue
        idx: int | None = None
        if token.lower() == "all":
            return list(range(len(evaluator.cfg.out_channels)))
        if token.lstrip("-").isdigit():
            value = int(token)
            if 0 <= value < len(evaluator.cfg.out_channels):
                idx = value
            elif value in evaluator.cfg.out_channels:
                idx = evaluator.cfg.out_channels.index(value)
        else:
            candidates = [_norm_name(token)]
            canonical = _canonical_for_name(token)
            if canonical:
                aliases = {
                    "z500": ["z500", "z_500", "geopotential_500", "z@500"],
                    "t2m": ["t2m", "2m_temperature", "temperature_2m"],
                    "t850": ["t850", "t_850", "temperature_850", "t@850"],
                    "msl": ["msl", "mslp", "mean_sea_level_pressure"],
                    "q700": ["q700", "q_700", "specific_humidity_700"],
                    "u850": ["u850", "u_850", "u_component_wind_850"],
                }
                candidates.extend(_norm_name(alias) for alias in aliases.get(canonical, []))
            for candidate in candidates:
                if candidate in lower_names:
                    idx = lower_names[candidate]
                    break
        if idx is None:
            logging.warning("WARNING: variable %s could not be resolved and will be skipped.", token)
            print(f"WARNING: variable {token} could not be resolved and will be skipped.")
            continue
        if idx not in resolved:
            resolved.append(idx)
    return resolved


def _zone_masks(latitudes: np.ndarray) -> dict[str, np.ndarray]:
    latitudes = np.asarray(latitudes, dtype=np.float64).reshape(-1)
    masks: dict[str, np.ndarray] = {}
    for zone_idx, zone in enumerate(ZONE_ORDER):
        lat_min = float(ZONE_BOUNDS[zone]["lat_min"])
        lat_max = float(ZONE_BOUNDS[zone]["lat_max"])
        if zone_idx == len(ZONE_ORDER) - 1:
            mask = (latitudes >= lat_min) & (latitudes <= lat_max)
        else:
            mask = (latitudes >= lat_min) & (latitudes < lat_max)
        masks[zone] = mask
    return masks


def _zone_metadata(latitudes: np.ndarray, lat_weights: np.ndarray, width: int) -> dict[str, dict[str, Any]]:
    masks = _zone_masks(latitudes)
    total_weight = float(lat_weights.sum() * int(width))
    metadata: dict[str, dict[str, Any]] = {}
    for zone in ZONE_ORDER:
        mask = masks[zone]
        weight_sum = float(lat_weights[mask].sum() * int(width))
        metadata[zone] = {
            **ZONE_BOUNDS[zone],
            "num_lat_rows": int(mask.sum()),
            "area_weight_sum": weight_sum,
            "area_weight_percent": float(100.0 * weight_sum / total_weight) if total_weight > 0.0 else None,
        }
    return metadata


def _empty_accumulator(n_leads: int, n_vars: int) -> dict[str, Any]:
    return {
        "global_sse": np.zeros((n_leads, n_vars), dtype=np.float64),
        "global_weight": np.zeros((n_leads, n_vars), dtype=np.float64),
        "zones": {
            zone: {
                "sse": np.zeros((n_leads, n_vars), dtype=np.float64),
                "weight": np.zeros((n_leads, n_vars), dtype=np.float64),
            }
            for zone in ZONE_ORDER
        },
    }


def _add_error_to_accumulator(
    acc: dict[str, Any],
    err_phys: np.ndarray,
    lead_idx: int,
    variable_indices: list[int],
    zone_masks: dict[str, np.ndarray],
    lat_weights: np.ndarray,
    width: int,
) -> None:
    err_selected = np.asarray(err_phys[variable_indices], dtype=np.float64)
    err2 = np.square(err_selected)
    weight_2d = lat_weights.reshape(1, -1, 1)
    weighted_err2 = err2 * weight_2d
    global_weight = float(lat_weights.sum() * int(width))
    acc["global_sse"][lead_idx] += weighted_err2.sum(axis=(-2, -1))
    acc["global_weight"][lead_idx] += global_weight
    for zone, mask in zone_masks.items():
        zone_weight = float(lat_weights[mask].sum() * int(width))
        acc["zones"][zone]["sse"][lead_idx] += weighted_err2[:, mask, :].sum(axis=(-2, -1))
        acc["zones"][zone]["weight"][lead_idx] += zone_weight


@torch.no_grad()
def _evaluate_model_zones(
    evaluator: GraphWeatherEvaluator,
    fields: Any,
    ics: list[int],
    variable_indices: list[int],
    fixed_steps: int,
    zone_masks: dict[str, np.ndarray],
    lat_weights: np.ndarray,
    height: int,
    width: int,
) -> dict[str, Any]:
    if evaluator.model is None:
        raise RuntimeError("Model must be loaded before model zone evaluation.")
    acc = _empty_accumulator(fixed_steps, len(variable_indices))
    std_view = torch.as_tensor(evaluator.out_stds, device=evaluator.device, dtype=torch.float32).view(1, -1, 1, 1)
    n_channels = len(evaluator.cfg.out_channels)
    for count, ic in enumerate(ics):
        raw_prev = np.asarray(fields[ic - evaluator.cfg.dt, :, :height, :width], dtype=np.float32)
        raw_cur = np.asarray(fields[ic, :, :height, :width], dtype=np.float32)
        previous = evaluator._normalize_input_step(raw_prev)
        current = evaluator._normalize_input_step(raw_cur)
        initial_state = current
        for step_idx in range(fixed_steps):
            lead = step_idx + 1
            target_idx = ic + lead * evaluator.cfg.dt
            raw_target = np.asarray(fields[target_idx, :, :height, :width], dtype=np.float32)
            target_norm = evaluator._normalize_target_step(raw_target)
            aux = None
            if evaluator.feature_builder is not None:
                target_doy, target_days = evaluator._target_time_tensors(target_idx)
                aux = evaluator.feature_builder.build_step_features(
                    current=current,
                    target_norm=target_norm,
                    target_dayofyear=target_doy,
                    target_days_in_year=target_days,
                    step_idx=0,
                )
            pred_norm = (
                evaluator.model.forward_steps(previous, current)
                if aux is None
                else evaluator.model.forward_steps(previous, current, aux_features=aux)
            )
            if evaluator.feature_builder is not None:
                pred_norm = evaluator.feature_builder.apply_overrides(pred_norm, current=current, target_norm=target_norm)
            if getattr(evaluator, "target_handler", None) is not None:
                pred_norm = evaluator.target_handler.apply(
                    pred_next=pred_norm,
                    current_state=current,
                    initial_state=initial_state,
                    target_norm=target_norm,
                    lead=lead,
                )
            err_phys = ((pred_norm - target_norm) * std_view)[0].detach().cpu().double().numpy()
            _add_error_to_accumulator(acc, err_phys, step_idx, variable_indices, zone_masks, lat_weights, width)
            next_current = current.clone()
            next_current[:, :n_channels] = pred_norm
            previous, current = current, next_current
        if (count + 1) % 10 == 0 or (count + 1) == len(ics):
            logging.info("model completed IC %d/%d (ic=%d)", count + 1, len(ics), ic)
    return acc


@torch.no_grad()
def _evaluate_persistence_zones(
    evaluator: GraphWeatherEvaluator,
    fields: Any,
    ics: list[int],
    variable_indices: list[int],
    fixed_steps: int,
    zone_masks: dict[str, np.ndarray],
    lat_weights: np.ndarray,
    height: int,
    width: int,
) -> dict[str, Any]:
    acc = _empty_accumulator(fixed_steps, len(variable_indices))
    std_view = torch.as_tensor(evaluator.out_stds, device=evaluator.device, dtype=torch.float32).view(1, -1, 1, 1)
    for ic in ics:
        raw_cur = np.asarray(fields[ic, :, :height, :width], dtype=np.float32)
        pred_norm = evaluator._normalize_target_step(raw_cur)
        for step_idx in range(fixed_steps):
            target_idx = ic + (step_idx + 1) * evaluator.cfg.dt
            raw_target = np.asarray(fields[target_idx, :, :height, :width], dtype=np.float32)
            target_norm = evaluator._normalize_target_step(raw_target)
            err_phys = ((pred_norm - target_norm) * std_view)[0].detach().cpu().double().numpy()
            _add_error_to_accumulator(acc, err_phys, step_idx, variable_indices, zone_masks, lat_weights, width)
    return acc


def _finalize_metrics(acc: dict[str, Any], variable_names: list[str], zone_metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    global_sse = acc["global_sse"]
    global_weight = acc["global_weight"]
    global_mse = np.divide(global_sse, global_weight, out=np.full_like(global_sse, np.nan), where=global_weight > 0.0)
    global_rmse = np.sqrt(global_mse)
    payload: dict[str, Any] = {}
    for var_idx, variable in enumerate(variable_names):
        item: dict[str, Any] = {
            "global_rmse_by_lead": global_rmse[:, var_idx].tolist(),
            "global_mse_by_lead": global_mse[:, var_idx].tolist(),
            "global_sse_by_lead": global_sse[:, var_idx].tolist(),
            "zones": {},
        }
        dominance_by_zone: dict[str, np.ndarray] = {}
        for zone in ZONE_ORDER:
            zone_sse = acc["zones"][zone]["sse"][:, var_idx]
            zone_weight = acc["zones"][zone]["weight"][:, var_idx]
            zone_mse = np.divide(zone_sse, zone_weight, out=np.full_like(zone_sse, np.nan), where=zone_weight > 0.0)
            zone_rmse = np.sqrt(zone_mse)
            dominance = np.divide(
                100.0 * zone_sse,
                global_sse[:, var_idx],
                out=np.full_like(zone_sse, np.nan),
                where=global_sse[:, var_idx] > 0.0,
            )
            area_percent = float(zone_metadata[zone]["area_weight_percent"] or 0.0)
            ratio = np.divide(
                dominance,
                area_percent,
                out=np.full_like(dominance, np.nan),
                where=abs(area_percent) > 1.0e-12,
            )
            dominance_by_zone[zone] = dominance
            item["zones"][zone] = {
                "rmse_by_lead": zone_rmse.tolist(),
                "mse_by_lead": zone_mse.tolist(),
                "sse_by_lead": zone_sse.tolist(),
                "dominance_percent_by_lead": dominance.tolist(),
                "area_weight_percent": area_percent,
                "dominance_to_area_ratio_by_lead": ratio.tolist(),
            }
        dominance_matrix = np.stack([dominance_by_zone[zone] for zone in ZONE_ORDER], axis=1)
        item["dominant_zone_by_day"] = [ZONE_ORDER[int(idx)] for idx in np.nanargmax(dominance_matrix, axis=1)]
        item["dominant_zone_day10"] = item["dominant_zone_by_day"][-1]
        avg_dominance = np.asarray([np.nanmean(dominance_by_zone[zone]) for zone in ZONE_ORDER], dtype=np.float64)
        item["dominant_zone_avg_1_10"] = ZONE_ORDER[int(np.nanargmax(avg_dominance))]
        payload[variable] = item
    return payload


def _add_skill(model_payload: dict[str, Any], persistence_payload: dict[str, Any]) -> None:
    for variable, item in model_payload.items():
        p_item = persistence_payload.get(variable)
        if not p_item:
            continue
        model_global_mse = np.asarray(item["global_mse_by_lead"], dtype=np.float64)
        persistence_global_mse = np.asarray(p_item["global_mse_by_lead"], dtype=np.float64)
        global_skill = 1.0 - np.divide(
            model_global_mse,
            persistence_global_mse,
            out=np.full_like(model_global_mse, np.nan),
            where=persistence_global_mse > 0.0,
        )
        item["global_skill_by_lead"] = global_skill.tolist()
        for zone in ZONE_ORDER:
            model_mse = np.asarray(item["zones"][zone]["mse_by_lead"], dtype=np.float64)
            persistence_mse = np.asarray(p_item["zones"][zone]["mse_by_lead"], dtype=np.float64)
            zone_skill = 1.0 - np.divide(
                model_mse,
                persistence_mse,
                out=np.full_like(model_mse, np.nan),
                where=persistence_mse > 0.0,
            )
            item["zones"][zone]["skill_by_lead"] = zone_skill.tolist()


def _rows_by_lead(
    model_name: str,
    metrics: dict[str, Any],
    persistence: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variable, item in metrics.items():
        global_rmse = np.asarray(item["global_rmse_by_lead"], dtype=np.float64)
        global_mse = np.asarray(item["global_mse_by_lead"], dtype=np.float64)
        global_sse = np.asarray(item["global_sse_by_lead"], dtype=np.float64)
        global_skill = np.asarray(item.get("global_skill_by_lead", np.full_like(global_mse, np.nan)), dtype=np.float64)
        for lead_idx in range(global_rmse.size):
            for zone in ZONE_ORDER:
                z = item["zones"][zone]
                zone_skill = z.get("skill_by_lead")
                rows.append(
                    {
                        "model_name": model_name,
                        "variable": variable,
                        "lead_day": lead_idx + 1,
                        "zone": zone,
                        "zone_rmse": z["rmse_by_lead"][lead_idx],
                        "zone_mse": z["mse_by_lead"][lead_idx],
                        "zone_sse": z["sse_by_lead"][lead_idx],
                        "zone_dominance_percent": z["dominance_percent_by_lead"][lead_idx],
                        "zone_area_weight_percent": z["area_weight_percent"],
                        "dominance_to_area_ratio": z["dominance_to_area_ratio_by_lead"][lead_idx],
                        "global_rmse": global_rmse[lead_idx],
                        "global_mse": global_mse[lead_idx],
                        "global_sse": global_sse[lead_idx],
                        "zone_skill_vs_persistence": None if zone_skill is None else zone_skill[lead_idx],
                        "global_skill_vs_persistence": None if not np.isfinite(global_skill[lead_idx]) else global_skill[lead_idx],
                    }
                )
    return rows


def _summary_rows(model_name: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variable, item in metrics.items():
        for zone in ZONE_ORDER:
            z = item["zones"][zone]
            rmse = np.asarray(z["rmse_by_lead"], dtype=np.float64)
            dominance = np.asarray(z["dominance_percent_by_lead"], dtype=np.float64)
            ratio = np.asarray(z["dominance_to_area_ratio_by_lead"], dtype=np.float64)
            skill = np.asarray(z.get("skill_by_lead", np.full_like(rmse, np.nan)), dtype=np.float64)
            rows.append(
                {
                    "model_name": model_name,
                    "variable": variable,
                    "zone": zone,
                    "avg_rmse_1_10": float(np.nanmean(rmse)),
                    "day10_rmse": float(rmse[-1]),
                    "avg_dominance_percent_1_10": float(np.nanmean(dominance)),
                    "day10_dominance_percent": float(dominance[-1]),
                    "area_weight_percent": z["area_weight_percent"],
                    "avg_dominance_to_area_ratio_1_10": float(np.nanmean(ratio)),
                    "day10_dominance_to_area_ratio": float(ratio[-1]),
                    "avg_skill_1_10": None if np.all(~np.isfinite(skill)) else float(np.nanmean(skill)),
                    "day10_skill": None if not np.isfinite(skill[-1]) else float(skill[-1]),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Saved CSV: %s", path)


def _safe_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name)).strip("_") or "variable"


def _plot_outputs(output_dir: Path, model_metrics: dict[str, Any], persistence_metrics: dict[str, Any] | None, fixed_steps: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required for zone dominance plots.") from exc

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    leads = np.arange(1, int(fixed_steps) + 1)
    cmap = plt.get_cmap("tab20", len(ZONE_ORDER))
    colors = {zone: cmap(zone_idx) for zone_idx, zone in enumerate(ZONE_ORDER)}
    for variable, item in model_metrics.items():
        safe = _safe_stem(variable)
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        for zone in ZONE_ORDER:
            ax.plot(leads, item["zones"][zone]["rmse_by_lead"], marker="o", label=zone, color=colors[zone])
        ax.set_title(f"{variable} zone RMSE")
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel(f"RMSE ({UNITS.get(variable, 'native units')})")
        ax.set_xticks(leads)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"zone_rmse_{safe}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        for zone in ZONE_ORDER:
            ax.plot(leads, item["zones"][zone]["dominance_percent_by_lead"], marker="o", label=zone, color=colors[zone])
        ax.set_title(f"{variable} zone dominance")
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel("Share of global weighted squared error (%)")
        ax.set_xticks(leads)
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"zone_dominance_{safe}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        bottom = np.zeros((int(fixed_steps),), dtype=np.float64)
        for zone in ZONE_ORDER:
            values = np.asarray(item["zones"][zone]["dominance_percent_by_lead"], dtype=np.float64)
            ax.bar(leads, values, bottom=bottom, label=zone, color=colors[zone])
            bottom += values
        ax.set_title(f"{variable} stacked dominance")
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel("Share of global weighted squared error (%)")
        ax.set_xticks(leads)
        ax.set_ylim(0.0, 100.0)
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"zone_dominance_stacked_{safe}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10.5, 5.4))
        day10 = [item["zones"][zone]["dominance_percent_by_lead"][fixed_steps - 1] for zone in ZONE_ORDER]
        ax.bar(ZONE_ORDER, day10, color=[colors[zone] for zone in ZONE_ORDER])
        ax.set_title(f"{variable} day-{fixed_steps} zone dominance")
        ax.set_xlabel("Zone")
        ax.set_ylabel("Share of global weighted squared error (%)")
        ax.set_ylim(0.0, max(100.0, float(np.nanmax(day10)) * 1.15))
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        fig.tight_layout()
        fig.savefig(plot_dir / f"zone_day10_dominance_{safe}.png", dpi=160)
        plt.close(fig)

        if persistence_metrics and "skill_by_lead" in next(iter(item["zones"].values())):
            fig, ax = plt.subplots(figsize=(10.5, 5.8))
            for zone in ZONE_ORDER:
                ax.plot(leads, item["zones"][zone]["skill_by_lead"], marker="o", label=zone, color=colors[zone])
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(f"{variable} zone skill vs persistence")
            ax.set_xlabel("Lead time (days)")
            ax.set_ylabel("Skill = 1 - model MSE / persistence MSE")
            ax.set_xticks(leads)
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=3, fontsize=8)
            fig.tight_layout()
            fig.savefig(plot_dir / f"zone_skill_{safe}.png", dpi=160)
            plt.close(fig)


def _write_text_report(
    output_dir: Path,
    model_metrics: dict[str, Any],
    persistence_metrics: dict[str, Any] | None,
    fixed_steps: int,
) -> None:
    lines = ["Latitude-zone RMSE dominance summary:", ""]
    for variable, item in model_metrics.items():
        lines.append(f"{variable}:")
        lines.append(f"- dominant zone day {fixed_steps}: {item['dominant_zone_day10']}")
        lines.append(f"- dominant zone avg days 1-{fixed_steps}: {item['dominant_zone_avg_1_10']}")
        for zone in ZONE_ORDER:
            z = item["zones"][zone]
            day_dom = float(z["dominance_percent_by_lead"][fixed_steps - 1])
            avg_dom = float(np.nanmean(z["dominance_percent_by_lead"]))
            area = float(z["area_weight_percent"])
            ratio = day_dom / area if abs(area) > 1.0e-12 else float("nan")
            message = (
                f"  {zone}: day-{fixed_steps} dominance {day_dom:.2f}% "
                f"(avg {avg_dom:.2f}%, area {area:.2f}%, ratio {ratio:.2f})"
            )
            if ratio > 2.0:
                message += " - contributes more than twice its area share"
            lines.append(message)
            if "skill_by_lead" in z:
                skill = np.asarray(z["skill_by_lead"], dtype=np.float64)
                day_skill = float(skill[fixed_steps - 1])
                avg_skill = float(np.nanmean(skill))
                direction = "improves" if day_skill > 0.0 else "does not improve"
                lines.append(f"    model {direction} this zone vs persistence at day {fixed_steps} (skill {day_skill:.3f}, avg {avg_skill:.3f})")
        lines.append(
            f"For {variable} day {fixed_steps}, the dominant RMSE contribution comes from "
            f"{item['dominant_zone_day10']} with "
            f"{item['zones'][item['dominant_zone_day10']]['dominance_percent_by_lead'][fixed_steps - 1]:.2f}% "
            "of weighted squared error."
        )
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    path = output_dir / "zone_rmse_dominance_summary.txt"
    path.write_text(text, encoding="utf-8")
    print(text)
    logging.info("Saved text report: %s", path)


def _warn_external_baseline(path: str | None) -> None:
    if not path:
        return
    try:
        import pandas as pd
    except ModuleNotFoundError:
        print("External baseline CSV was provided, but pandas is unavailable; skipping external baseline inspection.")
        return
    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"external_baseline_csv does not exist: {csv_path}")
    df = pd.read_csv(csv_path, nrows=5)
    zone_columns = {"zone", "zone_rmse", "zone_sse", "zone_dominance_percent"}
    if not zone_columns.intersection(df.columns):
        message = "External baseline CSV has global metrics only. Zone dominance cannot be computed for this baseline."
        logging.warning(message)
        print(message)


def _json_safe(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(k): _json_safe(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_json_safe(v) for v in payload]
    if isinstance(payload, tuple):
        return [_json_safe(v) for v in payload]
    if isinstance(payload, np.ndarray):
        return _json_safe(payload.tolist())
    if isinstance(payload, np.generic):
        return payload.item()
    return payload


def _run(args: argparse.Namespace) -> None:
    _warn_external_baseline(args.external_baseline_csv)
    yaml_path, config_name = _resolve_config_args(args)
    params = YParams(yaml_path, config_name, resolution_mode=args.resolution_mode)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(rank=0, log_file=str(output_dir / "zone_rmse_dominance.log"))
    params.log()

    cfg = _build_eval_config(params, args)
    evaluator = GraphWeatherEvaluator(cfg, logger=logging)

    files = find_nc_files(cfg.test_dataset_path)
    test_file = files[0]
    logging.info("Running zone dominance evaluation on: %s", test_file)
    ds = evaluator.nc.Dataset(test_file, "r")
    try:
        fields = ds["fields"]
        total_t, _, height, width = fields.shape
        if (int(height), int(width)) != tuple(cfg.grid_shape):
            raise ValueError(f"Expected grid {tuple(cfg.grid_shape)}, got {(int(height), int(width))}")
        evaluator._load_variable_names(ds)
        evaluator._time_values = decode_time_values(ds)
        if len(evaluator._time_values) != int(total_t):
            raise ValueError(f"Decoded time length {len(evaluator._time_values)} does not match fields length {total_t}.")
        if "latitude" in ds.variables:
            latitudes = np.asarray(ds["latitude"][:height], dtype=np.float64)
        elif "lat" in ds.variables:
            latitudes = np.asarray(ds["lat"][:height], dtype=np.float64)
        else:
            raise KeyError("Dataset has no latitude/lat coordinate; zone dominance requires actual dataset latitudes.")

        fixed_steps = int(cfg.eval_fixed_rollout_steps)
        ics = evaluator._build_ic_indices(int(total_t), fixed_steps)
        evaluator._selected_ics = list(ics)
        evaluator._selection_metadata.update(
            {
                "file": str(test_file),
                "calendar_aware": True,
                "file_years": sorted({int(date_from_time(t).year) for t in evaluator._time_values}),
            }
        )

        requested = _parse_string_list(args.variables, DEFAULT_VARIABLES)
        variable_indices = _resolve_variables(evaluator, requested)
        if not variable_indices:
            raise ValueError("No requested variables could be resolved.")
        variable_keys = [evaluator._variable_key(idx) for idx in variable_indices]
        variable_names = [key if key else evaluator.variable_names[idx] for key, idx in zip(variable_keys, variable_indices)]

        lat_weights = _latitude_weights(latitudes)
        zone_masks = _zone_masks(latitudes)
        zone_meta = _zone_metadata(latitudes, lat_weights, int(width))
        for zone, meta in zone_meta.items():
            if int(meta["num_lat_rows"]) <= 0:
                logging.warning("WARNING: latitude zone %s has no rows for this grid.", zone)

        checkpoint_path = evaluator._resolve_checkpoint_path(cfg.checkpoint_path)
        checkpoint_metadata = evaluator._checkpoint_metadata(checkpoint_path)
        evaluator.model = evaluator._load_model(int(height), int(width), checkpoint_path)
        model_acc = _evaluate_model_zones(
            evaluator,
            fields,
            ics,
            variable_indices,
            fixed_steps,
            zone_masks,
            lat_weights,
            int(height),
            int(width),
        )
        model_metrics = _finalize_metrics(model_acc, variable_names, zone_meta)

        persistence_metrics = None
        if args.include_persistence:
            persistence_acc = _evaluate_persistence_zones(
                evaluator,
                fields,
                ics,
                variable_indices,
                fixed_steps,
                zone_masks,
                lat_weights,
                int(height),
                int(width),
            )
            persistence_metrics = _finalize_metrics(persistence_acc, variable_names, zone_meta)
            _add_skill(model_metrics, persistence_metrics)

        detailed_rows = _rows_by_lead("model", model_metrics)
        if persistence_metrics is not None:
            detailed_rows.extend(_rows_by_lead("persistence", persistence_metrics))
        summary_rows = _summary_rows("model", model_metrics)
        if persistence_metrics is not None:
            summary_rows.extend(_summary_rows("persistence", persistence_metrics))

        _write_csv(output_dir / "zone_rmse_dominance_by_lead.csv", detailed_rows)
        _write_csv(output_dir / "zone_rmse_dominance_summary.csv", summary_rows)
        _write_text_report(output_dir, model_metrics, persistence_metrics, fixed_steps)
        _plot_outputs(output_dir, model_metrics, persistence_metrics, fixed_steps)

        payload = {
            "checkpoint": checkpoint_path,
            "checkpoint_metadata": checkpoint_metadata,
            "selection": evaluator._selection_metadata,
            "rollout_steps": fixed_steps,
            "lead_days": list(range(1, fixed_steps + 1)),
            "variables": variable_names,
            "zones": zone_meta,
            "model": model_metrics,
            "persistence": persistence_metrics or {},
            "notes": [
                "Dominance percent is computed from latitude-weighted squared-error sums, not RMSE.",
                "Zone RMSE is computed from each zone's weighted MSE.",
                "Lead day arrays use index 0 for day 1 and contain no lead 0.",
            ],
        }
        json_path = output_dir / "zone_rmse_dominance_metrics.json"
        json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
        logging.info("Saved JSON: %s", json_path)
    finally:
        ds.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate latitude-zone RMSE dominance for a GraphWeather checkpoint.")
    parser.add_argument("--yaml_config", default=str(project_root / "configs" / "gnn_5p625.yaml"), type=str)
    parser.add_argument("--config", default="raw_5p625", type=str)
    parser.add_argument("--config_name", default=None, type=str)
    parser.add_argument("--resolution_mode", default=None, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--fixed_rollout_steps", default=10, type=int)
    parser.add_argument("--selection", default="stride", choices=["first_n", "stride", "all"])
    parser.add_argument("--n_initial_conditions", default=None, type=int)
    parser.add_argument("--max_initial_conditions", default=None, type=int)
    parser.add_argument("--stride", default=7, type=int)
    parser.add_argument("--start_offset", default=0, type=int)
    parser.add_argument("--start_date", default=None, type=str)
    parser.add_argument("--end_date", default=None, type=str)
    parser.add_argument("--variables", nargs="*", default=None)
    parser.add_argument("--include_persistence", action="store_true")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--climatology_path", default=None, type=str, help="Accepted for compatibility; not used by zone RMSE dominance.")
    parser.add_argument("--external_baseline_csv", default=None, type=str)
    args = parser.parse_args()
    if int(args.fixed_rollout_steps) <= 0:
        raise ValueError("--fixed_rollout_steps must be positive.")
    if int(args.fixed_rollout_steps) != 10:
        logging.warning("This analysis supports arbitrary fixed steps, but requested acceptance expects lead days 1-10.")
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive.")
    return args


def main() -> None:
    _run(_parse_args())


if __name__ == "__main__":
    main()
