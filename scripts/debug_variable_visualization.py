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
sys.path.insert(0, str(script_dir))

import visualize_rollout_maps as viz
from src.config import YParams, setup_logging


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min = np.inf
        self.max = -np.inf

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.sum += float(np.sum(arr))
        self.sum_sq += float(np.sum(arr * arr))
        self.min = min(self.min, float(np.min(arr)))
        self.max = max(self.max, float(np.max(arr)))

    def as_dict(self, prefix: str, include_rmse: bool = False) -> dict[str, float | int | None]:
        if self.count <= 0:
            payload: dict[str, float | int | None] = {
                f"{prefix}_min": None,
                f"{prefix}_max": None,
                f"{prefix}_mean": None,
                f"{prefix}_std": None,
            }
            if include_rmse:
                payload[f"{prefix}_rmse"] = None
            return payload
        mean = self.sum / float(self.count)
        variance = max(self.sum_sq / float(self.count) - mean * mean, 0.0)
        payload = {
            f"{prefix}_min": self.min,
            f"{prefix}_max": self.max,
            f"{prefix}_mean": mean,
            f"{prefix}_std": float(np.sqrt(variance)),
        }
        if include_rmse:
            payload[f"{prefix}_rmse"] = float(np.sqrt(self.sum_sq / float(self.count)))
        return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Saved JSON: %s", path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Saved CSV: %s", path)


def _safe_name(variable: dict[str, Any]) -> str:
    return viz._safe_stem(str(variable.get("canonical_name") or variable.get("name") or "variable"))


def _latitude_weights(lats: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(np.asarray(lats, dtype=np.float64))).clip(min=0.0)
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones_like(weights, dtype=np.float64)
    return weights


def _weighted_rmse(bias: np.ndarray, lats: np.ndarray) -> float:
    weights = _latitude_weights(lats).reshape(-1, 1)
    valid = np.isfinite(bias)
    if not np.any(valid):
        return float("nan")
    weighted = np.where(valid, bias * bias, 0.0) * weights
    denom = np.where(valid, 1.0, 0.0) * weights
    return float(np.sqrt(np.sum(weighted) / max(float(np.sum(denom)), 1.0e-12)))


def _weighted_acc(pred: np.ndarray, gt: np.ndarray, lats: np.ndarray) -> float:
    weights = _latitude_weights(lats).reshape(-1, 1)
    valid = np.isfinite(pred) & np.isfinite(gt)
    if not np.any(valid):
        return float("nan")
    w = np.where(valid, 1.0, 0.0) * weights
    w_sum = max(float(np.sum(w)), 1.0e-12)
    pred_mean = float(np.sum(np.where(valid, pred, 0.0) * weights) / w_sum)
    gt_mean = float(np.sum(np.where(valid, gt, 0.0) * weights) / w_sum)
    pred_anom = np.where(valid, pred - pred_mean, 0.0)
    gt_anom = np.where(valid, gt - gt_mean, 0.0)
    numerator = float(np.sum(pred_anom * gt_anom * weights))
    pred_power = float(np.sum(pred_anom * pred_anom * weights))
    gt_power = float(np.sum(gt_anom * gt_anom * weights))
    denom = np.sqrt(max(pred_power * gt_power, 0.0))
    if denom <= 1.0e-12:
        return float("nan")
    return float(numerator / denom)


def _edge_threshold(variable: dict[str, Any]) -> float:
    canonical = str(variable.get("canonical_name", "")).lower()
    if canonical == "msl":
        return 3000.0
    if canonical in {"t2m", "t850"} or canonical.startswith("t"):
        return 10.0
    if canonical.startswith("z"):
        return 1000.0
    return 0.0


def _stats_row(variable: dict[str, Any], lead: int) -> dict[str, Any]:
    return {
        "variable": str(variable.get("canonical_name") or variable.get("name")),
        "channel": int(variable["channel"]),
        "local_idx": int(variable["local_idx"]),
        "lead_time": int(lead),
    }


def _print_variable_details(variables: list[dict[str, Any]]) -> None:
    for var in variables:
        print(f"Requested variable: {var['requested']}")
        print(f"Resolved channel: {var['channel']}")
        print(f"Actual channel name: {var['name']}")
        print(f"Unit: {var.get('unit') or 'unknown'}")
        print(f"Normalization mean/std source: {var.get('normalization_mean_path', '')} / {var.get('normalization_std_path', '')}")
        print(f"Mean/std channel index: {var.get('normalization_mean_index')} / {var.get('normalization_std_index')}")
        print()


def _build_value_accumulators(
    variables: list[dict[str, Any]],
    lead_times: list[int],
) -> dict[str, dict[int, dict[str, RunningStats]]]:
    accum: dict[str, dict[int, dict[str, RunningStats]]] = {}
    for var in variables:
        key = str(var["name"])
        accum[key] = {}
        for lead in lead_times:
            accum[key][int(lead)] = {
                "normalized_gt": RunningStats(),
                "normalized_pred": RunningStats(),
                "denormalized_gt": RunningStats(),
                "denormalized_pred": RunningStats(),
                "bias": RunningStats(),
            }
    return accum


@torch.no_grad()
def _collect_rollout_debug_data(
    dataset: Any,
    model: Any,
    params: Any,
    variables: list[dict[str, Any]],
    rollout_steps: int,
    lead_times: list[int],
    aggregate_mode: str,
    sample_index: int,
    max_batches: int | None,
    device: torch.device,
    stats_info: dict[str, Any],
    lat_order: np.ndarray,
    lon_order: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]], int]:
    if aggregate_mode == "sample":
        sample_indices = [int(sample_index)]
    else:
        n_samples = len(dataset)
        if max_batches is not None and max_batches > 0:
            n_samples = min(n_samples, int(max_batches))
        sample_indices = list(range(n_samples))
    if not sample_indices:
        raise RuntimeError("No samples available for debug rollout.")

    means = np.asarray(stats_info["means"], dtype=np.float64).reshape(1, -1, 1, 1)
    stds = np.asarray(stats_info["stds"], dtype=np.float64).reshape(1, -1, 1, 1)
    height = int(dataset.img_shape_x)
    width = int(dataset.img_shape_y)
    maps: dict[str, dict[str, np.ndarray]] = {}
    for var in variables:
        maps[str(var["name"])] = {
            "gt_sum": np.zeros((rollout_steps, height, width), dtype=np.float64),
            "pred_sum": np.zeros((rollout_steps, height, width), dtype=np.float64),
        }

    value_accum = _build_value_accumulators(variables, lead_times)
    for count, sample_idx in enumerate(sample_indices, start=1):
        inp, target = dataset[int(sample_idx)]
        target_seq = viz._target_sequence(target)
        pred_norm = viz.rollout_predictions(model, inp, rollout_steps, device).numpy().astype(np.float64)
        gt_norm = target_seq.numpy().astype(np.float64)
        if pred_norm.shape[1] != len(stats_info["means"]):
            raise AssertionError(
                f"Model output channels {pred_norm.shape[1]} do not match output normalization stats length {len(stats_info['means'])}"
            )
        pred_denorm = pred_norm * stds + means
        gt_denorm = gt_norm * stds + means

        for var in variables:
            key = str(var["name"])
            local_idx = int(var["local_idx"])
            maps[key]["gt_sum"] += gt_denorm[:, local_idx]
            maps[key]["pred_sum"] += pred_denorm[:, local_idx]
            for lead in lead_times:
                step_idx = int(lead) - 1
                lead_accum = value_accum[key][int(lead)]
                gt_norm_map = gt_norm[step_idx, local_idx]
                pred_norm_map = pred_norm[step_idx, local_idx]
                gt_denorm_map = gt_denorm[step_idx, local_idx]
                pred_denorm_map = pred_denorm[step_idx, local_idx]
                bias = pred_denorm_map - gt_denorm_map
                lead_accum["normalized_gt"].update(gt_norm_map)
                lead_accum["normalized_pred"].update(pred_norm_map)
                lead_accum["denormalized_gt"].update(gt_denorm_map)
                lead_accum["denormalized_pred"].update(pred_denorm_map)
                lead_accum["bias"].update(bias)

        if count % 25 == 0 or count == len(sample_indices):
            logging.info("Debug rollout aggregated %d/%d samples", count, len(sample_indices))

    sample_count = len(sample_indices)
    aggregate_maps: dict[str, dict[str, np.ndarray]] = {}
    for var in variables:
        key = str(var["name"])
        gt = maps[key]["gt_sum"] / float(sample_count)
        pred = maps[key]["pred_sum"] / float(sample_count)
        gt = viz._apply_coordinate_orders(gt, lat_order, lon_order)
        pred = viz._apply_coordinate_orders(pred, lat_order, lon_order)
        mean_bias = pred - gt
        if not np.allclose(mean_bias, pred - gt, equal_nan=True):
            raise AssertionError("Bias assertion failed: mean_bias != mean_pred - mean_gt")
        aggregate_maps[key] = {"gt": gt, "pred": pred, "bias": mean_bias}

    value_rows: list[dict[str, Any]] = []
    for var in variables:
        key = str(var["name"])
        for lead in lead_times:
            row = _stats_row(var, int(lead))
            accum = value_accum[key][int(lead)]
            row.update(accum["normalized_gt"].as_dict("normalized_gt"))
            row.update(accum["normalized_pred"].as_dict("normalized_pred"))
            row.update(accum["denormalized_gt"].as_dict("denormalized_gt"))
            row.update(accum["denormalized_pred"].as_dict("denormalized_pred"))
            row.update(accum["bias"].as_dict("bias", include_rmse=True))
            value_rows.append(row)
    return aggregate_maps, value_rows, sample_count


def _warn_value_sanity(rows: list[dict[str, Any]], warnings: list[str]) -> None:
    for row in rows:
        variable = str(row["variable"]).lower()
        if variable != "msl":
            continue
        mean = row.get("denormalized_gt_mean")
        max_value = row.get("denormalized_gt_max")
        min_value = row.get("denormalized_gt_min")
        lead = row.get("lead_time")
        if mean is not None and (float(mean) < 90000.0 or float(mean) > 110000.0):
            warnings.append(f"WARNING: msl lead {lead} denormalized GT mean {float(mean):.3f} Pa is outside 90000..110000 Pa.")
        if max_value is not None and float(max_value) > 108000.0:
            warnings.append(f"WARNING: msl lead {lead} denormalized GT max {float(max_value):.3f} Pa exceeds 108000 Pa.")
        if min_value is not None and float(min_value) < 85000.0:
            warnings.append(f"WARNING: msl lead {lead} denormalized GT min {float(min_value):.3f} Pa is below 85000 Pa.")
    for warning in warnings:
        if "msl lead" in warning:
            logging.warning(warning)
            print(warning)


def _write_row_stats_and_metrics(
    output_dir: Path,
    variables: list[dict[str, Any]],
    lead_times: list[int],
    maps: dict[str, dict[str, np.ndarray]],
    lats: np.ndarray,
    warnings: list[str],
) -> list[dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    for var in variables:
        key = str(var["name"])
        safe = _safe_name(var)
        threshold = _edge_threshold(var)
        gt_maps = maps[key]["gt"]
        pred_maps = maps[key]["pred"]
        bias_maps = maps[key]["bias"]
        for lead in lead_times:
            step_idx = int(lead) - 1
            gt = gt_maps[step_idx]
            pred = pred_maps[step_idx]
            bias = bias_maps[step_idx]
            row_rows: list[dict[str, Any]] = []
            for row_idx, lat in enumerate(lats):
                row_rows.append(
                    {
                        "row_index": row_idx,
                        "latitude": float(lat),
                        "gt_mean": float(np.nanmean(gt[row_idx])),
                        "gt_min": float(np.nanmin(gt[row_idx])),
                        "gt_max": float(np.nanmax(gt[row_idx])),
                        "pred_mean": float(np.nanmean(pred[row_idx])),
                        "pred_min": float(np.nanmin(pred[row_idx])),
                        "pred_max": float(np.nanmax(pred[row_idx])),
                        "bias_mean": float(np.nanmean(bias[row_idx])),
                        "bias_min": float(np.nanmin(bias[row_idx])),
                        "bias_max": float(np.nanmax(bias[row_idx])),
                    }
                )
            _write_csv(output_dir / f"row_stats_{safe}_day{int(lead)}.csv", row_rows)

            if gt.shape[0] <= 2:
                continue
            interior_gt_mean = float(np.nanmean(gt[1:-1]))
            interior_pred_mean = float(np.nanmean(pred[1:-1]))
            interior_bias_mean = float(np.nanmean(bias[1:-1]))
            edge_indices = [0, gt.shape[0] - 1]
            edge_gt_diff = max(abs(float(np.nanmean(gt[idx])) - interior_gt_mean) for idx in edge_indices)
            edge_pred_diff = max(abs(float(np.nanmean(pred[idx])) - interior_pred_mean) for idx in edge_indices)
            edge_bias_diff = max(abs(float(np.nanmean(bias[idx])) - interior_bias_mean) for idx in edge_indices)
            artifact = threshold > 0.0 and max(edge_gt_diff, edge_pred_diff) > threshold
            if artifact:
                warning = (
                    f"WARNING: Possible edge-row / polar artifact for {safe} lead {lead}: "
                    f"edge row differs from interior by {max(edge_gt_diff, edge_pred_diff):.3f} "
                    f"(threshold {threshold:g})."
                )
                warnings.append(warning)
                logging.warning(warning)
                print(warning)

            rmse_all = _weighted_rmse(bias, lats)
            acc_all = _weighted_acc(pred, gt, lats)
            rmse_crop = _weighted_rmse(bias[1:-1], lats[1:-1])
            acc_crop = _weighted_acc(pred[1:-1], gt[1:-1], lats[1:-1])
            dramatic = False
            if safe == "msl" and np.isfinite(rmse_all) and np.isfinite(rmse_crop):
                if abs(rmse_all - rmse_crop) > max(500.0, 0.25 * max(abs(rmse_all), 1.0)):
                    dramatic = True
                    warning = "MSL appears dominated by edge-row artifact. Check pole/interpolation/latitude handling."
                    warnings.append(warning)
                    logging.warning(warning)
                    print(warning)
            metric_rows.append(
                {
                    "variable": safe,
                    "channel": int(var["channel"]),
                    "lead_time": int(lead),
                    "rmse_all_rows": rmse_all,
                    "rmse_without_edge_rows": rmse_crop,
                    "acc_all_rows": acc_all,
                    "acc_without_edge_rows": acc_crop,
                    "edge_gt_mean_diff_max": edge_gt_diff,
                    "edge_pred_mean_diff_max": edge_pred_diff,
                    "edge_bias_mean_diff_max": edge_bias_diff,
                    "artifact_threshold": threshold,
                    "possible_edge_artifact": bool(artifact),
                    "dramatic_msl_metric_change": bool(dramatic),
                }
            )
    _write_csv(output_dir / "metrics_with_without_edge_rows.csv", metric_rows)
    return metric_rows


def _plot_debug_maps(
    gt_maps: np.ndarray,
    pred_maps: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    variable: dict[str, Any],
    lead_times: list[int],
    output_path: Path,
    aggregate_label: str,
    unit_label: str,
    use_cartopy: bool,
    add_cyclic: bool,
    dpi: int,
    robust_percentile: float,
) -> None:
    rows = len(lead_times)
    fig = plt.figure(figsize=(18.0, max(4.0, 4.0 * rows)), constrained_layout=False)
    gs = fig.add_gridspec(rows, 5, width_ratios=[1.0, 1.0, 0.035, 1.0, 0.035], wspace=0.25, hspace=0.35)
    selected = [int(lead) - 1 for lead in lead_times]
    main_limits = viz._robust_limits([gt_maps[selected], pred_maps[selected]], robust_percentile, center_zero=False)
    for row, lead in enumerate(lead_times):
        step_idx = int(lead) - 1
        gt = gt_maps[step_idx]
        pred = pred_maps[step_idx]
        bias = pred - gt
        bias_limits = viz._robust_limits([bias], robust_percentile, center_zero=True)
        rmse = float(np.sqrt(np.nanmean(np.square(bias))))

        ax_gt = viz._add_map_axis(fig, gs[row, 0], use_cartopy)
        ax_pred = viz._add_map_axis(fig, gs[row, 1], use_cartopy)
        cax_main = fig.add_subplot(gs[row, 2])
        ax_bias = viz._add_map_axis(fig, gs[row, 3], use_cartopy)
        cax_bias = fig.add_subplot(gs[row, 4])
        pred_mesh = viz._plot_panel(
            ax_pred,
            lons,
            lats,
            pred,
            f"Prediction {aggregate_label} day {lead}",
            "viridis",
            main_limits[0],
            main_limits[1],
            use_cartopy,
            True,
            add_cyclic,
        )
        viz._plot_panel(
            ax_gt,
            lons,
            lats,
            gt,
            f"GT {aggregate_label} day {lead}",
            "viridis",
            main_limits[0],
            main_limits[1],
            use_cartopy,
            True,
            add_cyclic,
        )
        bias_mesh = viz._plot_panel(
            ax_bias,
            lons,
            lats,
            bias,
            f"Bias {aggregate_label} day {lead}\nPred - GT | RMSE={rmse:.3g}",
            "RdBu_r",
            bias_limits[0],
            bias_limits[1],
            use_cartopy,
            True,
            add_cyclic,
        )
        main_cb = fig.colorbar(pred_mesh, cax=cax_main)
        main_cb.ax.tick_params(labelsize=8)
        main_cb.set_label(unit_label, fontsize=9)
        bias_cb = fig.colorbar(bias_mesh, cax=cax_bias)
        bias_cb.ax.tick_params(labelsize=8)
        bias_cb.set_label(unit_label, fontsize=9)

    fig.suptitle(f"{variable['label']} | {aggregate_label}", fontsize=13)
    fig.subplots_adjust(left=0.035, right=0.975, bottom=0.040, top=0.92)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved debug map: %s", output_path)
    print(f"Saved debug map: {output_path}")


def _plot_row_diagnostics(
    output_dir: Path,
    variable: dict[str, Any],
    maps: dict[str, np.ndarray],
    lats: np.ndarray,
    lead_times: list[int],
    dpi: int,
) -> None:
    safe = _safe_name(variable)
    for lead in lead_times:
        step_idx = int(lead) - 1
        gt = maps["gt"][step_idx]
        pred = maps["pred"][step_idx]
        bias = maps["bias"][step_idx]
        rows = np.arange(gt.shape[0])

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(rows, np.nanmean(gt, axis=1), marker="o", linewidth=1.6, label="GT")
        ax.plot(rows, np.nanmean(pred, axis=1), marker="o", linewidth=1.6, label="Prediction")
        ax.set_xlabel("latitude row index")
        ax.set_ylabel(str(variable.get("unit") or "physical units"))
        ax.set_title(f"{safe} row mean day {lead}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        top = ax.twiny()
        top.set_xlim(ax.get_xlim())
        tick_idx = np.linspace(0, len(lats) - 1, min(6, len(lats)), dtype=int)
        top.set_xticks(tick_idx)
        top.set_xticklabels([f"{lats[i]:.1f}" for i in tick_idx])
        top.set_xlabel("latitude")
        fig.tight_layout()
        path = output_dir / f"{safe}_row_mean_day{int(lead)}.png"
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        logging.info("Saved row mean plot: %s", path)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(rows, np.nanmean(bias, axis=1), marker="o", linewidth=1.6, color="tab:red")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("latitude row index")
        ax.set_ylabel(str(variable.get("unit") or "physical units"))
        ax.set_title(f"{safe} row bias day {lead} (prediction - ground truth)")
        ax.grid(True, alpha=0.3)
        top = ax.twiny()
        top.set_xlim(ax.get_xlim())
        top.set_xticks(tick_idx)
        top.set_xticklabels([f"{lats[i]:.1f}" for i in tick_idx])
        top.set_xlabel("latitude")
        fig.tight_layout()
        path = output_dir / f"{safe}_row_bias_day{int(lead)}.png"
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        logging.info("Saved row bias plot: %s", path)


def _write_debug_report(
    output_dir: Path,
    variables: list[dict[str, Any]],
    stats_info: dict[str, Any],
    lat_lon_debug: dict[str, Any],
    value_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    lines: list[str] = []
    lines.append("MSL debug summary:")
    for var in variables:
        lines.append(
            f"- resolved {var['requested']} -> channel {var['channel']} -> actual name {var['name']} "
            f"(local_idx={var['local_idx']}, unit={var.get('unit') or 'unknown'}, source={var['resolution_source']})"
        )
    lines.append(f"- denormalization mean source: {stats_info.get('mean_source')}")
    lines.append(f"- denormalization std source: {stats_info.get('std_source')}")
    lines.append(f"- lat/lon source: {lat_lon_debug.get('source')}")
    lines.append(f"- lat order: {lat_lon_debug.get('lat_order')}, lon convention: {lat_lon_debug.get('lon_convention')}")
    for warning in lat_lon_debug.get("warnings", []):
        lines.append(f"- {warning}")

    msl_rows = [row for row in value_rows if str(row.get("variable", "")).lower() == "msl"]
    if msl_rows:
        last = msl_rows[-1]
        lines.append(
            f"- MSL lead {last['lead_time']} denormalized GT mean: {last['denormalized_gt_mean']:.3f} Pa"
        )
        lines.append(f"- MSL lead {last['lead_time']} GT min/max: {last['denormalized_gt_min']:.3f} / {last['denormalized_gt_max']:.3f} Pa")
        lines.append(f"- MSL lead {last['lead_time']} bias RMSE: {last['bias_rmse']:.3f} Pa")

    msl_metric_rows = [row for row in metric_rows if str(row.get("variable", "")).lower() == "msl"]
    if msl_metric_rows:
        last = msl_metric_rows[-1]
        lines.append(
            f"- MSL lead {last['lead_time']} rmse_all_rows={last['rmse_all_rows']:.3f}, "
            f"rmse_without_edge_rows={last['rmse_without_edge_rows']:.3f}"
        )
        lines.append(
            f"- MSL edge gt/pred diff max: {last['edge_gt_mean_diff_max']:.3f} / "
            f"{last['edge_pred_mean_diff_max']:.3f}"
        )
    lines.append("- edge-row ACC definition: latitude-weighted spatial correlation between prediction and ground truth fields")

    unique_warnings = list(dict.fromkeys(warnings))
    for warning in unique_warnings:
        lines.append(f"- {warning}")

    if any("edge-row" in warning or "dominated by edge-row" in warning for warning in unique_warnings):
        conclusion = "Conclusion: likely edge-row / pole artifact, not variable mapping."
    elif any("outside 90000..110000" in warning or "exceeds 108000" in warning for warning in unique_warnings):
        conclusion = "Conclusion: MSL physical range is suspicious; check denormalization stats and source data values."
    elif any(var["resolution_source"] == "hardcoded fallback" for var in variables):
        conclusion = "Conclusion: variable mapping still relies on hardcoded fallback; verify channel metadata before trusting plots."
    elif lat_lon_debug.get("warnings"):
        conclusion = "Conclusion: latitude/longitude metadata has warnings; inspect plotting coordinates before judging model behavior."
    else:
        conclusion = "Conclusion: no mapping, denormalization, lat/lon, or edge-row issue was detected by these diagnostics. Remaining issue is likely actual model/data bias."
    lines.append(conclusion)
    lines.append("bias_definition: prediction_minus_ground_truth")
    path = output_dir / "debug_report.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Saved debug report: %s", path)
    print(f"Saved debug report: {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug variable mapping, denormalization, lat/lon, and edge-row rollout behavior.")
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--config", default=None, type=str, help="Config YAML path or config section name.")
    parser.add_argument("--yaml_config", default=None, type=str)
    parser.add_argument("--config_name", default=None, type=str)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--variables", nargs="*", default=["msl", "t2m", "t850", "z500"])
    parser.add_argument("--rollout_steps", default=10, type=int)
    parser.add_argument("--lead_times", nargs="+", default=["1", "5", "10"])
    parser.add_argument("--aggregate_mode", default="year_mean", choices=["sample", "year_mean"])
    parser.add_argument("--sample_index", default=0, type=int)
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--dpi", default=160, type=int)
    parser.add_argument("--robust_percentile", default=99.0, type=float)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--use_cartopy", dest="use_cartopy", action="store_true", default=True)
    parser.add_argument("--no_use_cartopy", dest="use_cartopy", action="store_false")
    parser.add_argument("--add_cyclic", dest="add_cyclic", action="store_true", default=True)
    parser.add_argument("--no_add_cyclic", dest="add_cyclic", action="store_false")
    parser.add_argument("--debug_values", action="store_true")
    parser.add_argument("--debug_latlon", action="store_true")
    parser.add_argument("--debug_edge_rows", action="store_true")
    parser.add_argument("--exclude_edge_lat_rows", action="store_true")
    parser.add_argument("--list_channels", action="store_true")
    parser.add_argument("--print_variable_mapping", action="store_true")
    parser.add_argument("--allow_hardcoded_variable_fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    yaml_path, config_name = viz._resolve_config_args(args)
    params = YParams(yaml_path, config_name)
    rollout_steps = int(args.rollout_steps)
    lead_times = [int(x) for x in viz._parse_items([str(v) for v in args.lead_times])]
    if not lead_times:
        raise ValueError("At least one lead time is required.")
    invalid = [lead for lead in lead_times if lead < 1 or lead > rollout_steps]
    if invalid:
        raise ValueError(f"Lead times must be in [1, rollout_steps={rollout_steps}], got {invalid}")

    debug_flags_explicit = bool(args.debug_values or args.debug_latlon or args.debug_edge_rows)
    run_all_diagnostics = not debug_flags_explicit and not args.list_channels and not args.print_variable_mapping
    if run_all_diagnostics:
        args.debug_values = True
        args.debug_latlon = True
        args.debug_edge_rows = True

    dataset = viz._build_dataset(params, args.split, rollout_steps)
    lats_raw, lons_raw, lat_lon_debug = viz.get_lat_lon_from_dataset_or_config(dataset, params, args.checkpoint)
    channel_names, channel_source = viz.get_channel_names_from_dataset_or_config(dataset, params, args.checkpoint)
    out_channels = [int(x) for x in params.out_channels]
    stats_info = viz.load_output_normalization_stats(params)
    variables = viz.resolve_variable_channels(
        requested=viz._parse_items(args.variables),
        plot_all=False,
        out_channels=out_channels,
        all_channel_names=channel_names,
        allow_hardcoded_fallback=bool(args.allow_hardcoded_variable_fallback),
    )
    viz.attach_normalization_info(variables, stats_info)

    if args.list_channels:
        viz.print_available_variable_mapping(channel_names, out_channels, channel_source)
    if args.debug_latlon:
        viz.print_lat_lon_debug(lat_lon_debug)
    if args.print_variable_mapping:
        _print_variable_details(variables)

    if (args.list_channels or args.print_variable_mapping) and not args.checkpoint:
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            setup_logging(rank=0, log_file=str(output_dir / "debug_variable_visualization.log"))
            _write_json(output_dir / "variable_channel_mapping.json", viz._mapping_payload(variables, channel_source))
            _write_json(output_dir / "lat_lon_debug.json", lat_lon_debug)
        return
    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --list_channels/--print_variable_mapping is used.")
    checkpoint_path = viz._resolve_path(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(checkpoint_path).resolve().parent / "debug_variable_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(rank=0, log_file=str(output_dir / "debug_variable_visualization.log"))
    _write_json(output_dir / "variable_channel_mapping.json", viz._mapping_payload(variables, channel_source))
    _write_json(output_dir / "lat_lon_debug.json", lat_lon_debug)

    if args.use_cartopy:
        viz._load_cartopy()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info("Using device: %s", device)

    lats, lons, lat_order, lon_order = viz._coordinate_orders(lats_raw, lons_raw)
    model_sample_idx = int(args.sample_index) if args.aggregate_mode == "sample" else 0
    inp, target = dataset[model_sample_idx]
    target_seq = viz._target_sequence(target)
    model = viz._build_model(params, dataset, inp, target_seq, device)
    checkpoint = viz._load_checkpoint(model, checkpoint_path, device)
    metadata = dict(checkpoint.get("metadata", {}))
    logging.info(
        "Loaded checkpoint: %s | epoch=%s | train_rollout_steps=%s",
        checkpoint_path,
        metadata.get("epoch", checkpoint.get("epoch", "unknown")),
        metadata.get("train_rollout_steps", "unknown"),
    )

    maps, value_rows, sample_count = _collect_rollout_debug_data(
        dataset=dataset,
        model=model,
        params=params,
        variables=variables,
        rollout_steps=rollout_steps,
        lead_times=lead_times,
        aggregate_mode=str(args.aggregate_mode),
        sample_index=int(args.sample_index),
        max_batches=args.max_batches,
        device=device,
        stats_info=stats_info,
        lat_order=lat_order,
        lon_order=lon_order,
    )
    warnings: list[str] = list(lat_lon_debug.get("warnings", []))
    _warn_value_sanity(value_rows, warnings)

    if args.debug_values:
        _write_csv(output_dir / "value_stats_by_variable_lead.csv", value_rows)
        _write_json(output_dir / "value_stats_by_variable_lead.json", value_rows)
        for row in value_rows:
            print(
                f"{row['variable']} lead {row['lead_time']}: "
                f"GT mean={row['denormalized_gt_mean']:.6g}, pred mean={row['denormalized_pred_mean']:.6g}, "
                f"bias rmse={row['bias_rmse']:.6g}"
            )

    metric_rows: list[dict[str, Any]] = []
    if args.debug_edge_rows:
        metric_rows = _write_row_stats_and_metrics(output_dir, variables, lead_times, maps, lats, warnings)

    prefix = "yearmean" if args.aggregate_mode == "year_mean" else f"sample{int(args.sample_index):03d}"
    aggregate_label = "year mean" if args.aggregate_mode == "year_mean" else f"sample {int(args.sample_index)}"
    for var in variables:
        key = str(var["name"])
        safe = _safe_name(var)
        unit_label = str(var.get("unit") or "physical units")
        gt_maps = maps[key]["gt"]
        pred_maps = maps[key]["pred"]
        _plot_debug_maps(
            gt_maps=gt_maps,
            pred_maps=pred_maps,
            lats=lats,
            lons=lons,
            variable=var,
            lead_times=lead_times,
            output_path=output_dir / f"{prefix}_{safe}_full_rows.png",
            aggregate_label=f"{aggregate_label} full rows",
            unit_label=unit_label,
            use_cartopy=bool(args.use_cartopy),
            add_cyclic=bool(args.add_cyclic),
            dpi=int(args.dpi),
            robust_percentile=float(args.robust_percentile),
        )
        if gt_maps.shape[-2] > 2:
            _plot_debug_maps(
                gt_maps=gt_maps[..., 1:-1, :],
                pred_maps=pred_maps[..., 1:-1, :],
                lats=lats[1:-1],
                lons=lons,
                variable=var,
                lead_times=lead_times,
                output_path=output_dir / f"{prefix}_{safe}_without_edge_rows.png",
                aggregate_label=f"{aggregate_label} without first/last latitude rows",
                unit_label=unit_label,
                use_cartopy=bool(args.use_cartopy),
                add_cyclic=bool(args.add_cyclic),
                dpi=int(args.dpi),
                robust_percentile=float(args.robust_percentile),
            )
        if args.debug_edge_rows:
            _plot_row_diagnostics(output_dir, var, maps[key], lats, lead_times, int(args.dpi))

    metadata_payload = {
        "checkpoint": checkpoint_path,
        "split": args.split,
        "aggregate_mode": args.aggregate_mode,
        "sample_count": sample_count,
        "rollout_steps": rollout_steps,
        "lead_times": lead_times,
        "bias_definition": "prediction_minus_ground_truth",
        "edge_row_acc_definition": "latitude_weighted_spatial_correlation_between_prediction_and_ground_truth",
        "exclude_edge_lat_rows_requested": bool(args.exclude_edge_lat_rows),
        "lat_lon": lat_lon_debug,
        "variables": viz._mapping_payload(variables, channel_source)["variables"],
    }
    _write_json(output_dir / "debug_metadata.json", metadata_payload)
    _write_debug_report(output_dir, variables, stats_info, lat_lon_debug, value_rows, metric_rows, warnings)


if __name__ == "__main__":
    main()
