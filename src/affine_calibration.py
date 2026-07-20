from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .features import canonical_name


DEFAULT_CALIBRATION_VARIABLES = ["z500", "t2m", "t850", "msl", "q700", "u850"]
VALID_CALIBRATION_APPLY_MODES = ("output_only", "autoregressive_state")


def json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def validate_fit_split(fit_split: str, *, allow_fit_on_test: bool = False) -> None:
    if str(fit_split).strip().lower() == "test" and not bool(allow_fit_on_test):
        raise ValueError("Refusing to fit affine calibration on eval/test split. Pass --allow_fit_on_test to override.")


def validate_apply_mode(mode: str) -> str:
    normalized = str(mode or "output_only").strip().lower()
    if normalized not in VALID_CALIBRATION_APPLY_MODES:
        raise ValueError(
            f"Unsupported calibration_apply_mode={mode!r}. "
            f"Expected one of: {', '.join(VALID_CALIBRATION_APPLY_MODES)}"
        )
    return normalized


def empty_affine_accumulator(n_leads: int, n_variables: int) -> dict[str, np.ndarray]:
    shape = (int(n_leads), int(n_variables))
    return {
        "sum_w": np.zeros(shape, dtype=np.float64),
        "sum_p": np.zeros(shape, dtype=np.float64),
        "sum_t": np.zeros(shape, dtype=np.float64),
        "sum_p2": np.zeros(shape, dtype=np.float64),
        "sum_t2": np.zeros(shape, dtype=np.float64),
        "sum_pt": np.zeros(shape, dtype=np.float64),
        "sum_err2_before": np.zeros(shape, dtype=np.float64),
    }


def update_affine_accumulator(
    accumulator: dict[str, np.ndarray],
    *,
    lead_idx: int,
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    lat_weights: torch.Tensor,
    variable_indices: list[int],
) -> None:
    if not variable_indices:
        return
    lead_idx = int(lead_idx)
    indices = torch.as_tensor(variable_indices, device=pred_norm.device, dtype=torch.long)
    p = pred_norm.index_select(1, indices).detach().to(dtype=torch.float64)
    t = target_norm.index_select(1, indices).detach().to(dtype=torch.float64)
    weight = lat_weights.to(device=pred_norm.device, dtype=torch.float64).view(1, 1, -1, 1)
    weighted_count = float(lat_weights.detach().double().sum().item()) * float(p.shape[0] * p.shape[-1])

    accumulator["sum_w"][lead_idx] += weighted_count
    accumulator["sum_p"][lead_idx] += (p * weight).sum(dim=(0, 2, 3)).cpu().numpy()
    accumulator["sum_t"][lead_idx] += (t * weight).sum(dim=(0, 2, 3)).cpu().numpy()
    accumulator["sum_p2"][lead_idx] += (p.square() * weight).sum(dim=(0, 2, 3)).cpu().numpy()
    accumulator["sum_t2"][lead_idx] += (t.square() * weight).sum(dim=(0, 2, 3)).cpu().numpy()
    accumulator["sum_pt"][lead_idx] += ((p * t) * weight).sum(dim=(0, 2, 3)).cpu().numpy()
    accumulator["sum_err2_before"][lead_idx] += ((p - t).square() * weight).sum(dim=(0, 2, 3)).cpu().numpy()


def finalize_affine_coefficients(
    accumulator: dict[str, np.ndarray],
    variables: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    sum_w = np.asarray(accumulator["sum_w"], dtype=np.float64)
    sum_p = np.asarray(accumulator["sum_p"], dtype=np.float64)
    sum_t = np.asarray(accumulator["sum_t"], dtype=np.float64)
    sum_p2 = np.asarray(accumulator["sum_p2"], dtype=np.float64)
    sum_t2 = np.asarray(accumulator["sum_t2"], dtype=np.float64)
    sum_pt = np.asarray(accumulator["sum_pt"], dtype=np.float64)
    sum_err2_before = np.asarray(accumulator["sum_err2_before"], dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        pred_mean = sum_p / sum_w
        target_mean = sum_t / sum_w
        pred_var = np.maximum(sum_p2 / sum_w - pred_mean * pred_mean, 0.0)
        target_var = np.maximum(sum_t2 / sum_w - target_mean * target_mean, 0.0)
        cov = sum_pt / sum_w - pred_mean * target_mean
        a = cov / (pred_var + float(eps))
        b = target_mean - a * pred_mean

    invalid = ~np.isfinite(a) | ~np.isfinite(b) | (sum_w <= 0.0)
    a[invalid] = 1.0
    b[invalid] = 0.0

    sum_err2_after = (
        a * a * sum_p2
        + sum_t2
        + b * b * sum_w
        + 2.0 * a * b * sum_p
        - 2.0 * a * sum_pt
        - 2.0 * b * sum_t
    )
    sum_err2_after = np.maximum(sum_err2_after, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        fit_rmse_before = np.sqrt(sum_err2_before / sum_w)
        fit_rmse_after = np.sqrt(sum_err2_after / sum_w)
        pred_std = np.sqrt(pred_var)
        target_std = np.sqrt(target_var)

    n_leads = int(sum_w.shape[0])
    payload: dict[str, Any] = {
        "metadata": {
            **metadata,
            "calibration_space": str(metadata.get("calibration_space", "normalized")),
            "eps": float(eps),
        },
        "leads": list(range(1, n_leads + 1)),
        "variables": variables,
        "coefficients": {},
    }
    for var_pos, variable in enumerate(variables):
        key = str(variable.get("canonical_name") or variable.get("name") or f"var{var_pos}")
        payload["coefficients"][key] = {}
        for lead_idx in range(n_leads):
            payload["coefficients"][key][str(lead_idx + 1)] = {
                "a": json_number(a[lead_idx, var_pos]),
                "b": json_number(b[lead_idx, var_pos]),
                "pred_mean": json_number(pred_mean[lead_idx, var_pos]),
                "target_mean": json_number(target_mean[lead_idx, var_pos]),
                "pred_std": json_number(pred_std[lead_idx, var_pos]),
                "target_std": json_number(target_std[lead_idx, var_pos]),
                "fit_rmse_before": json_number(fit_rmse_before[lead_idx, var_pos]),
                "fit_rmse_after": json_number(fit_rmse_after[lead_idx, var_pos]),
            }
    return payload


def coefficient_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    variables = list(payload.get("variables", []))
    coeffs = dict(payload.get("coefficients", {}))
    rows: list[dict[str, Any]] = []
    for variable in variables:
        key = str(variable.get("canonical_name") or variable.get("name"))
        for lead in payload.get("leads", []):
            values = coeffs.get(key, {}).get(str(int(lead)), {})
            rows.append(
                {
                    "variable": key,
                    "lead": int(lead),
                    "a": values.get("a"),
                    "b": values.get("b"),
                    "pred_mean": values.get("pred_mean"),
                    "target_mean": values.get("target_mean"),
                    "pred_std": values.get("pred_std"),
                    "target_std": values.get("target_std"),
                    "fit_rmse_before": values.get("fit_rmse_before"),
                    "fit_rmse_after": values.get("fit_rmse_after"),
                }
            )
    return rows


def save_affine_coefficients(payload: dict[str, Any], json_path: str | Path, csv_path: str | Path) -> None:
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = coefficient_rows(payload)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variable",
                "lead",
                "a",
                "b",
                "pred_mean",
                "target_mean",
                "pred_std",
                "target_std",
                "fit_rmse_before",
                "fit_rmse_after",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class AffineCalibration:
    payload: dict[str, Any]
    path: str
    resolved: dict[int, dict[int, tuple[float, float, str]]]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        variable_names: list[str],
        out_channels: list[int],
        logger: logging.Logger | Any = logging,
    ) -> "AffineCalibration":
        path = str(Path(path).expanduser())
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        resolved = resolve_calibration_indices(payload, variable_names, out_channels, logger=logger)
        return cls(payload=payload, path=path, resolved=resolved)

    @property
    def variables(self) -> list[str]:
        return sorted({entry[2] for lead_map in self.resolved.values() for entry in lead_map.values()})

    @property
    def leads(self) -> list[int]:
        return sorted({int(lead) for lead_map in self.resolved.values() for lead in lead_map})

    def apply(self, pred_norm: torch.Tensor, lead: int, *, in_place: bool = False) -> torch.Tensor:
        return apply_affine_calibration_to_tensor(pred_norm, int(lead), self.resolved, in_place=in_place)


def resolve_calibration_indices(
    payload: dict[str, Any],
    variable_names: list[str],
    out_channels: list[int],
    *,
    logger: logging.Logger | Any = logging,
) -> dict[int, dict[int, tuple[float, float, str]]]:
    coeffs = dict(payload.get("coefficients", {}))
    out_channels = [int(x) for x in out_channels]
    by_local: dict[int, str] = {}
    for variable in payload.get("variables", []):
        key = str(variable.get("canonical_name") or variable.get("name") or "")
        if not key or key == "orog":
            continue
        local = variable.get("variable_idx")
        channel = variable.get("channel")
        if local is not None and 0 <= int(local) < len(out_channels):
            by_local[int(local)] = key
        elif channel is not None and int(channel) in out_channels:
            by_local[out_channels.index(int(channel))] = key

    for local_idx, name in enumerate(variable_names):
        key = canonical_name(str(name)) or str(name)
        if key in coeffs and key != "orog":
            by_local.setdefault(int(local_idx), key)

    resolved: dict[int, dict[int, tuple[float, float, str]]] = {}
    for local_idx, key in sorted(by_local.items()):
        if key not in coeffs:
            logger.warning("Affine calibration variable %s has no coefficient payload; skipping.", key)
            continue
        lead_map: dict[int, tuple[float, float, str]] = {}
        for lead_text, values in coeffs[key].items():
            try:
                lead = int(lead_text)
                a = float(values["a"])
                b = float(values["b"])
            except (KeyError, TypeError, ValueError):
                logger.warning("Invalid affine coefficient for %s lead %s; skipping.", key, lead_text)
                continue
            if np.isfinite(a) and np.isfinite(b):
                lead_map[lead] = (a, b, key)
        if lead_map:
            resolved[int(local_idx)] = lead_map
    return resolved


def apply_affine_calibration_to_tensor(
    pred_norm: torch.Tensor,
    lead: int,
    resolved: dict[int, dict[int, tuple[float, float, str]]] | None,
    *,
    in_place: bool = False,
) -> torch.Tensor:
    if not resolved:
        return pred_norm
    out = pred_norm if in_place else pred_norm.clone()
    lead = int(lead)
    for local_idx, lead_map in resolved.items():
        coeff = lead_map.get(lead)
        if coeff is None:
            continue
        a, b, _ = coeff
        out[:, int(local_idx)] = out[:, int(local_idx)] * float(a) + float(b)
    return out
