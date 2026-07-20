from __future__ import annotations

from typing import Any

import numpy as np


def normalized_latitude_weights(latitudes_deg: np.ndarray) -> np.ndarray:
    """WeatherBench-style cosine latitude weights with mean 1."""
    latitudes = np.asarray(latitudes_deg, dtype=np.float64).reshape(-1)
    weights = np.cos(np.deg2rad(latitudes)).clip(min=0.0)
    if not np.isfinite(weights).all() or float(weights.mean()) <= 0.0:
        weights = np.ones_like(latitudes, dtype=np.float64)
    return weights / float(weights.mean())


def weighted_mse_per_ic(
    predictions: np.ndarray,
    targets: np.ndarray,
    latitudes_deg: np.ndarray | None = None,
    latitude_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return WeatherBench2-style per-IC MSE for [N, lead, variable, H, W] arrays.

    The final official RMSE is sqrt(mean(per_ic_mse, axis=0)). This function
    intentionally returns MSE so callers can do paired bootstrap resampling
    before the final square root.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"predictions and targets must have the same shape, got {pred.shape} and {target.shape}")
    if pred.ndim != 5:
        raise ValueError(f"Expected [N_ic, lead, variable, H, W] arrays, got shape {pred.shape}")

    height = int(pred.shape[-2])
    if latitude_weights is None:
        if latitudes_deg is None:
            raise ValueError("Either latitudes_deg or latitude_weights is required.")
        weights = normalized_latitude_weights(np.asarray(latitudes_deg, dtype=np.float64))
    else:
        weights = np.asarray(latitude_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != height:
            raise ValueError(f"latitude_weights length {weights.shape[0]} does not match height {height}")
        if float(weights.mean()) > 0.0:
            weights = weights / float(weights.mean())
    if weights.shape[0] != height:
        raise ValueError(f"Latitude weight length {weights.shape[0]} does not match height {height}")

    squared_error = np.square(pred - target)
    return np.nanmean(squared_error * weights.reshape(1, 1, 1, height, 1), axis=(-2, -1))


def rmse_from_per_ic_mse(per_ic_mse: np.ndarray) -> np.ndarray:
    """Compute official WeatherBench2 RMSE: sqrt after IC averaging."""
    return np.sqrt(np.nanmean(np.asarray(per_ic_mse, dtype=np.float64), axis=0))


def mean_per_ic_rmse(per_ic_mse: np.ndarray) -> np.ndarray:
    """Diagnostic legacy aggregation: mean of per-IC RMSEs."""
    return np.nanmean(np.sqrt(np.asarray(per_ic_mse, dtype=np.float64)), axis=0)


def weatherbench2_package_mse(
    predictions: np.ndarray,
    targets: np.ndarray,
    latitudes_deg: np.ndarray,
    variable_names: list[str] | None = None,
) -> np.ndarray:
    """Compute MSE through weatherbench2.metrics.MSE when the package is installed.

    This helper is intentionally isolated because WeatherBench2 is optional in
    this repository's environment. It returns the same [lead, variable] MSE
    aggregation as rmse_from_per_ic_mse(... ) ** 2.
    """
    try:
        import xarray as xr
        from weatherbench2.metrics import MSE
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("weatherbench2 and xarray are required for this optional backend.") from exc

    pred = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"predictions and targets must have the same shape, got {pred.shape} and {target.shape}")
    if pred.ndim != 5:
        raise ValueError(f"Expected [N_ic, lead, variable, H, W] arrays, got shape {pred.shape}")

    n_ic, n_lead, n_var, height, width = pred.shape
    variables = variable_names or [f"var_{idx}" for idx in range(n_var)]
    if len(variables) != n_var:
        raise ValueError(f"variable_names length {len(variables)} does not match variable dimension {n_var}")
    coords = {
        "time": np.arange(n_ic),
        "prediction_timedelta": np.arange(1, n_lead + 1),
        "latitude": np.asarray(latitudes_deg, dtype=np.float64).reshape(height),
        "longitude": np.arange(width),
    }
    forecast_vars: dict[str, Any] = {}
    truth_vars: dict[str, Any] = {}
    for var_idx, name in enumerate(variables):
        dims = ("time", "prediction_timedelta", "latitude", "longitude")
        forecast_vars[str(name)] = (dims, pred[:, :, var_idx])
        truth_vars[str(name)] = (dims, target[:, :, var_idx])
    forecast = xr.Dataset(forecast_vars, coords=coords)
    truth = xr.Dataset(truth_vars, coords=coords)
    result = MSE().compute(forecast, truth)
    return np.stack(
        [
            np.asarray(result[str(name)].transpose("prediction_timedelta").values, dtype=np.float64)
            for name in variables
        ],
        axis=-1,
    )
