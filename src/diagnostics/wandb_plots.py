from __future__ import annotations

import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is part of repo requirements
    pd = None  # type: ignore

try:
    import plotly.express as px  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    px = None

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _plots_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("plots", {}) or {})


def _wandb_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("wandb", {}) or {})


def _as_dataframe(data: Any):
    if pd is None:
        raise RuntimeError("pandas is required for diagnostics tables.")
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(list(data))


def _ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "tables": output_dir / "tables",
        "plots": output_dir / "plots",
        "maps": output_dir / "maps",
        "histograms": output_dir / "histograms",
        "artifacts": output_dir / "artifacts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _get_pyplot():
    mpl_dir = Path(os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "graphweather_mplconfig"))
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path(tempfile.gettempdir()) / "graphweather_cache"))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        probe = cache_dir / ".write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except Exception:
        cache_dir = Path(tempfile.gettempdir()) / "graphweather_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def wandb_available_and_enabled(config: dict[str, Any], run: Any = None) -> bool:
    wandb_config = _wandb_cfg(config)
    if wandb is None or not bool(wandb_config.get("enabled", False)):
        return False
    active_run = run if run is not None else getattr(wandb, "run", None)
    return active_run is not None


def _wandb_table(df: Any):
    if wandb is None or df is None or len(df) == 0:
        return None
    return wandb.Table(dataframe=df)


def _wandb_log(run: Any, payload: dict[str, Any], *, step: int | None, logger: Any = logging) -> bool:
    if run is None or not payload:
        return False
    try:
        run.log(payload, step=step)
        return True
    except Exception as exc:  # diagnostics must not interrupt training
        logger.warning("W&B diagnostics plot logging skipped: %s", exc)
        return False


def save_local_plot(fig: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=140, bbox_inches="tight")
    return target


def _log_matplotlib_image(run: Any, key: str, fig: Any, *, step: int | None, logger: Any = logging) -> bool:
    if wandb is None or run is None:
        return False
    try:
        return _wandb_log(run, {key: wandb.Image(fig)}, step=step, logger=logger)
    except Exception as exc:  # diagnostics must not interrupt training
        logger.warning("W&B diagnostics image logging skipped for %s: %s", key, exc)
        return False


def _log_plotly_or_image(
    run: Any,
    key: str,
    plotly_fig: Any,
    matplotlib_fig: Any,
    *,
    step: int | None,
    logger: Any = logging,
) -> bool:
    if wandb is None or run is None:
        return False
    if plotly_fig is not None and hasattr(wandb, "Plotly"):
        try:
            return _wandb_log(run, {key: wandb.Plotly(plotly_fig)}, step=step, logger=logger)
        except Exception as exc:
            logger.warning("W&B Plotly logging skipped for %s: %s", key, exc)
    return _log_matplotlib_image(run, key, matplotlib_fig, step=step, logger=logger)


def _line_fig(df: Any, *, x: str, y: str, color: str | None, title: str, ylabel: str | None = None):
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    if color and color in df.columns:
        for label, part in df.groupby(color, sort=False):
            ax.plot(part[x], part[y], marker="o", linewidth=1.8, label=str(label))
        ax.legend(loc="best", fontsize=8)
    else:
        ax.plot(df[x], df[y], marker="o", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _bar_fig(df: Any, *, x: str, y: str, color: str | None, title: str, ylabel: str | None = None):
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    if color and color in df.columns:
        labels = list(dict.fromkeys(df[x].astype(str).tolist()))
        groups = list(dict.fromkeys(df[color].astype(str).tolist()))
        width = 0.8 / max(len(groups), 1)
        xpos = np.arange(len(labels), dtype=np.float64)
        for idx, group in enumerate(groups):
            part = df[df[color].astype(str) == group]
            values = []
            for label in labels:
                row = part[part[x].astype(str) == label]
                values.append(float(row[y].iloc[0]) if len(row) else np.nan)
            ax.bar(xpos + (idx - (len(groups) - 1) / 2.0) * width, values, width=width, label=group)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.legend(loc="best", fontsize=8)
    else:
        ax.bar(df[x].astype(str), df[y])
        ax.tick_params(axis="x", rotation=25)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _heatmap_fig(matrix: np.ndarray, *, x_labels: list[str], y_labels: list[str], title: str, cbar_label: str):
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(8.0, max(3.5, min(9.0, 0.35 * max(len(y_labels), 1) + 2.0))))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("rollout_step")
    ax.set_ylabel("epoch")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    return fig


def _hist_fig(values: np.ndarray, *, title: str, xlabel: str):
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    finite = values[np.isfinite(values)]
    if finite.size:
        ax.hist(finite, bins=80)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _map_fig(array: np.ndarray, *, title: str, cmap: str | None = None):
    plt = _get_pyplot()
    arr = np.asarray(array, dtype=np.float64)
    if cmap is None:
        cmap = "coolwarm" if np.nanmin(arr) < 0.0 < np.nanmax(arr) else "viridis"
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    im = ax.imshow(arr, origin="upper", aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("longitude index")
    ax.set_ylabel("latitude index")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _plotly_line(df: Any, *, x: str, y: str, color: str | None, title: str):
    if px is None:
        return None
    try:
        return px.line(df, x=x, y=y, color=color, markers=True, title=title)
    except Exception:
        return None


def _plotly_bar(df: Any, *, x: str, y: str, color: str | None, title: str):
    if px is None:
        return None
    try:
        return px.bar(df, x=x, y=y, color=color, barmode="group", title=title)
    except Exception:
        return None


def _save_table(df: Any, path: Path) -> bool:
    if df is None or len(df) == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return True


def _maybe_log_table(
    run: Any,
    config: dict[str, Any],
    key: str,
    df: Any,
    *,
    step: int | None,
    logger: Any = logging,
) -> bool:
    if not wandb_available_and_enabled(config, run) or not bool(_wandb_cfg(config).get("log_tables", True)):
        return False
    table = _wandb_table(df)
    if table is None:
        return False
    return _wandb_log(run, {key: table}, step=step, logger=logger)


def _maybe_log_plot(
    run: Any,
    config: dict[str, Any],
    key: str,
    *,
    plotly_fig: Any,
    matplotlib_fig: Any,
    step: int | None,
    logger: Any = logging,
) -> bool:
    if not wandb_available_and_enabled(config, run) or not bool(_wandb_cfg(config).get("log_plots", False)):
        return False
    return _log_plotly_or_image(run, key, plotly_fig, matplotlib_fig, step=step, logger=logger)


def _close_fig(fig: Any) -> None:
    if fig is None:
        return
    try:
        plt = _get_pyplot()
        plt.close(fig)
    except Exception:
        pass


def add_layer_index(df: Any):
    out = df.copy()
    if "layer_index" in out.columns:
        return out
    order = {}
    indices = []
    for name in out.get("layer_name", []):
        key = str(name)
        if key not in order:
            order[key] = len(order)
        indices.append(order[key])
    out["layer_index"] = indices
    out["layer_label"] = [f"{idx}:{name}" for idx, name in zip(out["layer_index"], out.get("layer_name", []))]
    return out


def log_wandb_scalars(metrics: dict[str, Any], step: int, prefix: str = "", run: Any = None, config: dict[str, Any] | None = None) -> bool:
    if wandb is None:
        return False
    cfg = config or {"wandb": {"enabled": True}}
    if not wandb_available_and_enabled(cfg, run):
        return False
    active_run = run if run is not None else wandb.run
    payload = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            payload[f"{prefix}{key}" if prefix else str(key)] = float(value)
    return _wandb_log(active_run, payload, step=step) if payload else False


def log_rollout_tables_and_plots(
    rollout_df: Any,
    *,
    config: dict[str, Any],
    run: Any,
    output_dir: Path,
    step: int,
    phase: str,
    epoch: int,
    logger: Any = logging,
) -> dict[str, Any]:
    df = _as_dataframe(rollout_df)
    result = {"tables": [], "plots": [], "warnings": []}
    if df.empty:
        result["warnings"].append("Skipped rollout plots: no rollout rows")
        return result
    dirs = _ensure_dirs(output_dir)
    stem = f"epoch_{int(epoch):04d}_{_safe_name(phase)}"
    table_path = dirs["tables"] / f"{stem}_rollout_curve.csv"
    if _save_table(df, table_path):
        result["tables"].append("rollout")
    if _maybe_log_table(run, config, f"tables/{phase}/rollout_curve", df, step=step, logger=logger):
        result["tables"].append("wandb_rollout")

    final_rows = []
    for horizon, part in df.groupby("horizon", sort=True):
        part = part.sort_values("step")
        final_loss = float(part["loss"].iloc[-1])
        mean_loss = float(part["loss"].mean())
        final_rows.append(
            {
                "epoch": int(epoch),
                "phase": phase,
                "horizon": int(horizon),
                "horizon_label": f"S{int(horizon)}",
                "loss_final": final_loss,
                "loss_mean": mean_loss,
                "final_to_mean_ratio": final_loss / (mean_loss + 1.0e-12) if math.isfinite(mean_loss) else np.nan,
            }
        )
    final_df = _as_dataframe(final_rows)
    _save_table(final_df, dirs["tables"] / f"{stem}_rollout_summary.csv")

    plot_specs = [
        (
            "rollout_step_loss",
            df,
            "step",
            "loss",
            "horizon",
            "Rollout step loss",
            "loss",
            _line_fig,
            _plotly_line,
        ),
        (
            "final_loss_by_horizon",
            final_df,
            "horizon_label",
            "loss_final",
            None,
            "Final loss by rollout horizon",
            "loss_final",
            _bar_fig,
            _plotly_bar,
        ),
        (
            "final_to_mean_ratio",
            final_df,
            "horizon_label",
            "final_to_mean_ratio",
            None,
            "Final-to-mean rollout loss ratio",
            "final_to_mean_ratio",
            _bar_fig,
            _plotly_bar,
        ),
    ]
    for name, data, x, y, color, title, ylabel, mpl_factory, plotly_factory in plot_specs:
        fig = mpl_factory(data, x=x, y=y, color=color, title=title, ylabel=ylabel)
        save_local_plot(fig, dirs["plots"] / f"{stem}_{name}.png")
        plotly_fig = plotly_factory(data, x=x, y=y, color=color, title=title)
        result["plots"].append(name)
        _maybe_log_plot(run, config, f"plots/{phase}/{name}", plotly_fig=plotly_fig, matplotlib_fig=fig, step=step, logger=logger)
        _close_fig(fig)
    return result


def log_layer_tables_and_plots(
    layer_df: Any,
    *,
    config: dict[str, Any],
    run: Any,
    output_dir: Path,
    step: int,
    phase: str,
    epoch: int,
    logger: Any = logging,
) -> dict[str, Any]:
    df = add_layer_index(_as_dataframe(layer_df))
    result = {"tables": [], "plots": [], "warnings": []}
    if df.empty:
        result["warnings"].append("Skipped layer plots: no layer rows")
        return result
    dirs = _ensure_dirs(output_dir)
    stem = f"epoch_{int(epoch):04d}_{_safe_name(phase)}"
    if _save_table(df, dirs["tables"] / f"{stem}_layer_metrics.csv"):
        result["tables"].append("layer")
    if _maybe_log_table(run, config, f"tables/{phase}/layer_metrics", df, step=step, logger=logger):
        result["tables"].append("wandb_layer")
    for metric, title in (
        ("cosine_mean", "Layer cosine mean"),
        ("mad_cosine", "Layer MAD cosine"),
        ("effective_rank_norm", "Layer normalized effective rank"),
        ("embedding_variance", "Layer embedding variance"),
    ):
        if metric not in df.columns:
            continue
        fig = _line_fig(df, x="layer_index", y=metric, color=None, title=title, ylabel=metric)
        ax = fig.axes[0]
        ax.set_xticks(df["layer_index"])
        ax.set_xticklabels(df["layer_name"], rotation=45, ha="right", fontsize=7)
        fig.tight_layout()
        name = f"layer_{metric}"
        save_local_plot(fig, dirs["plots"] / f"{stem}_{name}.png")
        plotly_fig = _plotly_line(df, x="layer_index", y=metric, color=None, title=title)
        result["plots"].append(name)
        _maybe_log_plot(run, config, f"plots/{phase}/{name}", plotly_fig=plotly_fig, matplotlib_fig=fig, step=step, logger=logger)
        _close_fig(fig)
    return result


def log_attention_tables_and_plots(
    attention_df: Any,
    *,
    head_df: Any = None,
    config: dict[str, Any],
    run: Any,
    output_dir: Path,
    step: int,
    phase: str,
    epoch: int,
    logger: Any = logging,
) -> dict[str, Any]:
    df = add_layer_index(_as_dataframe(attention_df))
    result = {"tables": [], "plots": [], "warnings": []}
    if df.empty:
        result["warnings"].append("Skipped attention plots: no attention rows")
        return result
    dirs = _ensure_dirs(output_dir)
    stem = f"epoch_{int(epoch):04d}_{_safe_name(phase)}"
    if "uniform_baseline_max_weight" in df.columns and "max_weight_mean" in df.columns:
        df["selectivity_ratio"] = df["max_weight_mean"] / (df["uniform_baseline_max_weight"] + 1.0e-12)
    if _save_table(df, dirs["tables"] / f"{stem}_attention_metrics.csv"):
        result["tables"].append("attention")
    if _maybe_log_table(run, config, f"tables/{phase}/attention_metrics", df, step=step, logger=logger):
        result["tables"].append("wandb_attention")

    for metric, title in (
        ("entropy_norm", "Attention entropy norm by layer"),
        ("max_weight_mean", "Attention max weight by layer"),
        ("selectivity_ratio", "Attention selectivity ratio"),
    ):
        if metric not in df.columns:
            continue
        fig = _line_fig(df, x="layer_index", y=metric, color=None, title=title, ylabel=metric)
        ax = fig.axes[0]
        ax.set_xticks(df["layer_index"])
        ax.set_xticklabels(df["layer_name"], rotation=45, ha="right", fontsize=7)
        if metric == "max_weight_mean" and "uniform_baseline_max_weight" in df.columns:
            ax.plot(df["layer_index"], df["uniform_baseline_max_weight"], linestyle="--", marker="x", label="uniform baseline")
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        name = "attention_max_weight" if metric == "max_weight_mean" else f"attention_{metric}"
        save_local_plot(fig, dirs["plots"] / f"{stem}_{name}.png")
        plotly_fig = _plotly_line(df, x="layer_index", y=metric, color=None, title=title)
        result["plots"].append(name)
        _maybe_log_plot(run, config, f"plots/{phase}/{name}", plotly_fig=plotly_fig, matplotlib_fig=fig, step=step, logger=logger)
        _close_fig(fig)

    hdf = _as_dataframe(head_df)
    if not hdf.empty:
        _save_table(hdf, dirs["tables"] / f"{stem}_attention_head_metrics.csv")
        _maybe_log_table(run, config, f"tables/{phase}/attention_head_metrics", hdf, step=step, logger=logger)
        if "entropy_norm" in hdf.columns:
            fig = _line_fig(hdf, x="head", y="entropy_norm", color="layer_name", title="Attention entropy per head", ylabel="entropy_norm")
            save_local_plot(fig, dirs["plots"] / f"{stem}_attention_entropy_per_head.png")
            plotly_fig = _plotly_line(hdf, x="head", y="entropy_norm", color="layer_name", title="Attention entropy per head")
            result["plots"].append("attention_entropy_per_head")
            _maybe_log_plot(
                run,
                config,
                f"plots/{phase}/attention_entropy_per_head",
                plotly_fig=plotly_fig,
                matplotlib_fig=fig,
                step=step,
                logger=logger,
            )
            _close_fig(fig)
    return result


def log_spectral_tables_and_plots(
    spectral_curve_df: Any,
    spectral_band_df: Any,
    *,
    config: dict[str, Any],
    run: Any,
    output_dir: Path,
    step: int,
    phase: str,
    epoch: int,
    logger: Any = logging,
) -> dict[str, Any]:
    curve_df = _as_dataframe(spectral_curve_df)
    band_df = _as_dataframe(spectral_band_df)
    result = {"tables": [], "plots": [], "warnings": []}
    if curve_df.empty and band_df.empty:
        result["warnings"].append("Skipped spectral plots: no spectral rows")
        return result
    dirs = _ensure_dirs(output_dir)
    stem = f"epoch_{int(epoch):04d}_{_safe_name(phase)}"
    if not curve_df.empty and _save_table(curve_df, dirs["tables"] / f"{stem}_spectral_curve.csv"):
        result["tables"].append("spectral_curve")
        _maybe_log_table(run, config, f"tables/{phase}/spectral_curve", curve_df, step=step, logger=logger)
    if not band_df.empty and _save_table(band_df, dirs["tables"] / f"{stem}_spectral_bands.csv"):
        result["tables"].append("spectral_bands")
        _maybe_log_table(run, config, f"tables/{phase}/spectral_bands", band_df, step=step, logger=logger)

    if not band_df.empty:
        for horizon, part in band_df.groupby("horizon", sort=True):
            fig = _bar_fig(part, x="band", y="spectral_rmse", color="variable", title=f"Spectral bands S{int(horizon)}", ylabel="spectral_rmse")
            name = f"spectral_bands_S{int(horizon)}"
            save_local_plot(fig, dirs["plots"] / f"{stem}_{name}.png")
            plotly_fig = _plotly_bar(part, x="band", y="spectral_rmse", color="variable", title=f"Spectral bands S{int(horizon)}")
            result["plots"].append(name)
            _maybe_log_plot(
                run,
                config,
                f"plots/{phase}/spectral_bands/S{int(horizon)}",
                plotly_fig=plotly_fig,
                matplotlib_fig=fig,
                step=step,
                logger=logger,
            )
            _close_fig(fig)

    if not curve_df.empty:
        for horizon, part in curve_df.groupby("horizon", sort=True):
            fig = _line_fig(
                part,
                x="wavenumber_bin",
                y="spectral_rmse",
                color="variable",
                title=f"Spectral RMSE curve S{int(horizon)}",
                ylabel="spectral_rmse",
            )
            name = f"spectral_curve_S{int(horizon)}"
            save_local_plot(fig, dirs["plots"] / f"{stem}_{name}.png")
            plotly_fig = _plotly_line(part, x="wavenumber_bin", y="spectral_rmse", color="variable", title=f"Spectral RMSE curve S{int(horizon)}")
            result["plots"].append(name)
            _maybe_log_plot(
                run,
                config,
                f"plots/{phase}/spectral_curve/S{int(horizon)}",
                plotly_fig=plotly_fig,
                matplotlib_fig=fig,
                step=step,
                logger=logger,
            )
            _close_fig(fig)
        for variable, part in curve_df.groupby("variable", sort=False):
            fig = _line_fig(
                part,
                x="wavenumber_bin",
                y="spectral_rmse",
                color="horizon",
                title=f"Spectral RMSE curve {variable}",
                ylabel="spectral_rmse",
            )
            name = f"spectral_curve_{_safe_name(str(variable))}"
            save_local_plot(fig, dirs["plots"] / f"{stem}_{name}.png")
            plotly_fig = _plotly_line(part, x="wavenumber_bin", y="spectral_rmse", color="horizon", title=f"Spectral RMSE curve {variable}")
            result["plots"].append(name)
            _maybe_log_plot(
                run,
                config,
                f"plots/{phase}/spectral_curve/{_safe_name(str(variable))}",
                plotly_fig=plotly_fig,
                matplotlib_fig=fig,
                step=step,
                logger=logger,
            )
            _close_fig(fig)
    result["warnings"].append("Skipped normalized spectral error: target spectral power unavailable")
    return result


def log_histograms(
    hist_data: dict[str, dict[str, np.ndarray]],
    *,
    config: dict[str, Any],
    run: Any,
    output_dir: Path,
    step: int,
    phase: str,
    epoch: int,
    logger: Any = logging,
) -> dict[str, Any]:
    result = {"histograms": [], "warnings": []}
    dirs = _ensure_dirs(output_dir)
    stem = f"epoch_{int(epoch):04d}_{_safe_name(phase)}"
    wandb_hist_enabled = wandb_available_and_enabled(config, run) and bool(_wandb_cfg(config).get("log_histograms", False))
    for family, values_by_name in (hist_data or {}).items():
        for name, values in values_by_name.items():
            arr = np.asarray(values, dtype=np.float64).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            safe = _safe_name(str(name))
            np.save(dirs["histograms"] / f"{stem}_{family}_{safe}.npy", arr)
            fig = _hist_fig(arr, title=f"{family} histogram: {name}", xlabel=family)
            save_local_plot(fig, dirs["histograms"] / f"{stem}_{family}_{safe}.png")
            result["histograms"].append(f"{family}:{name}")
            if wandb_hist_enabled and wandb is not None and run is not None:
                key = f"hist/{phase}/{family}/{safe}"
                try:
                    _wandb_log(run, {key: wandb.Histogram(arr)}, step=step, logger=logger)
                except Exception as exc:
                    logger.warning("W&B histogram logging skipped for %s: %s", key, exc)
            _close_fig(fig)
    if not result["histograms"] and not wandb_hist_enabled:
        result["warnings"].append("Skipped W&B histograms: disabled or W&B unavailable")
    return result


def log_spatial_maps(
    map_data: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    run: Any,
    output_dir: Path,
    step: int,
    phase: str,
    epoch: int,
    logger: Any = logging,
) -> dict[str, Any]:
    result = {"images": [], "warnings": []}
    if not map_data:
        result["warnings"].append("Skipped spatial maps: no map data")
        return result
    dirs = _ensure_dirs(output_dir)
    stem = f"epoch_{int(epoch):04d}_{_safe_name(phase)}"
    wandb_image_enabled = wandb_available_and_enabled(config, run) and bool(_wandb_cfg(config).get("log_images", False))
    for item in map_data:
        variable = _safe_name(str(item.get("variable", "var")))
        horizon = int(item.get("horizon", 0))
        kind = _safe_name(str(item.get("kind", "map")))
        arr = np.asarray(item.get("array"), dtype=np.float64)
        if arr.ndim != 2:
            result["warnings"].append(f"Skipped spatial map {kind}/{variable}/S{horizon}: expected 2D array")
            continue
        cmap = "viridis" if kind == "rmse" else "coolwarm"
        fig = _map_fig(arr, title=f"{phase} {kind} {variable} S{horizon}", cmap=cmap)
        path = dirs["maps"] / f"{stem}_{kind}_{variable}_S{horizon}.png"
        save_local_plot(fig, path)
        result["images"].append(f"{kind}:{variable}:S{horizon}")
        if wandb_image_enabled:
            key = f"maps/{phase}/{kind}/{variable}/S{horizon}"
            _log_matplotlib_image(run, key, fig, step=step, logger=logger)
        _close_fig(fig)
    if not result["images"] and not wandb_image_enabled:
        result["warnings"].append("Skipped W&B spatial images: disabled or W&B unavailable")
    return result


def log_rollout_heatmap(
    rollout_history_df: Any,
    *,
    config: dict[str, Any],
    run: Any,
    output_dir: Path,
    step: int,
    phase: str,
    epoch: int,
    logger: Any = logging,
) -> dict[str, Any]:
    df = _as_dataframe(rollout_history_df)
    result = {"plots": [], "warnings": []}
    if df.empty or not {"epoch", "horizon", "step", "loss"}.issubset(df.columns):
        result["warnings"].append("Skipped rollout heatmap: no rollout history")
        return result
    phase_df = df[df["phase"].astype(str) == str(phase)] if "phase" in df.columns else df
    if phase_df.empty:
        result["warnings"].append("Skipped rollout heatmap: no rows for phase")
        return result
    rows = []
    for ep, part in phase_df.groupby("epoch", sort=True):
        max_horizon = int(part["horizon"].max())
        rows.append(part[part["horizon"] == max_horizon])
    selected = _as_dataframe([row for part in rows for row in part.to_dict("records")])
    pivot = selected.pivot_table(index="epoch", columns="step", values="loss", aggfunc="mean").sort_index()
    if pivot.empty:
        result["warnings"].append("Skipped rollout heatmap: empty pivot")
        return result
    matrix = pivot.to_numpy(dtype=np.float64)
    dirs = _ensure_dirs(output_dir)
    fig = _heatmap_fig(
        matrix,
        x_labels=[str(int(x)) for x in pivot.columns],
        y_labels=[str(int(x)) for x in pivot.index],
        title=f"{phase} rollout loss by epoch and step",
        cbar_label="loss",
    )
    name = "rollout_loss_heatmap"
    stem = f"epoch_{int(epoch):04d}_{_safe_name(phase)}"
    save_local_plot(fig, dirs["plots"] / f"{stem}_{name}.png")
    result["plots"].append(name)
    _maybe_log_plot(
        run,
        config,
        f"heatmaps/{phase}/rollout_loss_epoch_step",
        plotly_fig=None,
        matplotlib_fig=fig,
        step=step,
        logger=logger,
    )
    _close_fig(fig)
    return result


def log_optimization_plots(
    gradient_df: Any,
    *,
    config: dict[str, Any],
    run: Any,
    output_dir: Path,
    step: int,
    epoch: int,
    logger: Any = logging,
) -> dict[str, Any]:
    df = _as_dataframe(gradient_df)
    result = {"plots": [], "warnings": []}
    if df.empty or not {"epoch", "step", "block_name", "grad_norm"}.issubset(df.columns):
        result["warnings"].append("Skipped optimization plots: no gradient history")
        return result
    dirs = _ensure_dirs(output_dir)
    stem = f"epoch_{int(epoch):04d}_train"
    global_df = df[df["block_name"].astype(str) == "global"]
    if not global_df.empty:
        fig = _line_fig(global_df, x="step", y="grad_norm", color="stage" if "stage" in global_df.columns else None, title="Global gradient norm", ylabel="grad_norm")
        save_local_plot(fig, dirs["plots"] / f"{stem}_global_grad_norm.png")
        plotly_fig = _plotly_line(global_df, x="step", y="grad_norm", color="stage" if "stage" in global_df.columns else None, title="Global gradient norm")
        result["plots"].append("global_grad_norm")
        _maybe_log_plot(run, config, "plots/train/global_grad_norm", plotly_fig=plotly_fig, matplotlib_fig=fig, step=step, logger=logger)
        _close_fig(fig)
    block_df = df[df["block_name"].astype(str) != "global"]
    if not block_df.empty:
        fig = _line_fig(block_df, x="step", y="grad_norm", color="block_name", title="Block gradient norm", ylabel="grad_norm")
        save_local_plot(fig, dirs["plots"] / f"{stem}_block_grad_norm.png")
        plotly_fig = _plotly_line(block_df, x="step", y="grad_norm", color="block_name", title="Block gradient norm")
        result["plots"].append("block_grad_norm")
        _maybe_log_plot(run, config, "plots/train/block_grad_norm", plotly_fig=plotly_fig, matplotlib_fig=fig, step=step, logger=logger)
        _close_fig(fig)
    return result


def log_summary_table(
    summary_row: dict[str, Any],
    *,
    config: dict[str, Any],
    run: Any,
    output_dir: Path,
    step: int,
    epoch: int,
    logger: Any = logging,
) -> dict[str, Any]:
    result = {"tables": [], "warnings": []}
    if not summary_row:
        result["warnings"].append("Skipped summary table: no summary row")
        return result
    df = _as_dataframe([summary_row])
    dirs = _ensure_dirs(output_dir)
    path = dirs["tables"] / f"epoch_{int(epoch):04d}_model_diagnostics_summary.csv"
    if _save_table(df, path):
        result["tables"].append("summary")
    _save_table(df, dirs["tables"] / "model_diagnostics_summary_latest.csv")
    if _maybe_log_table(run, config, "tables/summary/model_diagnostics", df, step=step, logger=logger):
        result["tables"].append("wandb_summary")
    return result
