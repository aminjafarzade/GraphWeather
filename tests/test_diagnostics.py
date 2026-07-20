from __future__ import annotations

import math

import torch

from src.config import normalize_diagnostics_config_dict
from src.diagnostics.metrics import attention_entropy_stats, embedding_stats, gradient_metrics
from src.diagnostics.spectral import spectral_band_summary, spectral_rmse_2d
from src.diagnostics.wandb_plots import log_rollout_tables_and_plots
from src.features import VariableResolver


def test_diagnostics_config_default_disabled() -> None:
    resolved = normalize_diagnostics_config_dict({})
    assert resolved["diagnostics"] == {"enabled": False}


def test_diagnostics_visualization_defaults_are_safe() -> None:
    resolved = normalize_diagnostics_config_dict({"diagnostics": {"enabled": True}})
    diagnostics = resolved["diagnostics"]
    assert diagnostics["wandb"]["log_scalars"] is True
    assert diagnostics["wandb"]["log_tables"] is True
    assert diagnostics["wandb"]["log_plots"] is False
    assert diagnostics["wandb"]["log_images"] is False
    assert diagnostics["wandb"]["log_histograms"] is False
    assert diagnostics["plots"]["enabled"] is False
    assert diagnostics["plots"]["map_horizons"] == [1, 4, 10]


def test_rollout_plot_saves_local_files_without_wandb(tmp_path) -> None:
    rows = [
        {"epoch": 3, "phase": "valid", "horizon": 1, "step": 1, "loss": 0.4},
        {"epoch": 3, "phase": "valid", "horizon": 2, "step": 1, "loss": 0.4},
        {"epoch": 3, "phase": "valid", "horizon": 2, "step": 2, "loss": 0.7},
    ]
    cfg = {
        "wandb": {"enabled": False, "log_tables": True, "log_plots": True},
        "plots": {"enabled": True},
    }
    result = log_rollout_tables_and_plots(
        rows,
        config=cfg,
        run=None,
        output_dir=tmp_path,
        step=3,
        phase="valid",
        epoch=3,
    )
    assert "rollout" in result["tables"]
    assert (tmp_path / "tables" / "epoch_0003_valid_rollout_curve.csv").exists()
    assert (tmp_path / "plots" / "epoch_0003_valid_rollout_step_loss.png").exists()
    assert (tmp_path / "plots" / "epoch_0003_valid_final_loss_by_horizon.png").exists()


def test_variable_resolver_supports_u10_v10_aliases() -> None:
    resolver = VariableResolver(
        {},
        channel_names=["10u", "10v", "temperature_2m"],
        out_channels=[0, 1, 2],
    )
    assert resolver.resolve("u10").local_index == 0
    assert resolver.resolve("v_component_of_wind_10m").local_index == 1
    assert resolver.resolve("t2m").local_index == 2


def test_embedding_stats_handle_graph_and_grid_tensors() -> None:
    graph = torch.randn(2, 16, 8)
    grid = torch.randn(2, 8, 4, 4)
    for tensor in (graph, grid):
        stats = embedding_stats(tensor, embedding_sample_nodes=16, pairwise_sample_nodes=8)
        assert stats["embedding_variance"] > 0.0
        assert math.isfinite(stats["cosine_mean"])
        assert math.isfinite(stats["effective_rank_norm"])


def test_fixed_k_attention_uniform_entropy() -> None:
    attn = torch.full((2, 5, 4, 3), 0.25)
    stats = attention_entropy_stats(attn)
    assert abs(stats["entropy_norm"] - 1.0) < 1.0e-5
    assert abs(stats["max_weight_mean"] - 0.25) < 1.0e-6
    assert stats["degree_mean"] == 4.0


def test_spectral_rmse_produces_bands() -> None:
    pred = torch.zeros(2, 3, 8, 8)
    target = torch.ones(2, 3, 8, 8)
    curves = spectral_rmse_2d(pred, target, [1])
    assert 1 in curves
    bands = spectral_band_summary(curves[1])
    assert set(bands) == {"low_k_rmse", "mid_k_rmse", "high_k_rmse"}
    assert bands["low_k_rmse"] >= 0.0


def test_gradient_metrics_use_existing_backward() -> None:
    model = torch.nn.Linear(3, 2)
    loss = model(torch.ones(4, 3)).sum()
    loss.backward()
    metrics = gradient_metrics(model)
    assert metrics["global"] > 0.0
    assert metrics["weight"] > 0.0
