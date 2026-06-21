from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.config import YParams, setup_logging
from src.evaluator import run_evaluation_from_params


def _parse_forecast_steps(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = []
    for chunk in value.replace(",", " ").split():
        if chunk.strip():
            parts.append(int(chunk))
    return parts or None


def _parse_string_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [chunk.strip() for chunk in value.replace(",", " ").split() if chunk.strip()]
    return parts or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default=str(project_root / "configs" / "gnn_5p625.yaml"), type=str)
    parser.add_argument("--config", default="raw_5p625", type=str)
    parser.add_argument("--checkpoint", dest="checkpoint_path", default=None, type=str)
    parser.add_argument("--checkpoint_path", dest="checkpoint_path", default=None, type=str)
    parser.add_argument("--experiment_dir", default=None, type=str)
    parser.add_argument("--test_dataset_path", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--forecast_steps", default=None, type=str, help="Comma or space separated horizons, e.g. '1,2,4,8,10'")
    parser.add_argument("--eval_fixed_rollout_steps", default=None, type=int)
    parser.add_argument("--compare_stage_checkpoints", action="store_true")
    parser.add_argument("--n_initial_conditions", default=None, type=int)
    parser.add_argument("--start_timestep", default=None, type=int)
    parser.add_argument("--ic_stride", default=None, type=int)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--climatology_path", default=None, type=str)
    parser.add_argument("--compute_climatology", action="store_true")
    parser.add_argument("--no_compute_climatology", action="store_true")
    parser.add_argument("--plot_variables", default=None, type=str, help="Comma or space separated variable names/indices, e.g. 't2m,z500,60'")
    parser.add_argument("--plot_format", default=None, type=str, help="Plot image format, default from config is png")
    parser.add_argument("--plot_persistence", action="store_true")
    parser.add_argument("--no_plot_persistence", action="store_true")
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    if args.checkpoint_path:
        params["eval_checkpoint_path"] = args.checkpoint_path
    if args.experiment_dir:
        params["experiment_dir"] = os.path.abspath(args.experiment_dir)
    if args.test_dataset_path:
        params["test_dataset_path"] = args.test_dataset_path
    if args.output_dir:
        params["eval_output_dir"] = args.output_dir
    forecast_steps = _parse_forecast_steps(args.forecast_steps)
    if forecast_steps is not None:
        params["eval_forecast_steps"] = forecast_steps
        if args.eval_fixed_rollout_steps is None:
            params["eval_fixed_rollout_steps"] = max(forecast_steps)
    if args.eval_fixed_rollout_steps is not None:
        params["eval_fixed_rollout_steps"] = int(args.eval_fixed_rollout_steps)
    if args.compare_stage_checkpoints:
        params["eval_compare_stage_checkpoints"] = True
    if args.n_initial_conditions is not None:
        params["n_initial_conditions"] = args.n_initial_conditions
    if args.start_timestep is not None:
        params["eval_start_timestep"] = args.start_timestep
    if args.ic_stride is not None:
        params["eval_ic_stride"] = args.ic_stride
    if args.device is not None:
        params["eval_device"] = args.device
    if args.climatology_path is not None:
        params["climatology_path"] = args.climatology_path
    if args.compute_climatology:
        params["compute_climatology"] = True
    if args.no_compute_climatology:
        params["compute_climatology"] = False
    plot_variables = _parse_string_list(args.plot_variables)
    if plot_variables is not None:
        params["eval_plot_variables"] = plot_variables
    if args.plot_format is not None:
        params["eval_plot_format"] = args.plot_format
    if args.plot_persistence:
        params["plot_persistence"] = True
    if args.no_plot_persistence:
        params["plot_persistence"] = False

    output_dir = params.get("eval_output_dir", None) or os.path.join(
        params.get("experiment_dir", params.get("exp_dir", ".")),
        "evaluation",
    )
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(rank=0, log_file=os.path.join(output_dir, "evaluation.log"))
    params.log()
    run_evaluation_from_params(params, logger=logging)


if __name__ == "__main__":
    main()
