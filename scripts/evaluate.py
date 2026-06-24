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


def _looks_like_yaml(value: str) -> bool:
    return value.endswith((".yaml", ".yml")) or Path(value).expanduser().exists()


def _resolve_config_args(args: argparse.Namespace) -> tuple[str, str]:
    yaml_path = args.yaml_config
    config_name = args.config_name
    if args.config:
        if _looks_like_yaml(args.config):
            yaml_path = args.config
            if config_name is None and Path(args.config).name == "weather_dual_resolution.yaml":
                config_name = "raw"
        elif config_name is None:
            config_name = args.config
    yaml_path = yaml_path or str(project_root / "configs" / "gnn_5p625.yaml")
    config_name = config_name or "raw_5p625"
    return os.path.abspath(os.path.expanduser(yaml_path)), config_name


def _parse_forecast_steps(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = []
    for chunk in value.replace(",", " ").split():
        if chunk.strip():
            parts.append(int(chunk))
    return parts or None


def _parse_string_list(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    values = value if isinstance(value, list) else [value]
    parts: list[str] = []
    for item in values:
        parts.extend(chunk.strip() for chunk in str(item).replace(",", " ").split() if chunk.strip())
    return parts or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default=str(project_root / "configs" / "gnn_5p625.yaml"), type=str)
    parser.add_argument("--config", default="raw_5p625", type=str)
    parser.add_argument("--config_name", default=None, type=str)
    parser.add_argument("--resolution_mode", default=None, type=str, help="5p625 or 2p5; overrides YAML resolution_mode.")
    parser.add_argument("--checkpoint", dest="checkpoint_path", default=None, type=str)
    parser.add_argument("--checkpoint_path", dest="checkpoint_path", default=None, type=str)
    parser.add_argument("--experiment_dir", default=None, type=str)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--test_dataset_path", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--forecast_steps", default=None, type=str, help="Comma or space separated horizons, e.g. '1,2,4,8,10'")
    parser.add_argument("--fixed_rollout_steps", dest="eval_fixed_rollout_steps", default=None, type=int)
    parser.add_argument("--eval_fixed_rollout_steps", default=None, type=int)
    parser.add_argument("--rollout_steps", default=None, type=int, help="Alias for --eval_fixed_rollout_steps.")
    parser.add_argument("--compare_stage_checkpoints", action="store_true")
    parser.add_argument("--selection", default="stride", choices=["first_n", "stride", "all"])
    parser.add_argument("--n_initial_conditions", default=None, type=int)
    parser.add_argument("--max_initial_conditions", default=None, type=int)
    parser.add_argument("--start_timestep", default=None, type=int)
    parser.add_argument("--ic_stride", default=None, type=int)
    parser.add_argument("--stride", default=None, type=int, help="Alias for --ic_stride.")
    parser.add_argument("--start_offset", default=None, type=int)
    parser.add_argument("--start_date", default=None, type=str)
    parser.add_argument("--end_date", default=None, type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--climatology_path", default=None, type=str)
    parser.add_argument("--climatology_chunk_size", default=None, type=int)
    parser.add_argument("--build_climatology_if_missing", action="store_true")
    parser.add_argument("--compute_climatology", action="store_true")
    parser.add_argument("--no_compute_climatology", action="store_true")
    parser.add_argument("--plot_variables", nargs="*", default=None, help="Comma or space separated variable names/indices, e.g. 't2m,z500,60'")
    parser.add_argument("--variables", nargs="*", default=None, help="Alias for --plot_variables.")
    parser.add_argument("--plot_format", default=None, type=str, help="Plot image format, default from config is png")
    parser.add_argument("--plot_persistence", dest="plot_persistence", action="store_true", default=None)
    parser.add_argument("--include_persistence", dest="plot_persistence", action="store_true")
    parser.add_argument("--no_plot_persistence", dest="plot_persistence", action="store_false")
    parser.add_argument("--bootstrap_samples", default=None, type=int)
    parser.add_argument("--confidence_level", default=None, type=float)
    parser.add_argument("--bootstrap_seed", default=None, type=int)
    parser.add_argument("--plot_confidence_intervals", action="store_true")
    parser.add_argument("--external_baseline_csv", nargs="*", default=None)
    parser.add_argument("--external_baseline_label", nargs="*", default=None)
    parser.add_argument("--external_baseline_params_m", nargs="*", default=None, type=float)
    args = parser.parse_args()

    yaml_path, config_name = _resolve_config_args(args)
    params = YParams(yaml_path, config_name, resolution_mode=args.resolution_mode)

    params["eval_split"] = args.split
    if args.checkpoint_path:
        params["eval_checkpoint_path"] = args.checkpoint_path
    if args.experiment_dir:
        params["experiment_dir"] = os.path.abspath(args.experiment_dir)
    if args.test_dataset_path:
        params["test_dataset_path"] = args.test_dataset_path
    else:
        if args.split == "train":
            params["test_dataset_path"] = params.train_data_path
        elif args.split == "valid":
            params["test_dataset_path"] = params.valid_data_path
        else:
            params["test_dataset_path"] = params.get("test_dataset_path", params.valid_data_path)
    if args.output_dir:
        params["eval_output_dir"] = args.output_dir
    forecast_steps = _parse_forecast_steps(args.forecast_steps)
    if forecast_steps is not None:
        params["eval_forecast_steps"] = forecast_steps
        if args.eval_fixed_rollout_steps is None:
            params["eval_fixed_rollout_steps"] = max(forecast_steps)
    fixed_steps = args.eval_fixed_rollout_steps if args.eval_fixed_rollout_steps is not None else args.rollout_steps
    if fixed_steps is not None:
        params["eval_fixed_rollout_steps"] = int(fixed_steps)
    if args.compare_stage_checkpoints:
        params["eval_compare_stage_checkpoints"] = True
    params["eval_selection"] = args.selection
    if args.n_initial_conditions is not None:
        params["n_initial_conditions"] = args.n_initial_conditions
    if args.max_initial_conditions is not None:
        params["max_initial_conditions"] = args.max_initial_conditions
    if args.start_timestep is not None:
        params["eval_start_timestep"] = args.start_timestep
    stride = args.ic_stride if args.ic_stride is not None else args.stride
    if stride is not None:
        params["eval_ic_stride"] = stride
    elif args.selection == "stride":
        params["eval_ic_stride"] = 7
    if args.start_offset is not None:
        params["eval_start_offset"] = args.start_offset
    if args.start_date is not None:
        params["eval_start_date"] = args.start_date
    if args.end_date is not None:
        params["eval_end_date"] = args.end_date
    if args.device is not None:
        params["eval_device"] = args.device
    if args.climatology_path is not None:
        params["climatology_path"] = args.climatology_path
    if args.climatology_chunk_size is not None:
        params["eval_climatology_chunk_size"] = int(args.climatology_chunk_size)
    if args.compute_climatology:
        params["compute_climatology"] = True
    if args.build_climatology_if_missing:
        params["build_climatology_if_missing"] = True
        params["compute_climatology"] = True
    if args.no_compute_climatology:
        params["compute_climatology"] = False
    plot_variables = _parse_string_list(args.variables if args.variables is not None else args.plot_variables)
    if plot_variables is None:
        plot_variables = ["z500", "t2m", "t850", "msl"]
    if plot_variables is not None:
        params["eval_plot_variables"] = plot_variables
    if args.plot_format is not None:
        params["eval_plot_format"] = args.plot_format
    if args.plot_persistence is True:
        params["plot_persistence"] = True
    elif args.plot_persistence is False:
        params["plot_persistence"] = False
    if args.bootstrap_samples is not None:
        params["bootstrap_samples"] = args.bootstrap_samples
    if args.confidence_level is not None:
        params["confidence_level"] = args.confidence_level
    if args.bootstrap_seed is not None:
        params["bootstrap_seed"] = args.bootstrap_seed
    if args.plot_confidence_intervals:
        params["plot_confidence_intervals"] = True
    if args.external_baseline_csv:
        params["external_baseline_csv"] = args.external_baseline_csv
    if args.external_baseline_label:
        params["external_baseline_label"] = args.external_baseline_label
    if args.external_baseline_params_m:
        params["external_baseline_params_m"] = args.external_baseline_params_m

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
