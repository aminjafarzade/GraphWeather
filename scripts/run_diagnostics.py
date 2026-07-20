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
from src.device import _map_requested_device_to_visible
from src.trainer import Trainer, set_seed


def _looks_like_yaml(value: str) -> bool:
    return value.endswith((".yaml", ".yml")) or Path(value).expanduser().exists()


def _resolve_config_args(config: str, config_name: str | None) -> tuple[str, str]:
    config_path = Path(config).expanduser()
    yaml_path = config
    name = config_name

    if config_path.is_dir():
        if config_name is None:
            raise ValueError(f"--config points to a directory, so --config-name must name a YAML file in {config_path}.")
        candidate = Path(config_name)
        if candidate.suffix not in {".yaml", ".yml"}:
            raise ValueError(
                "--config points to a directory; pass a YAML filename as --config-name, "
                "or pass the full YAML file path to --config."
            )
        yaml_path = str(config_path / candidate.name)
        name = None
    elif not _looks_like_yaml(config):
        yaml_path = str(project_root / "configs" / "gnn_5p625.yaml")
        name = config

    if name is None:
        with open(Path(yaml_path).expanduser(), "r", encoding="utf-8") as f:
            import yaml

            root = yaml.safe_load(f)
        if not isinstance(root, dict) or not root:
            raise ValueError(f"No config roots found in {yaml_path}")
        if len(root) != 1:
            available = ", ".join(sorted(root.keys()))
            raise ValueError(f"--config-name is required for {yaml_path}. Available: {available}")
        name = next(iter(root))
    return os.path.abspath(os.path.expanduser(yaml_path)), str(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-training GraphWeather diagnostics from a checkpoint.")
    parser.add_argument("--config", required=True, type=str, help="YAML config path or config name.")
    parser.add_argument("--config-name", default=None, type=str, help="Root key inside the YAML config.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Checkpoint path to load.")
    parser.add_argument("--split", default="valid", choices=["valid", "test"], help="Data split to evaluate. Test falls back to valid loader if no test loader exists.")
    parser.add_argument("--output", default="diagnostics/full_eval", type=str, help="Local diagnostics output directory.")
    parser.add_argument("--resolution_mode", default=None, type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--seed", default=777, type=int)
    parser.add_argument("--max_full_diag_batches", default=None, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--num_data_workers", default=None, type=int)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=None)
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true", default=None)
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--disable_wandb", action="store_true")
    args = parser.parse_args()

    device_override = _map_requested_device_to_visible(args.device)
    yaml_path, config_name = _resolve_config_args(args.config, args.config_name)
    params = YParams(yaml_path, config_name, resolution_mode=args.resolution_mode)
    if device_override is not None:
        params["device"] = device_override
    for arg_name in ("batch_size", "num_data_workers", "pin_memory", "persistent_workers"):
        value = getattr(args, arg_name)
        if value is not None:
            params[arg_name] = value
    params["resume"] = False
    params["checkpoint_path"] = os.path.abspath(os.path.expanduser(args.checkpoint))
    params["name"] = str(params.get("name", params.get("experiment_name", config_name)))
    experiment_dir = params.get("experiment_dir", None) or params.get("exp_dir", None) or str(project_root / "diagnostics_eval")
    params["experiment_dir"] = os.path.abspath(os.path.expanduser(str(experiment_dir)))
    params["last_checkpoint_path"] = os.path.join(params["experiment_dir"], "last_ckpt.tar")
    params["best_checkpoint_path"] = os.path.join(params["experiment_dir"], "best_ckpt.tar")
    diagnostics = dict(params.get("diagnostics", {}) or {})
    diagnostics["enabled"] = True
    diagnostics["run_after_training"] = False
    diagnostics["output_dir"] = os.path.dirname(os.path.abspath(os.path.expanduser(args.output))) or "."
    if args.max_full_diag_batches is not None:
        diagnostics["max_full_diag_batches"] = int(args.max_full_diag_batches)
    wandb_cfg = dict(diagnostics.get("wandb", {}) or {})
    if args.disable_wandb:
        wandb_cfg["enabled"] = False
    diagnostics["wandb"] = wandb_cfg
    params["diagnostics"] = diagnostics

    output_dir = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(log_file=os.path.join(output_dir, "run_diagnostics.log"))
    set_seed(args.seed)
    logging.info("Diagnostics config: %s:%s", yaml_path, config_name)
    logging.info("Checkpoint: %s", params["checkpoint_path"])
    logging.info("Output: %s", output_dir)

    trainer = Trainer(params)
    trainer.restore_checkpoint(params["checkpoint_path"])
    trainer.diagnostics_manager.run_full_post_training_diagnostics(
        trainer=trainer,
        split=args.split,
        output_dir=output_dir,
    )
    logging.info("Diagnostics complete.")


if __name__ == "__main__":
    main()
