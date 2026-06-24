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
    config_name = config_name or "smoke_5p625"
    return os.path.abspath(os.path.expanduser(yaml_path)), config_name


def _visible_device_tokens() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [token.strip() for token in raw.split(",") if token.strip()]


def _map_requested_device_to_visible(device: str | None) -> str | None:
    if device is None:
        return None
    requested = str(device).strip()
    lowered = requested.lower()
    if lowered == "":
        return None
    if lowered.isdigit():
        tokens = _visible_device_tokens()
        if not tokens:
            os.environ["CUDA_VISIBLE_DEVICES"] = lowered
            return "cuda:0"
        if lowered in tokens:
            return f"cuda:{tokens.index(lowered)}"
        return f"cuda:{lowered}"
    return requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default="01", type=str)
    parser.add_argument("--yaml_config", default=str(project_root / "configs" / "gnn_5p625.yaml"), type=str)
    parser.add_argument("--config", default="smoke_5p625", type=str)
    parser.add_argument("--config_name", default=None, type=str)
    parser.add_argument("--experiment_name", default=None, type=str)
    parser.add_argument("--resolution_mode", default=None, type=str, help="5p625 or 2p5; overrides YAML resolution_mode.")
    parser.add_argument("--enable_amp", dest="enable_amp", action="store_true", default=None)
    parser.add_argument("--no_enable_amp", dest="enable_amp", action="store_false")
    parser.add_argument("--amp_dtype", default=None, type=str)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=None, type=int)
    parser.add_argument("--num_data_workers", default=None, type=int)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=None)
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true", default=None)
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--prefetch_factor", default=None, type=int)
    parser.add_argument("--max_epochs", default=None, type=int)
    parser.add_argument("--lr_schedule_type", default=None, type=str)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--min_lr", default=None, type=float)
    parser.add_argument("--warmup_epochs", default=None, type=int)
    parser.add_argument("--warmup_start_factor", default=None, type=float)
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help=(
            "Training device override: auto, cpu, cuda, cuda:N, or numeric GPU id. "
            "A bare numeric id is treated as a physical GPU id and sets CUDA_VISIBLE_DEVICES when it is unset."
        ),
    )
    parser.add_argument("--seed", default=777, type=int)
    args = parser.parse_args()
    device_override = _map_requested_device_to_visible(args.device)

    yaml_path, config_name = _resolve_config_args(args)
    params = YParams(yaml_path, config_name, resolution_mode=args.resolution_mode)
    if args.enable_amp is not None:
        params["enable_amp"] = bool(args.enable_amp)
    for arg_name in (
        "amp_dtype",
        "batch_size",
        "gradient_accumulation_steps",
        "num_data_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "max_epochs",
        "lr_schedule_type",
        "lr",
        "min_lr",
        "warmup_epochs",
        "warmup_start_factor",
    ):
        value = getattr(args, arg_name)
        if value is not None:
            params[arg_name] = value
    if device_override is not None:
        params["device"] = device_override
    from src.trainer import Trainer, set_seed

    set_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_rank = int(os.environ.get("RANK", "0"))

    mode = str(params.resolution_mode)
    if args.experiment_name:
        run_name = str(args.experiment_name)
    else:
        run_base = config_name if mode in config_name else f"{config_name}_{mode}"
        run_name = f"{run_base}_{args.run_num}"
    exp_dir = os.path.abspath(os.path.join(params.exp_dir, run_name))
    os.makedirs(exp_dir, exist_ok=True)
    params["name"] = run_name
    params["experiment_dir"] = exp_dir
    params["checkpoint_path"] = os.path.join(exp_dir, "ckpt.tar")
    params["last_checkpoint_path"] = os.path.join(exp_dir, "last_ckpt.tar")
    params["best_checkpoint_path"] = os.path.join(exp_dir, "best_ckpt.tar")
    setup_logging(rank=world_rank, log_file=os.path.join(exp_dir, "out.log"))
    if args.device is not None:
        logging.info("CLI device request: %s -> trainer device setting: %s", args.device, params.device)
        logging.info("CUDA_VISIBLE_DEVICES: %s", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    params.log()

    trainer = Trainer(params, world_rank=world_rank, local_rank=local_rank)
    trainer.train()
    logging.info("DONE rank %d", world_rank)


if __name__ == "__main__":
    main()
