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
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128_fixed_orog_random_rollout.yaml":
                    config_name = "raw_l3_hidden128_fixed_orog_random_rollout"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128_fixed_orog_random_rollout_l0mlp.yaml":
                    config_name = "raw_l3_hidden128_fixed_orog_random_rollout_l0mlp"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128_fixed_orog_uniform10_then_S10.yaml":
                    config_name = "raw_l3_hidden128_fixed_orog_uniform10_then_S10"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden160.yaml":
                    config_name = "raw_l3_hidden160"
                elif Path(args.config).name == "weather_dual_resolution_l3_orog_tisr_fixed.yaml":
                    config_name = "raw_l3_orog_tisr_fixed"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128_orog_tisr_fixed.yaml":
                    config_name = "raw_l3_hidden128_orog_tisr_fixed"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden160_orog_tisr_fixed.yaml":
                    config_name = "raw_l3_hidden160_orog_tisr_fixed"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128_dense_l3k24_fixed_orog.yaml":
                    config_name = "raw_l3_hidden128_dense_l3k24_fixed_orog"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128_scalar_gated_skip_fixed_orog.yaml":
                    config_name = "raw_l3_hidden128_scalar_gated_skip_fixed_orog"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128_scalar_gated_pooling_fixed_orog.yaml":
                    config_name = "raw_l3_hidden128_scalar_gated_pooling_fixed_orog"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128_lead_conditioned_fixed_orog.yaml":
                    config_name = "raw_l3_hidden128_lead_conditioned_fixed_orog"
                elif Path(args.config).name == "weather_dual_resolution_l4_ratio15_hidden128_fixed_orog.yaml":
                    config_name = "raw_l4_ratio15_hidden128_fixed_orog"
                elif Path(args.config).name == "weather_dual_resolution_l4_hidden128_72_36_24_18_9_fixed_orog.yaml":
                    config_name = "raw_l4_hidden128_72_36_24_18_9_fixed_orog"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128_spectral_loss_fixed_orog.yaml":
                    config_name = "raw_l3_hidden128_spectral_loss_fixed_orog"
                elif Path(args.config).name == "weather_dual_resolution_l3_hidden128_spectral_loss_w001_fixed_orog.yaml":
                    config_name = "raw_l3_hidden128_spectral_loss_w001_fixed_orog"
                elif Path(args.config).name == "weather_dual_resolution_1p5_l3_hidden128_orog_tisr_fixed.yaml":
                    config_name = "raw_1p5_l3_hidden128_orog_tisr_fixed"
                elif Path(args.config).name == "weather_dual_resolution_1p5_l4_hidden128_orog_tisr_fixed.yaml":
                    config_name = "raw_1p5_l4_hidden128_orog_tisr_fixed"
                elif Path(args.config).name == "weather_dual_resolution_1p5_l4_hidden160_orog_tisr_fixed.yaml":
                    config_name = "raw_1p5_l4_hidden160_orog_tisr_fixed"
        elif config_name is None:
            config_name = args.config
    yaml_path = yaml_path or str(project_root / "configs" / "gnn_5p625.yaml")
    config_name = config_name or "smoke_5p625"
    return os.path.abspath(os.path.expanduser(yaml_path)), config_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default="01", type=str)
    parser.add_argument("--yaml_config", default=str(project_root / "configs" / "gnn_5p625.yaml"), type=str)
    parser.add_argument("--config", default="smoke_5p625", type=str)
    parser.add_argument("--config_name", default=None, type=str)
    parser.add_argument("--experiment_name", default=None, type=str)
    parser.add_argument(
        "--exp_dir",
        default=None,
        type=str,
        help="Base output directory for experiments; overrides the YAML exp_dir "
        "(e.g. 'runs' to keep new experiments separate from the old experiments/ tree).",
    )
    parser.add_argument("--resolution_mode", default=None, type=str, help="5p625, 2p5, or 1p5; overrides YAML resolution_mode.")
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
    parser.add_argument("--max_train_batches", default=None, type=int)
    parser.add_argument("--max_valid_batches", default=None, type=int)
    parser.add_argument("--lr_schedule_type", default=None, type=str)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--min_lr", default=None, type=float)
    parser.add_argument("--warmup_epochs", default=None, type=int)
    parser.add_argument("--warmup_start_factor", default=None, type=float)
    parser.add_argument("--init_from_checkpoint_allow_partial", action="store_true")
    parser.add_argument(
        "--init_from_checkpoint",
        default=None,
        type=str,
        help=(
            "Warm-start model weights from this checkpoint path, then start a fresh run "
            "(new optimizer, LR schedule, and epoch counter). Unlike --resume, the training "
            "regime may differ from the checkpoint (e.g. continue a finished S1 run through a "
            "rollout curriculum). Ignored if an in-progress run checkpoint is resumed."
        ),
    )
    parser.add_argument(
        "--init_from_checkpoint_strict",
        dest="init_from_checkpoint_strict",
        action="store_true",
        default=None,
        help="Require the checkpoint architecture/graph to match exactly (default).",
    )
    parser.add_argument(
        "--no_init_from_checkpoint_strict",
        dest="init_from_checkpoint_strict",
        action="store_false",
        help="Load only tensors whose names+shapes match; leave the rest freshly initialized.",
    )
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
        "max_train_batches",
        "max_valid_batches",
        "lr_schedule_type",
        "lr",
        "min_lr",
        "warmup_epochs",
        "warmup_start_factor",
    ):
        value = getattr(args, arg_name)
        if value is not None:
            params[arg_name] = value
    if args.exp_dir is not None:
        params["exp_dir"] = args.exp_dir
    if device_override is not None:
        params["device"] = device_override
    if args.init_from_checkpoint_allow_partial:
        params["init_from_checkpoint_allow_partial"] = True
    if args.init_from_checkpoint is not None:
        params["init_from_checkpoint"] = args.init_from_checkpoint
    if args.init_from_checkpoint_strict is not None:
        params["init_from_checkpoint_strict"] = bool(args.init_from_checkpoint_strict)
    from src.trainer import Trainer, set_seed

    set_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_rank = int(os.environ.get("RANK", "0"))

    mode = str(params.resolution_mode)
    if args.experiment_name:
        run_name = str(args.experiment_name)
    elif params.get("experiment_name", None):
        run_name = str(params.experiment_name)
    else:
        run_base = config_name if mode in config_name else f"{config_name}_{mode}"
        run_name = f"{run_base}_{args.run_num}"
    # Guard the self-nesting bug (A1.2): if exp_dir already points AT the run dir
    # (a config baked exp_dir=runs/<name> rather than just runs/), don't re-append
    # run_name — that produced empty runs/<name>/<name>/ dirs. Normal paths
    # (exp_dir=runs or experiments) are unaffected: their basename != run_name.
    exp_dir_base = os.path.abspath(params.exp_dir)
    if os.path.basename(exp_dir_base) == run_name:
        exp_dir = exp_dir_base
    else:
        exp_dir = os.path.join(exp_dir_base, run_name)
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
