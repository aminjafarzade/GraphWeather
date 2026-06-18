from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch

from src.config import YParams, setup_logging
from src.trainer import Trainer, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default="01", type=str)
    parser.add_argument("--yaml_config", default=str(project_root / "configs" / "gnn_5p625.yaml"), type=str)
    parser.add_argument("--config", default="smoke_5p625", type=str)
    parser.add_argument("--enable_amp", action="store_true")
    parser.add_argument("--seed", default=777, type=int)
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params["enable_amp"] = bool(args.enable_amp)
    set_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_rank = int(os.environ.get("RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True

    run_name = f"{args.config}_{args.run_num}"
    exp_dir = os.path.abspath(os.path.join(params.exp_dir, run_name))
    os.makedirs(exp_dir, exist_ok=True)
    params["name"] = run_name
    params["experiment_dir"] = exp_dir
    params["checkpoint_path"] = os.path.join(exp_dir, "ckpt.tar")
    params["best_checkpoint_path"] = os.path.join(exp_dir, "best_ckpt.tar")
    setup_logging(rank=world_rank, log_file=os.path.join(exp_dir, "out.log"))
    params.log()

    trainer = Trainer(params, world_rank=world_rank, local_rank=local_rank)
    trainer.train()
    logging.info("DONE rank %d", world_rank)


if __name__ == "__main__":
    main()

