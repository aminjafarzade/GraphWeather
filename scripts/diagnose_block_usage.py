from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
import sys
from typing import Any

import torch

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.config import YParams, setup_logging
from src.trainer import Trainer, set_seed


BLOCK_GROUPS = (
    "l0_blocks",
    "l1_blocks",
    "l2_blocks",
    "l3_blocks",
    "l2_refine_after_l3",
    "l1_refine",
    "l0_refine",
)


def _looks_like_yaml(value: str) -> bool:
    return value.endswith((".yaml", ".yml")) or Path(value).expanduser().exists()


def _default_config_name(config_path: str) -> str:
    name = Path(config_path).name
    if name == "weather_dual_resolution.yaml":
        return "raw"
    if name == "weather_dual_resolution_l3.yaml":
        return "raw_l3"
    if name == "weather_dual_resolution_l3_stage_warmup_cosine.yaml":
        return "raw_l3_stage_warmup_cosine"
    if name == "weather_dual_resolution_l3_blocks3.yaml":
        return "raw_l3_blocks3"
    if name == "weather_dual_resolution_l3_heavy_unet.yaml":
        return "raw_l3_heavy_unet"
    if name == "weather_dual_resolution_l3_full_rollout.yaml":
        return "raw_l3_full_rollout"
    if name == "weather_dual_resolution_l3_hidden128.yaml":
        return "raw_l3_hidden128"
    if name == "weather_dual_resolution_l3_hidden160.yaml":
        return "raw_l3_hidden160"
    return "smoke_5p625"


def _resolve_config(config: str, config_name: str | None) -> tuple[str, str]:
    if _looks_like_yaml(config):
        path = os.path.abspath(os.path.expanduser(config))
        return path, config_name or _default_config_name(path)
    return str(project_root / "configs" / "gnn_5p625.yaml"), config_name or config


def _set(params: Any, key: str, value: Any) -> None:
    params[key] = value


def _configure_for_single_batch(params: Any, args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    _set(params, "name", output_dir.name)
    _set(params, "experiment_name", output_dir.name)
    _set(params, "experiment_dir", str(output_dir / "_trainer_startup"))
    _set(params, "checkpoint_path", str(output_dir / "_trainer_startup" / "ckpt.tar"))
    _set(params, "last_checkpoint_path", str(output_dir / "_trainer_startup" / "last_ckpt.tar"))
    _set(params, "best_checkpoint_path", str(output_dir / "_trainer_startup" / "best_ckpt.tar"))
    _set(params, "max_epochs", 1)
    _set(params, "max_train_batches", 1)
    _set(params, "max_valid_batches", 1)
    _set(params, "batch_size", int(args.batch_size))
    _set(params, "gradient_accumulation_steps", 1)
    _set(params, "num_data_workers", 0)
    _set(params, "pin_memory", False)
    _set(params, "persistent_workers", False)
    _set(params, "prefetch_factor", None)
    if args.device is not None:
        _set(params, "device", args.device)
    if str(params.get("device", "auto")).lower() == "cpu":
        _set(params, "enable_amp", False)
    _set(params, "log_cuda_memory", False)
    _set(params, "save_checkpoint", False)
    _set(params, "resume", False)
    _set(params, "load_only_current_rollout", True)
    _set(params, "max_rollout_steps", int(args.rollout_steps))
    _set(params, "rollout_schedule", [int(args.rollout_steps)])
    _set(params, "rollout_stage_epochs", [1])
    _set(params, "valid_rollout_steps", int(args.rollout_steps))
    _set(params, "eval_rollout_steps", [int(args.rollout_steps)])


def _block_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    processor = getattr(model, "processor")
    blocks: list[tuple[str, torch.nn.Module]] = []
    for group_name in BLOCK_GROUPS:
        group = getattr(processor, group_name, None)
        if group is None:
            continue
        for idx, module in enumerate(group):
            blocks.append((f"{group_name}.{idx}", module))
    return blocks


def _param_count(module: torch.nn.Module, requires_grad: bool | None = None) -> int:
    total = 0
    for param in module.parameters():
        if requires_grad is None or bool(param.requires_grad) == bool(requires_grad):
            total += param.numel()
    return int(total)


def _grad_norm(module: torch.nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().to(dtype=torch.float64)
        total += torch.sum(grad * grad).cpu()
    return float(torch.sqrt(total).item())


def _optimizer_rows(trainer: Trainer, model_label: str) -> list[dict[str, Any]]:
    optimizer_param_ids = {
        id(param)
        for group in trainer.optimizer.param_groups
        for param in group.get("params", [])
    }
    rows: list[dict[str, Any]] = []
    for name, param in trainer.model.named_parameters():
        trainable = bool(param.requires_grad)
        in_optimizer = id(param) in optimizer_param_ids
        rows.append(
            {
                "model": model_label,
                "parameter_name": name,
                "num_parameters": int(param.numel()),
                "requires_grad": trainable,
                "in_optimizer": in_optimizer,
                "missing_from_optimizer": trainable and not in_optimizer,
                "unexpectedly_frozen": not trainable,
            }
        )
    return rows


def _write_optimizer_check(rows: list[dict[str, Any]], output_dir: Path) -> None:
    csv_path = output_dir / "optimizer_parameter_check.csv"
    txt_path = output_dir / "optimizer_parameter_check.txt"
    fields = [
        "model",
        "parameter_name",
        "num_parameters",
        "requires_grad",
        "in_optimizer",
        "missing_from_optimizer",
        "unexpectedly_frozen",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    total = sum(int(row["num_parameters"]) for row in rows)
    trainable = sum(int(row["num_parameters"]) for row in rows if row["requires_grad"])
    optimizer = sum(int(row["num_parameters"]) for row in rows if row["in_optimizer"])
    missing = [row for row in rows if row["missing_from_optimizer"]]
    frozen = [row for row in rows if row["unexpectedly_frozen"]]
    lines = [
        f"total_model_parameters: {total}",
        f"trainable_parameters: {trainable}",
        f"optimizer_parameters: {optimizer}",
        f"parameters_missing_from_optimizer: {len(missing)}",
        f"parameters_with_requires_grad_false: {len(frozen)}",
    ]
    if missing:
        lines.append("missing_from_optimizer:")
        lines.extend(f"  {row['parameter_name']}" for row in missing)
    if frozen:
        lines.append("requires_grad_false:")
        lines.extend(f"  {row['parameter_name']}" for row in frozen)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config_name", default=None)
    parser.add_argument("--resolution_mode", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--one_train_batch", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rollout_steps", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(rank=0, log_file=str(output_dir / "diagnose_block_usage.log"))
    set_seed(args.seed)

    yaml_path, config_name = _resolve_config(args.config, args.config_name)
    params = YParams(yaml_path, config_name, resolution_mode=args.resolution_mode)
    _configure_for_single_batch(params, args)

    trainer = Trainer(params, world_rank=0, local_rank=0)
    model_label = Path(yaml_path).stem
    blocks = _block_modules(trainer.model)
    call_counts = {name: 0 for name, _ in blocks}
    handles = []

    def _make_hook(block_name: str):
        def hook(_module, _inputs, _outputs):
            call_counts[block_name] += 1
        return hook

    for name, module in blocks:
        handles.append(module.register_forward_hook(_make_hook(name)))

    if args.one_train_batch:
        trainer.model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        rollout_steps = int(args.rollout_steps)
        data = next(iter(trainer.train_data_loader))
        inp, target, metadata = trainer._to_device_batch(data)
        with trainer._autocast_context():
            loss, _ = trainer._rollout_loss(inp, target, rollout_steps, metadata=metadata)
        loss.backward()
        logging.info("one-batch loss: %.6f", float(loss.detach().item()))
    else:
        trainer.model.eval()
        data = next(iter(trainer.train_data_loader))
        inp, target, metadata = trainer._to_device_batch(data)
        with torch.no_grad():
            with trainer._autocast_context():
                loss, _ = trainer._rollout_loss(inp, target, int(args.rollout_steps), metadata=metadata)
        logging.info("one-batch forward-only loss: %.6f", float(loss.detach().item()))

    for handle in handles:
        handle.remove()

    rows: list[dict[str, Any]] = []
    for name, module in blocks:
        grad_norm = _grad_norm(module)
        rows.append(
            {
                "block_name": name,
                "module_class": module.__class__.__name__,
                "called_forward": call_counts[name] > 0,
                "num_forward_calls": int(call_counts[name]),
                "num_parameters": _param_count(module),
                "requires_grad_parameters": _param_count(module, requires_grad=True),
                "grad_norm_after_backward": grad_norm,
                "has_nonzero_grad": grad_norm > 0.0,
            }
        )

    csv_path = output_dir / "block_usage.csv"
    txt_path = output_dir / "block_usage.txt"
    fields = [
        "block_name",
        "module_class",
        "called_forward",
        "num_forward_calls",
        "num_parameters",
        "requires_grad_parameters",
        "grad_norm_after_backward",
        "has_nonzero_grad",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "\t".join(fields),
        *("\t".join(str(row[field]) for field in fields) for row in rows),
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_optimizer_check(_optimizer_rows(trainer, model_label), output_dir)
    print(txt_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
