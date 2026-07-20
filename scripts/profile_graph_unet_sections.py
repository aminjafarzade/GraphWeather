from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.config import YParams, setup_logging
from src.trainer import Trainer, set_seed


SECTION_NAMES = (
    "embedding",
    "encoder",
    "L0 processing",
    "pool L0->L1",
    "L1 processing",
    "pool L1->L2",
    "L2 processing",
    "pool L2->L3",
    "L3 processing",
    "unpool/fuse L3->L2",
    "L2 refinement after L3",
    "unpool/fuse L2->L1",
    "L1 refinement",
    "unpool/fuse L1->L0",
    "L0 refinement",
    "decoder",
    "head",
    "loss",
    "backward",
    "optimizer step",
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


def _configure_for_profile(params: Any, args: argparse.Namespace) -> None:
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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_section(totals: dict[str, float], device: torch.device, name: str, fn):
    _sync(device)
    start = time.perf_counter()
    result = fn()
    _sync(device)
    totals[name] += time.perf_counter() - start
    return result


def _profile_forward_loss(
    trainer: Trainer,
    inp: torch.Tensor,
    target: torch.Tensor,
    metadata: dict[str, Any],
    rollout_steps: int,
    totals: dict[str, float],
) -> torch.Tensor:
    model = trainer.model
    processor = model.processor
    target_seq = trainer._target_sequence(target)
    previous, current = model.adapter.extract_two_steps(inp)
    total = torch.zeros((), device=inp.device, dtype=inp.dtype)

    for step in range(int(rollout_steps)):
        gt = target_seq[:, step]
        aux = trainer._build_aux_for_step(current, target_seq, metadata, step)

        def embed_step():
            node_x, current_state_nodes = model.adapter.to_node_features_from_steps(previous, current)
            if aux is not None:
                node_x = torch.cat([node_x, aux.to(device=node_x.device, dtype=node_x.dtype)], dim=-1)
            return model.embed(node_x), current_state_nodes

        h, current_state_nodes = _elapsed_section(totals, trainer.device, "embedding", embed_step)
        for block in model.encoder:
            h = _elapsed_section(totals, trainer.device, "encoder", lambda block=block, h=h: block(h, model.graph.L0))

        for block in processor.l0_blocks:
            h = _elapsed_section(totals, trainer.device, "L0 processing", lambda block=block, h=h: block(h, model.graph.L0))
        skip0 = h

        h = _elapsed_section(
            totals,
            trainer.device,
            "pool L0->L1",
            lambda h=h: processor.pool01(h, model.graph.pool_L0_to_L1, model.graph.L1.num_nodes),
        )
        for block in processor.l1_blocks:
            h = _elapsed_section(totals, trainer.device, "L1 processing", lambda block=block, h=h: block(h, model.graph.L1))
        skip1 = h

        h = _elapsed_section(
            totals,
            trainer.device,
            "pool L1->L2",
            lambda h=h: processor.pool12(h, model.graph.pool_L1_to_L2, model.graph.L2.num_nodes),
        )
        for block in processor.l2_blocks:
            h = _elapsed_section(totals, trainer.device, "L2 processing", lambda block=block, h=h: block(h, model.graph.L2))

        if processor.use_l3:
            skip2 = h
            h = _elapsed_section(
                totals,
                trainer.device,
                "pool L2->L3",
                lambda h=h: processor.pool23(h, model.graph.pool_L2_to_L3, model.graph.L3.num_nodes),
            )
            for block in processor.l3_blocks:
                h = _elapsed_section(totals, trainer.device, "L3 processing", lambda block=block, h=h: block(h, model.graph.L3))
            h = _elapsed_section(
                totals,
                trainer.device,
                "unpool/fuse L3->L2",
                lambda h=h, skip2=skip2: processor.unpool32(h, model.graph.pool_L2_to_L3, skip2),
            )
            for block in processor.l2_refine_after_l3:
                h = _elapsed_section(
                    totals,
                    trainer.device,
                    "L2 refinement after L3",
                    lambda block=block, h=h: block(h, model.graph.L2),
                )

        h = _elapsed_section(
            totals,
            trainer.device,
            "unpool/fuse L2->L1",
            lambda h=h, skip1=skip1: processor.unpool21(h, model.graph.pool_L1_to_L2, skip1),
        )
        for block in processor.l1_refine:
            h = _elapsed_section(totals, trainer.device, "L1 refinement", lambda block=block, h=h: block(h, model.graph.L1))

        h = _elapsed_section(
            totals,
            trainer.device,
            "unpool/fuse L1->L0",
            lambda h=h, skip0=skip0: processor.unpool10(h, model.graph.pool_L0_to_L1, skip0),
        )
        for block in processor.l0_refine:
            h = _elapsed_section(totals, trainer.device, "L0 refinement", lambda block=block, h=h: block(h, model.graph.L0))

        for block in model.decoder:
            h = _elapsed_section(totals, trainer.device, "decoder", lambda block=block, h=h: block(h, model.graph.L0))

        def head_step():
            delta_nodes = model.head(h)
            delta_nodes = model.denormalize_delta_nodes(delta_nodes)
            pred_nodes = current_state_nodes + delta_nodes
            return model.adapter.nodes_to_grid(pred_nodes)

        pred = _elapsed_section(totals, trainer.device, "head", head_step)
        feature_builder = getattr(trainer, "feature_builder", None)
        if feature_builder is not None:
            pred = feature_builder.apply_overrides(pred, current=current, target_norm=gt)
        loss = _elapsed_section(totals, trainer.device, "loss", lambda pred=pred, gt=gt: trainer._step_loss(pred, gt))
        total = total + loss
        next_step = current.clone()
        next_step[:, : model.output_channels] = pred
        previous, current = current, next_step

    return total / float(rollout_steps)


def _profile_step(trainer: Trainer, batch: Any, rollout_steps: int, totals: dict[str, float]) -> float:
    trainer.optimizer.zero_grad(set_to_none=True)
    inp, target, metadata = trainer._to_device_batch(batch)
    with trainer._autocast_context():
        loss = _profile_forward_loss(trainer, inp, target, metadata, rollout_steps, totals)
    _sync(trainer.device)
    start = time.perf_counter()
    loss.backward()
    _sync(trainer.device)
    totals["backward"] += time.perf_counter() - start
    start = time.perf_counter()
    trainer._optimizer_step()
    _sync(trainer.device)
    totals["optimizer step"] += time.perf_counter() - start
    return float(loss.detach().item())


def _write_outputs(totals: dict[str, float], steps: int, output_dir: Path, losses: list[float]) -> None:
    rows = []
    total_time = sum(totals.values())
    for name in SECTION_NAMES:
        seconds = float(totals.get(name, 0.0))
        rows.append(
            {
                "section": name,
                "total_seconds": seconds,
                "mean_seconds_per_step": seconds / float(max(steps, 1)),
                "percent_profiled_time": 100.0 * seconds / total_time if total_time > 0.0 else 0.0,
            }
        )
    csv_path = output_dir / "section_profile.csv"
    txt_path = output_dir / "section_profile.txt"
    fields = ["section", "total_seconds", "mean_seconds_per_step", "percent_profiled_time"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"profile_steps: {steps}",
        f"mean_loss: {sum(losses) / float(max(len(losses), 1)):.6f}",
        "",
        "\t".join(fields),
    ]
    lines.extend("\t".join(str(row[field]) for field in fields) for row in rows)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(txt_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config_name", default=None)
    parser.add_argument("--resolution_mode", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rollout_steps", type=int, default=1)
    parser.add_argument("--num_warmup_steps", type=int, default=3)
    parser.add_argument("--num_profile_steps", type=int, default=10)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(rank=0, log_file=str(output_dir / "profile_graph_unet_sections.log"))
    set_seed(args.seed)

    yaml_path, config_name = _resolve_config(args.config, args.config_name)
    params = YParams(yaml_path, config_name, resolution_mode=args.resolution_mode)
    _configure_for_profile(params, args)
    trainer = Trainer(params, world_rank=0, local_rank=0)
    trainer.model.train()

    batch = next(iter(trainer.train_data_loader))
    warmup_totals = {name: 0.0 for name in SECTION_NAMES}
    for _ in range(max(0, int(args.num_warmup_steps))):
        _profile_step(trainer, batch, int(args.rollout_steps), warmup_totals)

    totals = {name: 0.0 for name in SECTION_NAMES}
    losses = []
    for _ in range(max(1, int(args.num_profile_steps))):
        losses.append(_profile_step(trainer, batch, int(args.rollout_steps), totals))

    _write_outputs(totals, max(1, int(args.num_profile_steps)), output_dir, losses)


if __name__ == "__main__":
    main()
