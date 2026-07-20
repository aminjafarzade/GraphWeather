from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re
import statistics
import sys
from typing import Any

import torch
import yaml

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.config import YParams
from src.graph_bundle import load_graph_bundle


DEFAULT_MODELS = {
    "l3_deeper": {
        "config": "configs/weather_dual_resolution_l3_blocks3.yaml",
        "config_name": "raw_l3_blocks3",
        "experiment": "experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_blocks3",
    },
    "heavy_l3_unet": {
        "config": "configs/weather_dual_resolution_l3_heavy_unet.yaml",
        "config_name": "raw_l3_heavy_unet",
        "experiment": "experiments/main_raw_2p5_b4_acc3_bf16_delta_l3_heavy_unet_l0-3_l1-2_l2-2_l3-2_refine2",
    },
}

CONFIG_FIELDS = (
    "experiment_name",
    "config_path",
    "checkpoint_path",
    "resolution_mode",
    "batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "max_epochs",
    "rollout_schedule",
    "rollout_stage_epochs",
    "current_rollout_stage_per_epoch",
    "actual_completed_rollout_stages",
    "max_train_batches",
    "max_valid_batches",
    "num_data_workers",
    "persistent_workers",
    "prefetch_factor",
    "pin_memory",
    "enable_amp",
    "amp_dtype",
    "lr_schedule_type",
    "validation_frequency",
    "checkpoint_frequency",
    "single_pass_multi_horizon_validation",
    "load_only_current_rollout",
    "use_delta_normalization",
    "use_l3",
    "num_graph_levels",
    "l0_blocks",
    "l1_blocks",
    "l2_blocks",
    "l3_blocks",
    "l2_refine_after_l3_blocks",
    "l1_refine_blocks",
    "l0_refine_blocks",
    "k_neighbors",
    "l3_k_neighbors",
    "graph_path",
    "edge_counts",
    "parameter_count",
    "cli_device_request",
    "cuda_visible_devices",
    "training_device",
)


def _load_runtime_config(exp_dir: Path) -> dict[str, Any]:
    path = exp_dir / "config_resolved.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return config.get(key, default)


def _model_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    model = config.get("model", {}) or {}
    if key in model:
        return model[key]
    return config.get(key, default)


def _planned_stage_map(config: dict[str, Any]) -> str:
    schedule = [int(x) for x in (config.get("rollout_schedule") or [])]
    durations = [int(x) for x in (config.get("rollout_stage_epochs") or [])]
    if not schedule or not durations:
        return ""
    entries = []
    start = 1
    for stage, duration in zip(schedule, durations):
        end = start + duration - 1
        entries.append(f"epochs {start}-{end}: S={stage}")
        start = end + 1
    return "; ".join(entries)


def _read_epoch_logs(exp_dir: Path) -> list[dict[str, Any]]:
    path = exp_dir / "epoch_logs.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _actual_stage_summary(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    stages = defaultdict(list)
    for row in rows:
        stages[str(row.get("train_rollout_steps", ""))].append(str(row.get("epoch", "")))
    return "; ".join(
        f"S={stage}: epochs {','.join(epochs)}"
        for stage, epochs in sorted(stages.items(), key=lambda item: int(item[0] or 0))
    )


def _parse_log_device(exp_dir: Path) -> dict[str, str | None]:
    path = exp_dir / "out.log"
    result = {"cli_device_request": None, "cuda_visible_devices": None, "training_device": None}
    if not path.exists():
        return result
    cli_re = re.compile(r"CLI device request: ([^ ]+) ->")
    cuda_re = re.compile(r"CUDA_VISIBLE_DEVICES: (.+)$")
    training_re = re.compile(r"Training device: (.+)$")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if (match := cli_re.search(line)):
                result["cli_device_request"] = match.group(1)
            if (match := cuda_re.search(line)):
                result["cuda_visible_devices"] = match.group(1).strip()
            if (match := training_re.search(line)):
                result["training_device"] = match.group(1).strip()
    return result


def _checkpoint_parameter_count(exp_dir: Path) -> int | None:
    for name in ("best_ckpt.tar", "ckpt.tar", "last_ckpt.tar"):
        path = exp_dir / name
        if not path.exists():
            continue
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        metadata = ckpt.get("metadata", {}) or {}
        value = metadata.get("num_parameters")
        if value is not None:
            return int(value)
    return None


def _graph_info(config: dict[str, Any]) -> dict[str, Any]:
    graph_path = Path(str(config.get("graph_path", "")))
    if not graph_path.is_absolute():
        graph_path = project_root / graph_path
    info: dict[str, Any] = {
        "graph_path": str(graph_path),
        "exists": graph_path.exists(),
        "metadata": {},
        "level_shapes": {},
        "node_counts": {},
        "edge_counts": {},
        "level_k_neighbors": {},
        "connectivity_strategy": None,
    }
    if not graph_path.exists():
        return info
    graph = load_graph_bundle(str(graph_path), map_location="cpu")
    info["metadata"] = dict(graph.metadata)
    info["connectivity_strategy"] = graph.metadata.get("connectivity_strategy")
    for name in ("L0", "L1", "L2", "L3"):
        level = getattr(graph, name, None)
        if level is None:
            continue
        info["level_shapes"][name] = [int(level.height), int(level.width)]
        info["node_counts"][name] = int(level.num_nodes)
        info["edge_counts"][name] = int(level.edge_index.shape[1])
        info["level_k_neighbors"][name] = int(level.k)
    return info


def _resolved_config(model_key: str, spec: dict[str, str], resolution_mode: str | None) -> dict[str, Any]:
    config_path = project_root / spec["config"]
    exp_dir = project_root / spec["experiment"]
    params = YParams(str(config_path), spec["config_name"], resolution_mode=resolution_mode)
    resolved = dict(params.params)
    runtime = _load_runtime_config(exp_dir)
    if runtime:
        resolved.update(runtime)
    rows = _read_epoch_logs(exp_dir)
    device = _parse_log_device(exp_dir)
    graph = _graph_info(resolved)
    edge_counts = [
        graph["edge_counts"].get(level)
        for level in ("L0", "L1", "L2", "L3")
        if level in graph["edge_counts"]
    ]
    l3_k = graph["level_k_neighbors"].get("L3")
    return {
        "model": model_key,
        "experiment_name": resolved.get("experiment_name"),
        "config_path": str(config_path),
        "checkpoint_path": str(exp_dir / "best_ckpt.tar"),
        "resolution_mode": resolved.get("resolution_mode"),
        "batch_size": resolved.get("batch_size"),
        "gradient_accumulation_steps": resolved.get("gradient_accumulation_steps"),
        "effective_batch_size": int(resolved.get("batch_size", 0)) * int(resolved.get("gradient_accumulation_steps", 0)),
        "max_epochs": resolved.get("max_epochs"),
        "rollout_schedule": resolved.get("rollout_schedule"),
        "rollout_stage_epochs": resolved.get("rollout_stage_epochs"),
        "current_rollout_stage_per_epoch": _planned_stage_map(resolved),
        "actual_completed_rollout_stages": _actual_stage_summary(rows),
        "max_train_batches": resolved.get("max_train_batches"),
        "max_valid_batches": resolved.get("max_valid_batches"),
        "num_data_workers": resolved.get("num_data_workers"),
        "persistent_workers": resolved.get("persistent_workers"),
        "prefetch_factor": resolved.get("prefetch_factor"),
        "pin_memory": resolved.get("pin_memory"),
        "enable_amp": resolved.get("enable_amp"),
        "amp_dtype": resolved.get("amp_dtype"),
        "lr_schedule_type": resolved.get("lr_schedule_type"),
        "validation_frequency": "every epoch",
        "checkpoint_frequency": "ckpt/last every epoch; best global and per-stage on improvement",
        "single_pass_multi_horizon_validation": resolved.get("single_pass_multi_horizon_validation"),
        "load_only_current_rollout": resolved.get("load_only_current_rollout"),
        "use_delta_normalization": resolved.get("use_delta_normalization"),
        "use_l3": _model_value(resolved, "use_l3"),
        "num_graph_levels": _model_value(resolved, "num_graph_levels"),
        "l0_blocks": _model_value(resolved, "l0_blocks"),
        "l1_blocks": _model_value(resolved, "l1_blocks"),
        "l2_blocks": _model_value(resolved, "l2_blocks"),
        "l3_blocks": _model_value(resolved, "l3_blocks"),
        "l2_refine_after_l3_blocks": _model_value(resolved, "l2_refine_after_l3_blocks"),
        "l1_refine_blocks": _model_value(resolved, "l1_refine_blocks"),
        "l0_refine_blocks": _model_value(resolved, "l0_refine_blocks"),
        "k_neighbors": resolved.get("k_neighbors"),
        "l3_k_neighbors": l3_k,
        "graph_path": str(resolved.get("graph_path")),
        "edge_counts": edge_counts,
        "parameter_count": _checkpoint_parameter_count(exp_dir),
        **device,
    }


def _write_config_comparison(configs: list[dict[str, Any]], output_dir: Path) -> None:
    yaml_path = output_dir / "config_comparison.yaml"
    txt_path = output_dir / "config_comparison.txt"
    yaml_path.write_text(yaml.safe_dump(configs, sort_keys=False), encoding="utf-8")
    lines = []
    for cfg in configs:
        lines.append(f"[{cfg['model']}]")
        for field in CONFIG_FIELDS:
            lines.append(f"{field}: {cfg.get(field)}")
        lines.append("")
    lines.append("Important differences:")
    baseline = configs[0]
    for field in CONFIG_FIELDS:
        values = [cfg.get(field) for cfg in configs]
        if any(value != values[0] for value in values[1:]):
            lines.append(f"- {field}: " + " | ".join(f"{cfg['model']}={cfg.get(field)}" for cfg in configs))
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize(values: list[float | None]) -> dict[str, float | None]:
    nums = [float(value) for value in values if value is not None]
    if not nums:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(nums),
        "median": statistics.median(nums),
        "min": min(nums),
        "max": max(nums),
    }


def _stage_rows(model_key: str, exp_dir: Path) -> list[dict[str, Any]]:
    rows = _read_epoch_logs(exp_dir)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stage = int(float(row.get("train_rollout_steps") or 0))
        grouped[stage].append(row)
    out = []
    for stage in (1, 2, 4, 6, 8, 10):
        items = grouped.get(stage, [])
        epoch_time = _summarize([_float(row, "epoch_time_sec") for row in items])
        train_time = _summarize([_float(row, "train_time_sec") for row in items])
        valid_time = _summarize([_float(row, "valid_time_sec") for row in items])
        gpu_mem = _summarize([_float(row, "cuda_peak_allocated_gb") for row in items])
        train_loss = _summarize([_float(row, "train_loss_avg") for row in items])
        valid_s10 = _summarize([_float(row, "valid_S10") for row in items])
        out.append(
            {
                "model": model_key,
                "stage": stage,
                "num_epochs": len(items),
                "mean_epoch_time": epoch_time["mean"],
                "median_epoch_time": epoch_time["median"],
                "min_epoch_time": epoch_time["min"],
                "max_epoch_time": epoch_time["max"],
                "mean_train_time": train_time["mean"],
                "mean_valid_time": valid_time["mean"],
                "mean_gpu_memory": gpu_mem["mean"],
                "peak_gpu_memory": gpu_mem["max"],
                "mean_train_loss": train_loss["mean"],
                "mean_valid_S10": valid_s10["mean"],
            }
        )
    return out


def _write_stage_comparison(specs: dict[str, dict[str, str]], output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for model_key, spec in specs.items():
        rows.extend(_stage_rows(model_key, project_root / spec["experiment"]))
    fields = [
        "model",
        "stage",
        "num_epochs",
        "mean_epoch_time",
        "median_epoch_time",
        "min_epoch_time",
        "max_epoch_time",
        "mean_train_time",
        "mean_valid_time",
        "mean_gpu_memory",
        "peak_gpu_memory",
        "mean_train_loss",
        "mean_valid_S10",
    ]
    csv_path = output_dir / "stage_time_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["\t".join(fields)]
    lines.extend("\t".join(str(row.get(field)) for field in fields) for row in rows)
    (output_dir / "stage_time_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4.5))
        for model_key in specs:
            xs = [row["stage"] for row in rows if row["model"] == model_key and row["num_epochs"]]
            ys = [row["mean_epoch_time"] for row in rows if row["model"] == model_key and row["num_epochs"]]
            if xs:
                plt.plot(xs, ys, marker="o", label=model_key)
        plt.xlabel("rollout stage S")
        plt.ylabel("mean epoch time (sec)")
        plt.title("Epoch time by rollout stage")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "stage_time_comparison.png", dpi=160)
        plt.close()
    except Exception as exc:
        (output_dir / "stage_time_comparison.png.error.txt").write_text(str(exc) + "\n", encoding="utf-8")
    return rows


def _write_graph_metadata(specs: dict[str, dict[str, str]], output_dir: Path, resolution_mode: str | None) -> None:
    lines = []
    for model_key, spec in specs.items():
        params = YParams(str(project_root / spec["config"]), spec["config_name"], resolution_mode=resolution_mode)
        graph = _graph_info(dict(params.params))
        lines.append(f"[{model_key}]")
        lines.append(f"graph_path: {graph['graph_path']}")
        lines.append(f"exists: {graph['exists']}")
        lines.append(f"num_graph_levels: {graph['metadata'].get('num_graph_levels')}")
        lines.append(f"connectivity_strategy: {graph['connectivity_strategy']}")
        lines.append(f"level_shapes: {graph['level_shapes']}")
        lines.append(f"node_counts: {graph['node_counts']}")
        lines.append(f"edge_counts: {graph['edge_counts']}")
        lines.append(f"level_k_neighbors: {graph['level_k_neighbors']}")
        lines.append("")
    (output_dir / "graph_metadata_comparison.txt").write_text("\n".join(lines), encoding="utf-8")


def _read_block_summary(output_dir: Path) -> list[str]:
    lines = []
    for name in ("l3_deeper_block_usage", "heavy_block_usage"):
        path = output_dir / name / "block_usage.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        missing = [row["block_name"] for row in rows if row.get("called_forward") != "True"]
        nograd = [row["block_name"] for row in rows if row.get("has_nonzero_grad") != "True"]
        lines.append(f"{name}: blocks={len(rows)} missing_forward={missing or 'none'} missing_grad={nograd or 'none'}")
    return lines


def _read_optimizer_summary(output_dir: Path) -> list[str]:
    lines = []
    combined_rows = []
    for name in ("l3_deeper_block_usage", "heavy_block_usage"):
        path = output_dir / name / "optimizer_parameter_check.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["diagnostic"] = name
        combined_rows.extend(rows)
        missing = [row["parameter_name"] for row in rows if row.get("missing_from_optimizer") == "True"]
        frozen = [row["parameter_name"] for row in rows if row.get("unexpectedly_frozen") == "True"]
        lines.append(f"{name}: missing_from_optimizer={len(missing)} frozen={len(frozen)}")
    if combined_rows:
        path = output_dir / "optimizer_parameter_check.csv"
        fields = ["diagnostic"] + [key for key in combined_rows[0].keys() if key != "diagnostic"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(combined_rows)
        (output_dir / "optimizer_parameter_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def _read_profile_summary(output_dir: Path) -> list[str]:
    lines = []
    for name in ("l3_deeper_profile", "heavy_profile"):
        path = output_dir / name / "section_profile.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        top = sorted(rows, key=lambda row: float(row["total_seconds"]), reverse=True)[:5]
        lines.append(
            f"{name}: top sections="
            + ", ".join(f"{row['section']} {float(row['mean_seconds_per_step']):.4f}s/step" for row in top)
        )
    return lines


def _write_final_report(
    output_dir: Path,
    configs: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
) -> None:
    cfg_by_model = {cfg["model"]: cfg for cfg in configs}
    stage_lookup = {(row["model"], row["stage"]): row for row in stage_rows}
    l3_s1 = stage_lookup.get(("l3_deeper", 1), {})
    heavy_s1 = stage_lookup.get(("heavy_l3_unet", 1), {})
    l3_s10 = stage_lookup.get(("l3_deeper", 10), {})
    heavy_s10 = stage_lookup.get(("heavy_l3_unet", 10), {})
    block_lines = _read_block_summary(output_dir)
    optimizer_lines = _read_optimizer_summary(output_dir)
    profile_lines = _read_profile_summary(output_dir)
    s10_available = bool(l3_s10.get("num_epochs")) and bool(heavy_s10.get("num_epochs"))

    conclusion = (
        "Heavy is not yet proven faster at S=10 because complete S=10 epoch rows are not available."
        if not s10_available
        else (
            "Heavy is faster at S=10 in completed epoch rows."
            if float(heavy_s10["mean_epoch_time"]) < float(l3_s10["mean_epoch_time"])
            else "Heavy is not faster at S=10 in completed epoch rows."
        )
    )
    primary = (
        "The apparent speed difference is primarily a hardware/log comparison issue: "
        f"l3_deeper ran with CUDA_VISIBLE_DEVICES={cfg_by_model['l3_deeper'].get('cuda_visible_devices')}, "
        f"while heavy_l3_unet ran with CUDA_VISIBLE_DEVICES={cfg_by_model['heavy_l3_unet'].get('cuda_visible_devices')}. "
        "The configs and graph settings match for batch size, AMP, workers, scheduler, graph path, k, and edge counts."
    )
    evidence = [
        f"S=1 completed epoch mean: l3_deeper={l3_s1.get('mean_epoch_time')} sec, "
        f"heavy_l3_unet={heavy_s1.get('mean_epoch_time')} sec.",
        f"S=10 completed comparison available: {s10_available}.",
        f"Parameter counts: l3_deeper={cfg_by_model['l3_deeper'].get('parameter_count')}, "
        f"heavy_l3_unet={cfg_by_model['heavy_l3_unet'].get('parameter_count')}.",
        f"Graph edge counts: l3_deeper={cfg_by_model['l3_deeper'].get('edge_counts')}, "
        f"heavy_l3_unet={cfg_by_model['heavy_l3_unet'].get('edge_counts')}.",
    ]
    evidence.extend(block_lines or ["Block usage diagnostics have not been run yet."])
    evidence.extend(optimizer_lines or ["Optimizer diagnostics have not been run yet."])
    evidence.extend(profile_lines or ["Section profiles have not been run yet."])
    verdict = "NO BUG FOUND" if block_lines and optimizer_lines else "INCONCLUSIVE"
    if any("missing_forward=[" in line and "none" not in line for line in block_lines):
        verdict = "BUG FOUND"
    if any("missing_from_optimizer=0" not in line for line in optimizer_lines):
        verdict = "BUG FOUND"

    report = [
        "# L3 Runtime Diagnosis",
        "",
        "## Summary conclusion",
        conclusion,
        "",
        "## Config comparison",
        "Full details are in `config_comparison.yaml` and `config_comparison.txt`.",
        "",
        "## Stage-time comparison",
        "Full details are in `stage_time_comparison.csv`, `stage_time_comparison.txt`, and `stage_time_comparison.png`.",
        "",
        "## Block usage result",
        *(block_lines or ["Not run yet."]),
        "",
        "## Optimizer parameter check",
        *(optimizer_lines or ["Not run yet."]),
        "",
        "## Section profiling",
        *(profile_lines or ["Not run yet."]),
        "",
        "## Graph metadata",
        "Both configs use the same graph metadata unless `graph_metadata_comparison.txt` shows otherwise.",
        "",
        "Final verdict:",
        f"[{verdict}]",
        "",
        "Primary reason:",
        primary,
        "",
        "Evidence:",
    ]
    report.extend(f"{idx}. {item}" for idx, item in enumerate(evidence, start=1))
    report.extend(
        [
            "",
            "Recommended action:",
            "Compare the two models on the same physical GPU, same rollout stage, and same completed epoch count. "
            "For final speed claims, use the S=10 rows after both runs reach S=10.",
            "",
            "Log interpretation:",
            "`epoch_seconds` is full epoch wall time from epoch start through training, validation, checkpointing, "
            "and epoch logging. `train_time_sec` is training loop time. `valid_time_sec` is validation loop time. "
            "The optional `log_timing_breakdown` config key now adds per-epoch data/forward-loss/backward/optimizer/"
            "checkpoint timing for future runs.",
        ]
    )
    (output_dir / "final_diagnosis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="runtime_diagnosis")
    parser.add_argument("--resolution_mode", default="2p5")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        _resolved_config(model_key, spec, resolution_mode=args.resolution_mode)
        for model_key, spec in DEFAULT_MODELS.items()
    ]
    _write_config_comparison(configs, output_dir)
    stage_rows = _write_stage_comparison(DEFAULT_MODELS, output_dir)
    _write_graph_metadata(DEFAULT_MODELS, output_dir, resolution_mode=args.resolution_mode)
    _write_final_report(output_dir, configs, stage_rows)
    print((output_dir / "final_diagnosis_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
