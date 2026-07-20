from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.batch_adapter import GridNodeAdapter
from src.config import YParams, setup_logging
from src.data import ClimateNetCDFDataset, DataConfig
from src.features import VariableResolver
from src.losses import LatitudeWeightedMSE
from src.target_handling import TargetHandling


def _looks_like_yaml(value: str) -> bool:
    return value.endswith((".yaml", ".yml")) or Path(value).expanduser().exists()


def _default_config_name(path: str) -> str:
    name = Path(path).name
    if name == "weather_dual_resolution_l3_orog_tisr_fixed.yaml":
        return "raw_l3_orog_tisr_fixed"
    if name == "weather_dual_resolution_l3_hidden128_orog_tisr_fixed.yaml":
        return "raw_l3_hidden128_orog_tisr_fixed"
    if name == "weather_dual_resolution_l3_hidden160_orog_tisr_fixed.yaml":
        return "raw_l3_hidden160_orog_tisr_fixed"
    if name == "weather_dual_resolution_l3_hidden128.yaml":
        return "raw_l3_hidden128"
    if name == "weather_dual_resolution_l3_hidden160.yaml":
        return "raw_l3_hidden160"
    if name == "weather_dual_resolution_l3_blocks3.yaml":
        return "raw_l3_blocks3"
    if name == "weather_dual_resolution_l3_heavy_unet.yaml":
        return "raw_l3_heavy_unet"
    if name == "weather_dual_resolution_l3_full_rollout.yaml":
        return "raw_l3_full_rollout"
    if name == "weather_dual_resolution_l3_stage_warmup_cosine.yaml":
        return "raw_l3_stage_warmup_cosine"
    if name == "weather_dual_resolution_l3.yaml":
        return "raw_l3"
    return "raw_5p625"


def _resolve_config_args(args: argparse.Namespace) -> tuple[str, str]:
    yaml_path = args.yaml_config
    config_name = args.config_name
    if args.config:
        if _looks_like_yaml(args.config):
            yaml_path = args.config
            config_name = config_name or _default_config_name(args.config)
        elif config_name is None:
            config_name = args.config
    yaml_path = yaml_path or str(project_root / "configs" / "gnn_5p625.yaml")
    config_name = config_name or "raw_5p625"
    return str(Path(yaml_path).expanduser().resolve()), config_name


def _get(params: Any, name: str, default: Any = None) -> Any:
    return getattr(params, name, default)


def _split_data_path(params: Any, split: str) -> str:
    if split == "train":
        return str(params.train_data_path)
    if split in {"valid", "val"}:
        return str(params.valid_data_path)
    if split == "test":
        return str(_get(params, "test_dataset_path", params.valid_data_path))
    raise ValueError("split must be train, valid, or test.")


def _target_sequence(target: torch.Tensor) -> torch.Tensor:
    if target.dim() == 3:
        return target.unsqueeze(0)
    if target.dim() == 4:
        return target
    raise ValueError(f"Expected target [C,H,W] or [S,C,H,W], got {tuple(target.shape)}")


def _unpack_sample(sample: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if isinstance(sample, dict):
        return sample["input"], sample["target"], {key: value for key, value in sample.items() if key not in {"input", "target"}}
    inp, target = sample
    return inp, target, {}


def _status(passed: bool, message: str = "") -> dict[str, Any]:
    return {"pass": bool(passed), "message": message}


def _build_dataset(params: Any, split: str, rollout_steps: int) -> ClimateNetCDFDataset:
    cfg = DataConfig(
        dt=int(params.dt),
        n_history=int(params.n_history),
        in_channels=list(params.in_channels),
        out_channels=list(params.out_channels),
        crop_size_x=_get(params, "crop_size_x", None),
        crop_size_y=_get(params, "crop_size_y", None),
        roll=False,
        orography=bool(_get(params, "orography", False)),
        orography_path=_get(params, "orography_path", None),
        add_noise=False,
        noise_std=0.0,
        normalize=(str(_get(params, "normalization", "zscore")).lower() == "zscore"),
        normalization=str(_get(params, "normalization", "zscore")),
        global_means_path=params.global_means_path,
        global_stds_path=params.global_stds_path,
        add_grid=bool(_get(params, "add_grid", False)),
        gridtype=str(_get(params, "gridtype", "linear")),
        N_grid_channels=int(_get(params, "N_grid_channels", 0)),
        rollout_steps=int(rollout_steps),
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        resolution_mode=str(_get(params, "resolution_mode", "5p625")),
        expected_grid_shape=_get(params, "expected_grid_shape", _get(params, "grid_shape", None)),
        return_metadata=False,
    )
    return ClimateNetCDFDataset(cfg, _split_data_path(params, split), train=False)


def _check_rollout(
    handler: TargetHandling,
    inp: torch.Tensor,
    target_seq: torch.Tensor,
    params: Any,
    orog_idx: int,
    rollout_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    adapter = GridNodeAdapter(
        grid_shape=(int(target_seq.shape[-2]), int(target_seq.shape[-1])),
        input_channels=int(inp.shape[0]),
        output_channels=int(target_seq.shape[1]),
        n_history=int(params.n_history),
    )
    _, current = adapter.extract_two_steps(inp.unsqueeze(0))
    initial_state = current.clone()
    orog_rows = []
    next_input_rows = []

    for lead in range(1, int(rollout_steps) + 1):
        pred_next = current[:, : int(target_seq.shape[1])].clone()
        pred_next[:, orog_idx] = pred_next[:, orog_idx] + 100000.0 + float(lead)
        pred_next = handler.apply(
            pred_next=pred_next,
            current_state=current,
            initial_state=initial_state,
            lead=lead,
        )
        orog_diff = float((pred_next[:, orog_idx] - initial_state[:, orog_idx]).abs().max().item())
        orog_rows.append({"lead": lead, "max_abs_diff": orog_diff, "pass": orog_diff <= 1.0e-6})
        next_step = current.clone()
        next_step[:, : int(target_seq.shape[1])] = pred_next
        next_input_diff = float((next_step[:, orog_idx] - initial_state[:, orog_idx]).abs().max().item())
        next_input_rows.append({"lead": lead, "max_abs_diff": next_input_diff, "pass": next_input_diff <= 1.0e-6})
        current = next_step

    return (
        _status(all(row["pass"] for row in orog_rows), "" if all(row["pass"] for row in orog_rows) else "orog changed during rollout")
        | {"leads": orog_rows},
        _status(
            all(row["pass"] for row in next_input_rows),
            "" if all(row["pass"] for row in next_input_rows) else "corrected orog was not used in next autoregressive input",
        )
        | {"leads": next_input_rows},
    )


def _check_loss_mask(
    handler: TargetHandling,
    target_seq: torch.Tensor,
    orog_idx: int,
    t2m_idx: int,
) -> dict[str, Any]:
    mask = handler.loss_channel_mask(int(target_seq.shape[1]))
    if mask is None:
        return _status(False, "target_handling did not create a loss mask")
    loss_fn = LatitudeWeightedMSE(torch.zeros(int(target_seq.shape[-2]), dtype=torch.float32))
    target = target_seq[0:1]
    pred = target.clone()
    pred[:, orog_idx] = pred[:, orog_idx] + 100000.0
    static_loss = float(loss_fn(pred, target, channel_mask=mask).item())
    pred = target.clone()
    pred[:, t2m_idx] = pred[:, t2m_idx] + 1.0
    dynamic_loss = float(loss_fn(pred, target, channel_mask=mask).item())
    included = int(mask.sum().item())
    passed = static_loss <= 1.0e-8 and dynamic_loss > 0.0 and included == int(target_seq.shape[1]) - 1
    return {
        **_status(passed, "" if passed else "loss mask did not exclude exactly orog"),
        "orog_only_loss": static_loss,
        "t2m_error_loss": dynamic_loss,
        "loss_channels": included,
        "output_channels": int(target_seq.shape[1]),
        "mask": [float(x) for x in mask.tolist()],
    }


def _write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "target_handling_report.json"
    txt_path = output_dir / "target_handling_report.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    checks = report["checks"]
    lines = [
        "Target handling debug report",
        "",
        ("[PASS] " if checks["orog_channel_resolved"]["pass"] else "[FAIL] ") + "orog channel resolved",
        ("[PASS] " if checks["orog_copied"]["pass"] else "[FAIL] ")
        + f"orog copied unchanged through {report['rollout_steps']}-step rollout",
        ("[PASS] " if checks["corrected_orog_next_input"]["pass"] else "[FAIL] ")
        + "corrected orog is used in next autoregressive input",
        ("[PASS] " if checks["loss_exclusion"]["pass"] else "[FAIL] ") + "orog excluded from loss",
        ("[PASS] " if checks["loss_channels"]["pass"] else "[FAIL] ") + "loss channels = 66/67",
        ("[PASS] " if checks["extra_features_disabled_target_active"]["pass"] else "[FAIL] ")
        + "extra_features disabled but target handling active",
        "",
        f"JSON: {json_path}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug fixed-orography target_handling copy/loss behavior.")
    parser.add_argument("--yaml_config", default=str(project_root / "configs" / "gnn_5p625.yaml"), type=str)
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--config_name", default=None, type=str)
    parser.add_argument("--resolution_mode", default=None, type=str)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--sample_index", default=0, type=int)
    parser.add_argument("--rollout_steps", default=10, type=int)
    parser.add_argument("--output_dir", required=True, type=str)
    args = parser.parse_args()

    yaml_path, config_name = _resolve_config_args(args)
    params = YParams(yaml_path, config_name, resolution_mode=args.resolution_mode)
    output_dir = Path(args.output_dir)
    setup_logging(rank=0, log_file=str(output_dir / "debug_target_handling.log"))
    params.log()

    dataset = _build_dataset(params, args.split, int(args.rollout_steps))
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample_index={args.sample_index} is outside split length {len(dataset)}")
    inp, target, _ = _unpack_sample(dataset[int(args.sample_index)])
    target_seq = _target_sequence(target)
    if int(target_seq.shape[0]) < int(args.rollout_steps):
        raise ValueError(f"Target sequence length {target_seq.shape[0]} is shorter than rollout_steps={args.rollout_steps}.")

    handler = TargetHandling.from_params(
        params,
        channel_names=getattr(dataset, "channel_names", None),
        out_channels=list(params.out_channels),
        logger=logging,
    )
    handler.log_startup(output_channels=int(target_seq.shape[1]))
    resolver = VariableResolver(params, getattr(dataset, "channel_names", None), list(params.out_channels), logger=logging)
    orog = resolver.resolve("orog", required=True)
    t2m = resolver.resolve("t2m", required=True)
    if orog.local_index is None or t2m.local_index is None:
        raise RuntimeError("orog and t2m must be present in output channels for this debug check.")

    orog_check, next_input_check = _check_rollout(
        handler,
        inp,
        target_seq,
        params,
        int(orog.local_index),
        int(args.rollout_steps),
    )
    loss_check = _check_loss_mask(handler, target_seq, int(orog.local_index), int(t2m.local_index))
    loss_channels_check = _status(
        int(loss_check.get("loss_channels", -1)) == 66 and int(loss_check.get("output_channels", -1)) == 67,
        f"loss channels were {loss_check.get('loss_channels')}/{loss_check.get('output_channels')}",
    )
    extra = dict(_get(params, "extra_features", {}) or {})
    target = dict(_get(params, "target_handling", {}) or {})
    extra_target_check = _status(
        not bool(extra.get("enabled", False)) and bool(target.get("enabled", False)) and bool(handler.enabled),
        "extra_features must be disabled while target_handling is enabled",
    )
    report = {
        "config": {"yaml_config": yaml_path, "config_name": config_name, "resolution_mode": str(params.resolution_mode)},
        "split": args.split,
        "sample_index": int(args.sample_index),
        "rollout_steps": int(args.rollout_steps),
        "target_handling": handler.metadata,
        "channels": {
            "orog": int(orog.local_index),
            "t2m": int(t2m.local_index),
        },
        "checks": {
            "orog_channel_resolved": _status(orog.local_index is not None),
            "orog_copied": orog_check,
            "corrected_orog_next_input": next_input_check,
            "loss_exclusion": loss_check,
            "loss_channels": loss_channels_check,
            "extra_features_disabled_target_active": extra_target_check,
        },
    }
    _write_reports(output_dir, report)
    text = (output_dir / "target_handling_report.txt").read_text(encoding="utf-8")
    print(text)
    if not all(check["pass"] for check in report["checks"].values()):
        raise RuntimeError("Target handling debug checks failed.")


if __name__ == "__main__":
    main()
