from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.climatology import build_dayofyear_climatology
from src.config import YParams, setup_logging


def _resolve_config_args(args: argparse.Namespace) -> tuple[str, str]:
    default_yaml = project_root / "configs" / "gnn_5p625.yaml"
    yaml_path = args.yaml_config
    config_name = args.config_name
    if args.config:
        looks_like_yaml = args.config.endswith((".yaml", ".yml")) or Path(args.config).exists()
        if looks_like_yaml:
            yaml_path = args.config
            if config_name is None and Path(args.config).name == "weather_dual_resolution.yaml":
                config_name = "raw"
        elif config_name is None:
            config_name = args.config
    yaml_path = yaml_path or str(default_yaml)
    config_name = config_name or "raw_5p625"
    return str(Path(yaml_path).expanduser().resolve()), config_name


def _split_data_path(params: YParams, split: str) -> str:
    if split == "train":
        return str(params.train_data_path)
    if split == "valid":
        return str(params.valid_data_path)
    if split == "test":
        return str(params.get("test_dataset_path", params.valid_data_path))
    raise ValueError(f"Unsupported split: {split}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build calendar-aware day-of-year climatology from NetCDF time coordinates.")
    parser.add_argument("--config", default=str(project_root / "configs" / "gnn_5p625.yaml"), type=str)
    parser.add_argument("--yaml_config", default=None, type=str)
    parser.add_argument("--config_name", default=None, type=str)
    parser.add_argument("--resolution_mode", default=None, type=str, help="5p625 or 2p5; overrides YAML resolution_mode.")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--data_path", default=None, type=str, help="Optional explicit NetCDF file or directory.")
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--chunk_size", default=32, type=int)
    parser.add_argument("--metadata_json", default=None, type=str)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    yaml_path, config_name = _resolve_config_args(args)
    params = YParams(yaml_path, config_name, resolution_mode=args.resolution_mode)
    output_path = Path(args.output).expanduser().resolve()
    log_path = output_path.with_suffix(output_path.suffix + ".log")
    setup_logging(rank=0, log_file=str(log_path))
    params.log()

    data_path = args.data_path or _split_data_path(params, args.split)
    metadata = build_dayofyear_climatology(
        data_path=data_path,
        output_path=output_path,
        out_channels=[int(x) for x in params.out_channels],
        split=str(args.split),
        chunk_size=int(args.chunk_size),
        logger=logging,
    )
    metadata_json = Path(args.metadata_json).expanduser().resolve() if args.metadata_json else output_path.with_suffix(".json")
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logging.info("Saved climatology metadata JSON: %s", metadata_json)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
