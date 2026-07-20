from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.config import YParams  # noqa: E402
from src.trainer import warmup_cosine_lr  # noqa: E402


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
    yaml_path = yaml_path or str(project_root / "configs" / "weather_dual_resolution.yaml")
    config_name = config_name or "raw"
    return os.path.abspath(os.path.expanduser(yaml_path)), config_name


def _lr_for_epoch(params: YParams, epoch: int) -> float:
    schedule_type = str(params.get("lr_schedule_type", "none")).strip().lower()
    if schedule_type == "warmup_cosine":
        return warmup_cosine_lr(
            epoch - 1,
            base_lr=float(params.lr),
            min_lr=float(params.get("min_lr", 0.0)),
            warmup_epochs=int(params.get("warmup_epochs", 0)),
            warmup_start_factor=float(params.get("warmup_start_factor", 0.1)),
            max_epochs=int(params.max_epochs),
        )
    return float(params.lr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default=str(project_root / "configs" / "weather_dual_resolution.yaml"), type=str)
    parser.add_argument("--config", default="raw", type=str)
    parser.add_argument("--config_name", default=None, type=str)
    parser.add_argument("--resolution_mode", default=None, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    args = parser.parse_args()

    yaml_path, config_name = _resolve_config_args(args)
    params = YParams(yaml_path, config_name, resolution_mode=args.resolution_mode)
    schedule_type = str(params.get("lr_schedule_type", "none")).strip().lower()
    if schedule_type == "warmup_cosine" and params.get("lr_by_rollout", None):
        print("lr_schedule_type=warmup_cosine: ignoring lr_by_rollout.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, int(params.max_epochs) + 1))
    lrs = [_lr_for_epoch(params, epoch) for epoch in epochs]

    csv_path = output_dir / "lr_schedule.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "lr"])
        writer.writeheader()
        for epoch, lr in zip(epochs, lrs):
            writer.writerow({"epoch": epoch, "lr": f"{lr:.12g}"})

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_path = output_dir / "lr_schedule.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, lrs, marker="o", markersize=2.5, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Warmup + Cosine LR Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
