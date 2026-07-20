from __future__ import annotations

import logging
import os
from typing import Any

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - depends on optional environment package
    wandb = None

DEFAULT_WANDB_ENTITY = "amin1jafarzade-kaist"


def maybe_get_wandb_run(config: dict[str, Any], logger: Any = logging):
    wandb_config = dict((config or {}).get("wandb", {}) or {})
    if not bool(wandb_config.get("enabled", False)):
        return None
    if wandb is None:
        logger.warning("Diagnostics W&B logging requested, but wandb is not installed. Using local diagnostics only.")
        return None
    if getattr(wandb, "run", None) is not None:
        return wandb.run
    project = wandb_config.get("project")
    if not project:
        logger.warning("Diagnostics W&B logging requested, but diagnostics.wandb.project is null. Using local diagnostics only.")
        return None
    return wandb.init(
        project=project,
        entity=wandb_config.get("entity") or os.environ.get("WANDB_ENTITY") or DEFAULT_WANDB_ENTITY,
        name=wandb_config.get("run_name"),
        tags=list(wandb_config.get("tags", []) or []),
    )


def log_metrics(run: Any, metrics: dict[str, float], step: int | None = None) -> None:
    if run is None or not metrics:
        return
    payload = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
    if not payload:
        return
    run.log(payload, step=step)
