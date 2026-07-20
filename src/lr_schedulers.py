from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StageLRSchedule:
    stage_index: int
    rollout_steps: int
    stage_epochs: int
    start_epoch: int
    end_epoch: int
    start_step: int
    end_step: int
    max_lr: float
    min_lr: float
    warmup_epochs: int
    warmup_steps: int
    warmup_start_factor: float

    @property
    def total_steps(self) -> int:
        return max(1, int(self.end_step - self.start_step))

    @property
    def warmup_start_lr(self) -> float:
        return float(self.max_lr) * float(self.warmup_start_factor)


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}.") from exc


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def build_stage_lr_schedules(
    rollout_schedule: list[int],
    rollout_stage_epochs: list[int],
    rollout_stage_lr_schedule: list[dict[str, Any]],
    steps_per_epoch: int,
) -> list[StageLRSchedule]:
    if len(rollout_schedule) != len(rollout_stage_epochs):
        raise ValueError(
            "rollout_schedule length must equal rollout_stage_epochs length for rollout_stage_warmup_cosine."
        )
    steps_per_epoch = max(1, int(steps_per_epoch))
    entries_by_steps: dict[int, dict[str, Any]] = {}
    for raw in list(rollout_stage_lr_schedule or []):
        if not isinstance(raw, dict):
            raise ValueError("Each rollout_stage_lr_schedule entry must be a mapping.")
        if "rollout_steps" not in raw:
            raise ValueError("Each rollout_stage_lr_schedule entry must include rollout_steps.")
        steps = int(raw["rollout_steps"])
        if steps in entries_by_steps:
            raise ValueError(f"Duplicate rollout_stage_lr_schedule entry for rollout_steps={steps}.")
        entries_by_steps[steps] = dict(raw)

    expected = {int(x) for x in rollout_schedule}
    actual = set(entries_by_steps)
    if actual != expected:
        raise ValueError(
            "rollout_stage_lr_schedule must contain exactly one entry per rollout stage. "
            f"Expected {sorted(expected)}, got {sorted(actual)}."
        )

    schedules: list[StageLRSchedule] = []
    epoch_cursor = 0
    step_cursor = 0
    for index, (rollout_steps, stage_epochs_raw) in enumerate(zip(rollout_schedule, rollout_stage_epochs)):
        stage_epochs = _as_int(stage_epochs_raw, f"rollout_stage_epochs[{index}]")
        if stage_epochs < 1:
            raise ValueError(f"rollout_stage_epochs[{index}] must be >= 1, got {stage_epochs}.")
        entry = entries_by_steps[int(rollout_steps)]
        missing = [
            key
            for key in ("rollout_steps", "max_lr", "min_lr", "warmup_epochs", "warmup_start_factor")
            if key not in entry
        ]
        if missing:
            raise ValueError(f"rollout_stage_lr_schedule entry S={rollout_steps} is missing keys: {missing}.")
        max_lr = _as_float(entry["max_lr"], f"S={rollout_steps} max_lr")
        min_lr = _as_float(entry["min_lr"], f"S={rollout_steps} min_lr")
        warmup_epochs = _as_int(entry["warmup_epochs"], f"S={rollout_steps} warmup_epochs")
        warmup_start_factor = _as_float(
            entry["warmup_start_factor"],
            f"S={rollout_steps} warmup_start_factor",
        )
        if max_lr <= 0.0:
            raise ValueError(f"S={rollout_steps} max_lr must be > 0, got {max_lr}.")
        if min_lr < 0.0:
            raise ValueError(f"S={rollout_steps} min_lr must be >= 0, got {min_lr}.")
        if min_lr > max_lr:
            raise ValueError(f"S={rollout_steps} min_lr must be <= max_lr.")
        if not (0.0 < warmup_start_factor <= 1.0):
            raise ValueError(f"S={rollout_steps} warmup_start_factor must satisfy 0 < value <= 1.")
        if warmup_epochs < 0:
            raise ValueError(f"S={rollout_steps} warmup_epochs must be >= 0.")
        if stage_epochs > 1 and warmup_epochs >= stage_epochs:
            raise ValueError(
                f"S={rollout_steps} warmup_epochs must be < stage_epochs unless stage has one epoch."
            )
        total_steps = stage_epochs * steps_per_epoch
        warmup_steps = min(total_steps, max(0, warmup_epochs) * steps_per_epoch)
        schedules.append(
            StageLRSchedule(
                stage_index=index,
                rollout_steps=int(rollout_steps),
                stage_epochs=stage_epochs,
                start_epoch=epoch_cursor,
                end_epoch=epoch_cursor + stage_epochs,
                start_step=step_cursor,
                end_step=step_cursor + total_steps,
                max_lr=max_lr,
                min_lr=min_lr,
                warmup_epochs=warmup_epochs,
                warmup_steps=warmup_steps,
                warmup_start_factor=warmup_start_factor,
            )
        )
        epoch_cursor += stage_epochs
        step_cursor += total_steps
    return schedules


class RolloutStageWarmupCosineScheduler:
    def __init__(
        self,
        optimizer: Any,
        rollout_schedule: list[int],
        rollout_stage_epochs: list[int],
        rollout_stage_lr_schedule: list[dict[str, Any]],
        steps_per_epoch: int,
        current_step: int = 0,
    ):
        self.optimizer = optimizer
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.stages = build_stage_lr_schedules(
            [int(x) for x in rollout_schedule],
            [int(x) for x in rollout_stage_epochs],
            rollout_stage_lr_schedule,
            steps_per_epoch=self.steps_per_epoch,
        )
        self.global_step = max(0, int(current_step))
        self.last_lr = [self.get_lr_for_global_step(self.global_step)]
        self._set_optimizer_lr(self.last_lr[0])

    def _set_optimizer_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    @property
    def total_steps(self) -> int:
        return int(self.stages[-1].end_step)

    def state_dict(self) -> dict[str, Any]:
        return {
            "global_step": int(self.global_step),
            "steps_per_epoch": int(self.steps_per_epoch),
            "last_lr": list(self.last_lr),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.global_step = max(0, int(state.get("global_step", 0)))
        self.last_lr = [self.get_lr_for_global_step(self.global_step)]
        self._set_optimizer_lr(self.last_lr[0])

    def get_last_lr(self) -> list[float]:
        return list(self.last_lr)

    def get_stage_for_epoch(self, epoch_index_zero_based: int) -> StageLRSchedule:
        epoch = int(epoch_index_zero_based)
        for stage in self.stages:
            if stage.start_epoch <= epoch < stage.end_epoch:
                return stage
        return self.stages[-1]

    def get_stage_for_step(self, global_step: int) -> StageLRSchedule:
        step = int(global_step)
        for stage in self.stages:
            if stage.start_step <= step < stage.end_step:
                return stage
        return self.stages[-1]

    def get_stage_progress(self, global_step: int) -> tuple[StageLRSchedule, int]:
        stage = self.get_stage_for_step(global_step)
        stage_step = min(max(0, int(global_step) - stage.start_step), stage.total_steps - 1)
        return stage, stage_step

    def get_lr_for_stage_progress(self, stage: StageLRSchedule, stage_step: int) -> float:
        stage_step = min(max(0, int(stage_step)), stage.total_steps - 1)
        if stage.warmup_steps > 0 and stage_step < stage.warmup_steps:
            if stage.warmup_steps == 1:
                return stage.warmup_start_lr
            alpha = float(stage_step) / float(stage.warmup_steps - 1)
            return stage.warmup_start_lr + alpha * (stage.max_lr - stage.warmup_start_lr)

        decay_steps = max(1, stage.total_steps - stage.warmup_steps)
        decay_step = min(max(0, stage_step - stage.warmup_steps), decay_steps - 1)
        if decay_steps == 1:
            progress = 1.0
        else:
            progress = float(decay_step) / float(decay_steps - 1)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return stage.min_lr + (stage.max_lr - stage.min_lr) * cosine_factor

    def get_lr_for_global_step(self, global_step: int) -> float:
        stage, stage_step = self.get_stage_progress(global_step)
        return self.get_lr_for_stage_progress(stage, stage_step)

    def step(self) -> dict[str, Any]:
        used_step = int(self.global_step)
        stage, stage_step = self.get_stage_progress(used_step)
        used_lr = float(self.last_lr[0])
        self.global_step += 1
        next_lr = self.get_lr_for_global_step(self.global_step)
        self.last_lr = [next_lr]
        self._set_optimizer_lr(next_lr)
        return {
            "global_step": used_step,
            "epoch": int(used_step // self.steps_per_epoch) + 1,
            "rollout_steps": int(stage.rollout_steps),
            "stage_index": int(stage.stage_index),
            "stage_step": int(stage_step),
            "lr": used_lr,
        }

    def stage_for_epoch_info(self, epoch_index_zero_based: int) -> dict[str, Any]:
        stage = self.get_stage_for_epoch(epoch_index_zero_based)
        return {
            "stage_index": int(stage.stage_index),
            "rollout_steps": int(stage.rollout_steps),
            "stage_epoch": int(epoch_index_zero_based) - int(stage.start_epoch) + 1,
            "stage_epochs": int(stage.stage_epochs),
        }

    def stage_schedule_dicts(self) -> list[dict[str, Any]]:
        return [asdict(stage) for stage in self.stages]
