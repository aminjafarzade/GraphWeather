from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import YParams  # noqa: E402
from src.lr_schedulers import RolloutStageWarmupCosineScheduler  # noqa: E402
from src.trainer import Trainer, warmup_cosine_lr  # noqa: E402


ROLL_OUT = [1, 2, 4, 6, 8, 10]
STAGE_EPOCHS = [5, 5, 10, 10, 10, 10]
STAGE_LRS = [
    {"rollout_steps": 1, "max_lr": 1.0e-4, "min_lr": 6.0e-5, "warmup_epochs": 1, "warmup_start_factor": 0.25},
    {"rollout_steps": 2, "max_lr": 1.0e-4, "min_lr": 5.0e-5, "warmup_epochs": 1, "warmup_start_factor": 0.25},
    {"rollout_steps": 4, "max_lr": 8.0e-5, "min_lr": 3.0e-5, "warmup_epochs": 1, "warmup_start_factor": 0.25},
    {"rollout_steps": 6, "max_lr": 6.0e-5, "min_lr": 2.0e-5, "warmup_epochs": 1, "warmup_start_factor": 0.25},
    {"rollout_steps": 8, "max_lr": 4.0e-5, "min_lr": 1.0e-5, "warmup_epochs": 1, "warmup_start_factor": 0.25},
    {"rollout_steps": 10, "max_lr": 3.0e-5, "min_lr": 5.0e-6, "warmup_epochs": 1, "warmup_start_factor": 0.25},
]


def _scheduler(steps_per_epoch: int = 2) -> tuple[RolloutStageWarmupCosineScheduler, torch.optim.Optimizer]:
    param = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.SGD([param], lr=1.0)
    sched = RolloutStageWarmupCosineScheduler(
        opt,
        rollout_schedule=ROLL_OUT,
        rollout_stage_epochs=STAGE_EPOCHS,
        rollout_stage_lr_schedule=STAGE_LRS,
        steps_per_epoch=steps_per_epoch,
    )
    return sched, opt


class RolloutStageWarmupCosineSchedulerTest(unittest.TestCase):
    def test_stage_mapping(self) -> None:
        sched, _ = _scheduler()
        expected = {
            **{epoch: 1 for epoch in range(0, 5)},
            **{epoch: 2 for epoch in range(5, 10)},
            **{epoch: 4 for epoch in range(10, 20)},
            **{epoch: 6 for epoch in range(20, 30)},
            **{epoch: 8 for epoch in range(30, 40)},
            **{epoch: 10 for epoch in range(40, 50)},
        }
        for epoch, rollout in expected.items():
            with self.subTest(epoch=epoch):
                self.assertEqual(sched.get_stage_for_epoch(epoch).rollout_steps, rollout)

    def test_lr_restarts_at_long_rollout_stages(self) -> None:
        sched, _ = _scheduler(steps_per_epoch=4)
        s8_step = sched.get_stage_for_epoch(30).start_step
        s10_step = sched.get_stage_for_epoch(40).start_step
        s8_lr = sched.get_lr_for_global_step(s8_step)
        s10_lr = sched.get_lr_for_global_step(s10_step)
        global_cosine_final = warmup_cosine_lr(49, 1.0e-4, 3.0e-6, 2, 0.1, 50)
        self.assertGreater(s8_lr, global_cosine_final)
        self.assertAlmostEqual(s8_lr, 1.0e-5)
        self.assertAlmostEqual(s10_lr, 7.5e-6)

    def test_lr_bounds(self) -> None:
        sched, _ = _scheduler(steps_per_epoch=3)
        by_rollout = {entry["rollout_steps"]: entry for entry in STAGE_LRS}
        for step in range(sched.total_steps):
            stage, _ = sched.get_stage_progress(step)
            lr = sched.get_lr_for_global_step(step)
            cfg = by_rollout[stage.rollout_steps]
            lower = min(float(cfg["min_lr"]), float(cfg["max_lr"]) * float(cfg["warmup_start_factor"]))
            self.assertGreaterEqual(lr + 1.0e-14, lower)
            self.assertLessEqual(lr, float(cfg["max_lr"]) + 1.0e-14)

    def test_cosine_stage_end_reaches_min_lr(self) -> None:
        sched, _ = _scheduler(steps_per_epoch=4)
        by_rollout = {entry["rollout_steps"]: entry for entry in STAGE_LRS}
        for stage in sched.stages:
            final_lr = sched.get_lr_for_global_step(stage.end_step - 1)
            self.assertAlmostEqual(final_lr, float(by_rollout[stage.rollout_steps]["min_lr"]))

    def test_backward_compatibility_old_warmup_cosine_config(self) -> None:
        params = YParams("configs/weather_dual_resolution.yaml", "raw", resolution_mode="2p5")
        self.assertEqual(params.lr_schedule_type, "warmup_cosine")
        self.assertIsNone(params.get("rollout_stage_lr_schedule", None))
        trainer = object.__new__(Trainer)
        trainer.lr_schedule_type = "warmup_cosine"
        trainer.epoch = 0
        trainer.base_lr = 1.0e-4
        trainer.min_lr = 3.0e-6
        trainer.warmup_epochs = 2
        trainer.warmup_start_factor = 0.1
        trainer.params = type("P", (), {"max_epochs": 50})()
        self.assertAlmostEqual(Trainer._lr_for_epoch(trainer, rollout_steps=1), 5.5e-5)

    def test_resume_state_matches_uninterrupted_sequence(self) -> None:
        full, _ = _scheduler(steps_per_epoch=2)
        for _ in range(full.get_stage_for_epoch(30).start_step + 3):
            full.step()
        state = full.state_dict()
        expected = []
        for _ in range(12):
            expected.append(full.get_last_lr()[0])
            full.step()

        resumed, _ = _scheduler(steps_per_epoch=2)
        resumed.load_state_dict(state)
        actual = []
        for _ in range(12):
            actual.append(resumed.get_last_lr()[0])
            resumed.step()
        self.assertEqual(actual, expected)

    def test_gradient_accumulation_advances_only_on_optimizer_steps(self) -> None:
        sched, _ = _scheduler(steps_per_epoch=10)
        microbatches = 9
        accumulation = 3
        for idx in range(microbatches):
            if (idx + 1) % accumulation == 0:
                sched.step()
        self.assertEqual(sched.global_step, 3)


if __name__ == "__main__":
    unittest.main()
