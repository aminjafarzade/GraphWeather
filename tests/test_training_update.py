from __future__ import annotations

import csv
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import ClimateNetCDFDataset, DataConfig  # noqa: E402
from src.delta_stats import compute_delta_stats  # noqa: E402
from src.evaluator import GraphWeatherEvaluator, _accumulate_daily_climatology_chunk  # noqa: E402
from src.models import GraphWeatherModel  # noqa: E402
from src.trainer import Trainer, _assert_cuda_device_usable, _resolve_device, resolve_amp_dtype, warmup_cosine_lr  # noqa: E402
from scripts.train import _map_requested_device_to_visible  # noqa: E402
from scripts.evaluate_stage_rollout_progress import (  # noqa: E402
    _check_identical_metric_curves,
    _discover_checkpoints,
    _evaluate_checkpoints,
)


class AmpResolverTest(unittest.TestCase):
    def test_auto_selects_bf16_when_supported(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.is_bf16_supported",
            return_value=True,
        ):
            enabled, dtype, scaler = resolve_amp_dtype(True, "auto_bf16_fp16")
        self.assertTrue(enabled)
        self.assertIs(dtype, torch.bfloat16)
        self.assertFalse(scaler)

    def test_auto_falls_back_to_fp16_on_cuda(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.is_bf16_supported",
            return_value=False,
        ):
            enabled, dtype, scaler = resolve_amp_dtype(True, "auto_bf16_fp16")
        self.assertTrue(enabled)
        self.assertIs(dtype, torch.float16)
        self.assertTrue(scaler)

    def test_disables_amp_without_cuda(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            enabled, dtype, scaler = resolve_amp_dtype(True, "auto_bf16_fp16")
        self.assertFalse(enabled)
        self.assertIsNone(dtype)
        self.assertFalse(scaler)


class DeviceSelectionTest(unittest.TestCase):
    def test_resolve_device_maps_physical_id_under_cuda_visible_devices(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False), mock.patch(
            "torch.cuda.is_available",
            return_value=True,
        ), mock.patch("torch.cuda.device_count", return_value=1):
            device = _resolve_device("cuda:1", local_rank=0)
        self.assertEqual(str(device), "cuda:0")

    def test_bare_numeric_device_sets_visible_devices_when_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            mapped = _map_requested_device_to_visible("2")
            self.assertEqual(mapped, "cuda:0")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")

    def test_bare_numeric_device_maps_existing_visible_physical_id(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3"}, clear=True):
            mapped = _map_requested_device_to_visible("3")
        self.assertEqual(mapped, "cuda:1")

    def test_cuda_ordinal_is_left_for_trainer_resolution(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=True):
            mapped = _map_requested_device_to_visible("cuda:1")
        self.assertEqual(mapped, "cuda:1")

    def test_cuda_kernel_image_error_is_rewritten_with_device_diagnostics(self) -> None:
        with mock.patch("torch.cuda.device", return_value=contextlib.nullcontext()), mock.patch(
            "torch.empty",
            side_effect=RuntimeError("CUDA error: no kernel image is available for execution on the device"),
        ), mock.patch("torch.cuda.current_device", return_value=0), mock.patch(
            "torch.cuda.get_device_name",
            return_value="Test GPU",
        ), mock.patch(
            "torch.cuda.get_device_capability",
            return_value=(3, 7),
        ), mock.patch(
            "torch.cuda.get_arch_list",
            return_value=["sm_70", "sm_80"],
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot run CUDA kernels with this PyTorch build"):
                _assert_cuda_device_usable(torch.device("cuda:0"))


class WarmupCosineScheduleTest(unittest.TestCase):
    def test_warmup_cosine_values(self) -> None:
        kwargs = {
            "base_lr": 1.0e-4,
            "min_lr": 3.0e-6,
            "warmup_epochs": 2,
            "warmup_start_factor": 0.1,
            "max_epochs": 50,
        }
        epoch1 = warmup_cosine_lr(0, **kwargs)
        epoch2 = warmup_cosine_lr(1, **kwargs)
        epoch50 = warmup_cosine_lr(49, **kwargs)

        self.assertGreater(epoch1, 1.0e-5)
        self.assertLess(epoch1, 1.0e-4)
        self.assertAlmostEqual(epoch2, 1.0e-4)
        self.assertAlmostEqual(epoch50, 3.0e-6)

    def test_warmup_cosine_warns_and_ignores_lr_by_rollout(self) -> None:
        trainer = object.__new__(Trainer)
        trainer.lr_schedule_type = "warmup_cosine"
        trainer.params = SimpleNamespace(lr_by_rollout={1: 1.0e-4, 10: 3.0e-5})
        with self.assertLogs(level="WARNING") as captured:
            Trainer._warn_lr_config_conflicts(trainer)
        self.assertIn("lr_schedule_type=warmup_cosine: ignoring lr_by_rollout.", "\n".join(captured.output))

    def test_warmup_cosine_lr_depends_on_epoch_not_micro_batches(self) -> None:
        trainer = object.__new__(Trainer)
        trainer.lr_schedule_type = "warmup_cosine"
        trainer.base_lr = 1.0e-4
        trainer.min_lr = 3.0e-6
        trainer.warmup_epochs = 2
        trainer.warmup_start_factor = 0.1
        trainer.params = SimpleNamespace(max_epochs=50)
        trainer.epoch = 0

        epoch0_values = [Trainer._lr_for_epoch(trainer, rollout_steps=1) for _ in range(8)]
        self.assertTrue(all(value == epoch0_values[0] for value in epoch0_values))
        trainer.epoch = 1
        self.assertNotEqual(Trainer._lr_for_epoch(trainer, rollout_steps=1), epoch0_values[0])

    def test_rollout_stage_lr_still_works(self) -> None:
        trainer = object.__new__(Trainer)
        trainer.lr_schedule_type = "rollout_stage"
        trainer.warmup_epochs = 0
        trainer.warmup_start_factor = 0.1
        trainer.epoch = 20
        trainer.params = SimpleNamespace(lr=1.0e-4, lr_by_rollout={4: 7.0e-5})
        self.assertAlmostEqual(Trainer._lr_for_epoch(trainer, rollout_steps=4), 7.0e-5)

    def test_resume_recomputes_warmup_cosine_lr_from_epoch(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=9.0e-5)
        graph_meta = {
            "resolution_mode": "test",
            "graph_connectivity_strategy": "unit",
            "graph_format_version": 1,
            "graph_k": 2,
            "graph_coordinate_hash": "abc",
        }
        trainer = object.__new__(Trainer)
        trainer.model = model
        trainer.optimizer = optimizer
        trainer.scheduler = None
        trainer.device = torch.device("cpu")
        trainer.graph_topology_metadata = graph_meta
        trainer.params = SimpleNamespace(resolution_mode="test", max_epochs=50)
        trainer.use_delta_normalization = False
        trainer.lr_schedule_type = "warmup_cosine"
        trainer.base_lr = 1.0e-4
        trainer.min_lr = 3.0e-6
        trainer.warmup_epochs = 2
        trainer.warmup_start_factor = 0.1
        trainer.best_score_global = float("inf")
        trainer.best_score_by_stage = {}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.tar"
            torch.save(
                {
                    "iters": 7,
                    "epoch": 2,
                    "model_state": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metadata": graph_meta,
                },
                path,
            )
            Trainer.restore_checkpoint(trainer, str(path))

        expected = warmup_cosine_lr(2, 1.0e-4, 3.0e-6, 2, 0.1, 50)
        self.assertEqual(trainer.start_epoch, 2)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], expected)


class DataHorizonTest(unittest.TestCase):
    def _make_dataset(self, root: Path, rollout_steps: int) -> ClimateNetCDFDataset:
        try:
            import netCDF4 as nc
        except ModuleNotFoundError:
            self.skipTest("netCDF4 is not installed")

        data_dir = root / "data"
        data_dir.mkdir(parents=True)
        with nc.Dataset(data_dir / "sample.nc", "w") as ds:
            ds.createDimension("time", 12)
            ds.createDimension("channel", 3)
            ds.createDimension("lat", 2)
            ds.createDimension("lon", 2)
            fields = ds.createVariable("fields", "f4", ("time", "channel", "lat", "lon"))
            values = np.arange(12 * 3 * 2 * 2, dtype=np.float32).reshape(12, 3, 2, 2)
            fields[:] = values

        mean_path = root / "mean.npy"
        std_path = root / "std.npy"
        np.save(mean_path, np.zeros((3,), dtype=np.float32))
        np.save(std_path, np.ones((3,), dtype=np.float32))

        cfg = DataConfig(
            dt=1,
            n_history=1,
            in_channels=[0, 1, 2],
            out_channels=[0, 1, 2],
            normalize=True,
            global_means_path=str(mean_path),
            global_stds_path=str(std_path),
            rollout_steps=rollout_steps,
            batch_size=1,
            num_workers=0,
            resolution_mode="test",
            expected_grid_shape=[2, 2],
        )
        return ClimateNetCDFDataset(cfg, str(data_dir), train=False)

    def test_dataset_returns_requested_target_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target1 = self._make_dataset(root / "s1", rollout_steps=1)[0][1]
            target4 = self._make_dataset(root / "s4", rollout_steps=4)[0][1]
            target10 = self._make_dataset(root / "s10", rollout_steps=10)[0][1]
        self.assertEqual(tuple(target1.shape), (3, 2, 2))
        self.assertEqual(tuple(target4.shape), (4, 3, 2, 2))
        self.assertEqual(tuple(target10.shape), (10, 3, 2, 2))


class DeltaNormalizationTest(unittest.TestCase):
    def _bare_model(self) -> GraphWeatherModel:
        model = object.__new__(GraphWeatherModel)
        torch.nn.Module.__init__(model)
        model.output_channels = 2
        model.use_delta_normalization = True
        model.delta_norm_center = False
        model.delta_norm_eps = 1.0e-6
        model.register_buffer("delta_mean", torch.tensor([10.0, 20.0]))
        model.register_buffer("delta_std", torch.tensor([2.0, 3.0]))
        return model

    def test_delta_conversion_without_centering(self) -> None:
        model = self._bare_model()
        delta_hat = torch.tensor([[[1.0, -2.0]]])
        converted = model.denormalize_delta_nodes(delta_hat)
        self.assertTrue(torch.allclose(converted, torch.tensor([[[2.0, -6.0]]])))

    def test_delta_conversion_with_centering(self) -> None:
        model = self._bare_model()
        model.delta_norm_center = True
        delta_hat = torch.tensor([[[1.0, -2.0]]])
        converted = model.denormalize_delta_nodes(delta_hat)
        self.assertTrue(torch.allclose(converted, torch.tensor([[[12.0, 14.0]]])))

    def test_delta_conversion_disabled_is_old_behavior(self) -> None:
        model = self._bare_model()
        model.use_delta_normalization = False
        delta_hat = torch.tensor([[[1.0, -2.0]]])
        self.assertTrue(torch.equal(model.denormalize_delta_nodes(delta_hat), delta_hat))

    def test_compute_delta_stats_from_normalized_tensors(self) -> None:
        inp = torch.zeros(2, 4, 1, 1)
        inp[:, 2:, :, :] = torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]])
        target = torch.zeros(2, 1, 2, 1, 1)
        target[:, 0, :, :, :] = torch.tensor([[[[2.0]], [[5.0]]], [[[4.0]], [[7.0]]]])
        loader = DataLoader(TensorDataset(inp, target), batch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "delta.npz"
            payload = compute_delta_stats(loader, path, output_channels=2, n_history=1, resolution_mode="test")
            self.assertTrue(path.exists())
            self.assertTrue(np.allclose(payload["delta_mean"], np.asarray([1.0, 3.0], dtype=np.float32)))


class SinglePassValidationTest(unittest.TestCase):
    def test_single_pass_prefix_and_final_metrics(self) -> None:
        class Adapter:
            def extract_two_steps(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return inp[:, :1], inp[:, 1:2]

        class Model:
            output_channels = 1

            def __init__(self) -> None:
                self.adapter = Adapter()
                self.calls = 0

            def eval(self) -> None:
                return None

            def forward_steps(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                return current[:, :1] + 1.0

        trainer = object.__new__(Trainer)
        trainer.model = Model()
        trainer.device = torch.device("cpu")
        trainer.pin_memory = False
        trainer.amp_enabled = False
        trainer.max_rollout_steps = 2
        trainer.max_valid_batches = None
        trainer.graph_gradient_weight = 0.0
        trainer.loss_obj = lambda pred, target: torch.mean((pred - target) ** 2)
        trainer.valid_data_loader = [
            (
                torch.tensor([[[[0.0]], [[0.0]]]]),
                torch.tensor([[[[[1.0]]], [[[3.0]]]]]),
            )
        ]

        logs = Trainer.validate_multi_horizon_single_pass(trainer, [1, 2], valid_rollout_steps=2)
        self.assertEqual(trainer.model.calls, 2)
        self.assertAlmostEqual(logs["valid_S1"], 0.0)
        self.assertAlmostEqual(logs["valid_S2"], 0.5)
        self.assertAlmostEqual(logs["valid_S2_final"], 1.0)
        self.assertAlmostEqual(logs["persistence_S2"], 5.0)
        self.assertAlmostEqual(logs["skill_S2"], 0.9)

    def test_training_validation_many_horizons_no_stage_labels(self) -> None:
        class Adapter:
            def extract_two_steps(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return inp[:, :1], inp[:, 1:2]

        class Model:
            output_channels = 1

            def __init__(self) -> None:
                self.adapter = Adapter()
                self.calls = 0

            def eval(self) -> None:
                return None

            def forward_steps(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                return current[:, :1] + 1.0

        target = torch.arange(1, 11, dtype=torch.float32).view(1, 10, 1, 1, 1)
        trainer = object.__new__(Trainer)
        trainer.model = Model()
        trainer.device = torch.device("cpu")
        trainer.pin_memory = False
        trainer.amp_enabled = False
        trainer.max_rollout_steps = 10
        trainer.max_valid_batches = None
        trainer.graph_gradient_weight = 0.0
        trainer.loss_obj = lambda pred, target: torch.mean((pred - target) ** 2)
        trainer.valid_data_loader = [(torch.tensor([[[[0.0]], [[0.0]]]]), target)]

        logs = Trainer.validate_multi_horizon_single_pass(
            trainer,
            [1, 2, 4, 6, 8, 10],
            valid_rollout_steps=10,
        )

        self.assertEqual(trainer.model.calls, 10)
        for horizon in (1, 2, 4, 6, 8, 10):
            self.assertIn(f"valid_S{horizon}", logs)
            self.assertIn(f"valid_S{horizon}_final", logs)
        self.assertFalse(any("trained" in key.lower() for key in logs))


class LeadOutputTest(unittest.TestCase):
    def test_saved_metric_csv_starts_at_lead_one(self) -> None:
        evaluator = object.__new__(GraphWeatherEvaluator)
        evaluator.cfg = SimpleNamespace(out_channels=[0])
        evaluator.variable_names = ["x"]
        evaluator.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
        metrics = {
            "rmse": np.arange(10, dtype=np.float64).reshape(10, 1),
            "acc": np.ones((10, 1), dtype=np.float64),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            GraphWeatherEvaluator._save_metrics(evaluator, metrics, 10, out)
            with open(out / "rollout_rmse.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["lead_time"], "1")
        self.assertEqual(rows[-1]["lead_time"], "10")


class ClimatologyAccumulationTest(unittest.TestCase):
    def test_vectorized_chunk_matches_timestep_loop(self) -> None:
        n_days = 5
        chunk = np.arange(8 * 2 * 2 * 3, dtype=np.float32).reshape(8, 2, 2, 3)
        fast_sum = np.zeros((n_days, 2, 2, 3), dtype=np.float64)
        fast_count = np.zeros((n_days,), dtype=np.int64)
        slow_sum = np.zeros_like(fast_sum)
        slow_count = np.zeros_like(fast_count)

        _accumulate_daily_climatology_chunk(fast_sum, fast_count, chunk, start_time_idx=3, n_days=n_days)
        for local_idx, time_idx in enumerate(range(3, 3 + chunk.shape[0])):
            day = time_idx % n_days
            slow_sum[day] += chunk[local_idx]
            slow_count[day] += 1

        np.testing.assert_allclose(fast_sum, slow_sum)
        np.testing.assert_array_equal(fast_count, slow_count)


class StageWiseCheckpointEvaluationTest(unittest.TestCase):
    def test_discovers_stage_checkpoints_and_warns_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save({"metadata": {"epoch": 1, "train_rollout_steps": 1}}, root / "best_ckpt_S1.tar")
            torch.save({"metadata": {"epoch": 2, "train_rollout_steps": 2}}, root / "best_ckpt_S2.tar")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                items = _discover_checkpoints(root, [1, 2, 4], include_global_best=False)

        self.assertEqual([Path(item["checkpoint_path"]).name for item in items], ["best_ckpt_S1.tar", "best_ckpt_S2.tar"])
        self.assertEqual([item["label"] for item in items], ["trained S=1", "trained S=2"])
        self.assertIn("WARNING: missing best_ckpt_S4.tar, skipping trained S=4 curve.", stdout.getvalue())

    def test_evaluates_separate_checkpoints_with_same_fixed_horizon(self) -> None:
        class FakeEvaluator:
            def __init__(self) -> None:
                self.loaded: list[str] = []
                self.evaluated: list[tuple[str, int]] = []
                self.model = None

            def _load_model(self, height: int, width: int, checkpoint_path: str) -> object:
                self.loaded.append(Path(checkpoint_path).name)
                return object()

            def _evaluate_loaded_model_rollout(
                self,
                fields,
                total_t: int,
                height: int,
                width: int,
                fixed_rollout_steps: int,
                climatology_norm,
                lat_weights,
                lat_weights_np,
                label: str,
            ) -> dict[str, np.ndarray]:
                self.evaluated.append((label, fixed_rollout_steps))
                offset = 0.0 if label == "trained S=1" else 10.0
                return {
                    "rmse": (np.arange(fixed_rollout_steps, dtype=np.float64) + offset).reshape(fixed_rollout_steps, 1),
                    "acc": np.ones((fixed_rollout_steps, 1), dtype=np.float64),
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt1 = root / "best_ckpt_S1.tar"
            ckpt2 = root / "best_ckpt_S2.tar"
            torch.save({"metadata": {"epoch": 1, "train_rollout_steps": 1}}, ckpt1)
            torch.save({"metadata": {"epoch": 2, "train_rollout_steps": 2}}, ckpt2)
            items = [
                {"key": "trained_S1", "label": "trained S=1", "stage": 1, "checkpoint_path": str(ckpt1)},
                {"key": "trained_S2", "label": "trained S=2", "stage": 2, "checkpoint_path": str(ckpt2)},
            ]
            variables = [{"canonical_name": "z500", "variable_idx": 0}]
            evaluator = FakeEvaluator()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                payloads = _evaluate_checkpoints(
                    evaluator,
                    items,
                    variables,
                    fixed_steps=10,
                    fields=None,
                    total_t=20,
                    height=1,
                    width=1,
                    climatology_norm=None,
                    lat_weights=None,
                    lat_weights_np=np.ones((1,), dtype=np.float64),
                )

        self.assertEqual(evaluator.loaded, ["best_ckpt_S1.tar", "best_ckpt_S2.tar"])
        self.assertEqual(evaluator.evaluated, [("trained S=1", 10), ("trained S=2", 10)])
        self.assertEqual(payloads["trained_S1"]["label"], "trained S=1")
        self.assertEqual(payloads["trained_S2"]["label"], "trained S=2")
        self.assertEqual(payloads["trained_S1"]["metrics"]["z500"]["rmse_by_lead"][0], 0.0)
        self.assertEqual(payloads["trained_S2"]["metrics"]["z500"]["rmse_by_lead"][0], 10.0)
        self.assertIn("Evaluating checkpoint: best_ckpt_S1.tar", stdout.getvalue())
        self.assertIn("Fixed evaluation rollout: 10 days", stdout.getvalue())

    def test_identical_curve_warning(self) -> None:
        variables = [{"canonical_name": "z500"}]
        metric = {
            "z500": {
                "rmse_by_lead": [1.0, 2.0],
                "acc_by_lead": [0.5, 0.6],
            }
        }
        payloads = {
            "trained_S1": {"metrics": metric},
            "trained_S2": {"metrics": metric},
        }
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            _check_identical_metric_curves(payloads, variables)
        self.assertIn(
            "WARNING: all stage curves are identical. Check whether the same checkpoint or cached rollout result was reused.",
            stdout.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
