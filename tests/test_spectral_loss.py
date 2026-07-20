from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import YParams  # noqa: E402
from src.losses import low_frequency_spectral_loss, low_frequency_spectral_mask  # noqa: E402
from src.target_handling import TargetHandling  # noqa: E402
from src.trainer import Trainer  # noqa: E402


class _NullLogger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None


class SpectralLossModuleTest(unittest.TestCase):
    def test_fft_mask_shape_and_dc_handling(self) -> None:
        mask = low_frequency_spectral_mask(72, 144, lat_cutoff=8, lon_cutoff=16)
        self.assertEqual(tuple(mask.shape), (72, 73))
        self.assertTrue(bool(mask[0, 0]))
        self.assertTrue(bool(mask[8, 16]))
        self.assertTrue(bool(mask[71, 16]))
        self.assertFalse(bool(mask[9, 0]))
        self.assertFalse(bool(mask[0, 17]))
        no_dc = low_frequency_spectral_mask(72, 144, lat_cutoff=8, lon_cutoff=16, include_dc=False)
        self.assertFalse(bool(no_dc[0, 0]))
        self.assertTrue(bool(no_dc[1, 1]))

    def test_spectral_loss_zero_when_predictions_match(self) -> None:
        pred = torch.randn(2, 3, 8, 12)
        loss = low_frequency_spectral_loss(pred, pred.clone(), [0, 2], 3, 4)
        self.assertEqual(float(loss.item()), 0.0)

    def test_spectral_loss_positive_when_predictions_differ(self) -> None:
        pred = torch.ones(2, 3, 8, 12)
        target = torch.zeros_like(pred)
        loss = low_frequency_spectral_loss(pred, target, [0, 2], 3, 4)
        self.assertGreater(float(loss.item()), 0.0)

    def test_bf16_inputs_are_cast_before_fft(self) -> None:
        pred = torch.ones(2, 3, 8, 12, dtype=torch.bfloat16)
        target = torch.zeros_like(pred)
        loss = low_frequency_spectral_loss(pred, target, [1], 3, 4)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.dtype, torch.float32)

    def test_spectral_loss_backpropagates_to_selected_output(self) -> None:
        pred = torch.randn(2, 3, 8, 12, requires_grad=True)
        target = torch.zeros_like(pred)
        loss = low_frequency_spectral_loss(pred, target, [0, 2], 3, 4)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertGreater(float(pred.grad[:, [0, 2]].abs().sum().item()), 0.0)
        self.assertEqual(float(pred.grad[:, 1].abs().sum().item()), 0.0)


class SpectralVariableConfigTest(unittest.TestCase):
    def _params(self, spectral_loss: dict | None) -> SimpleNamespace:
        params = {
            "variable_metadata": {"path": None},
            "global_means_path": "",
            "global_stds_path": "",
            "experiment_dir": "",
        }
        if spectral_loss is not None:
            params["spectral_loss"] = spectral_loss
        return SimpleNamespace(**params)

    def _channel_names(self) -> list[str]:
        names = [f"var{i}" for i in range(67)]
        names[1] = "msl"
        names[8] = "u850"
        names[20] = "v850"
        names[32] = "t850"
        names[60] = "z500"
        names[66] = "orog"
        names[5] = "tisr"
        return names

    def _configure(self, spectral_loss: dict | None, out_channels: list[int] | None = None) -> Trainer:
        trainer = object.__new__(Trainer)
        trainer.device = torch.device("cpu")
        Trainer._configure_spectral_loss(
            trainer,
            self._params(spectral_loss),
            channel_names=self._channel_names(),
            out_channels=list(range(67)) if out_channels is None else out_channels,
        )
        return trainer

    def test_configured_variables_resolve_to_output_channel_indices(self) -> None:
        trainer = self._configure(
            {
                "enabled": True,
                "weight": 0.005,
                "variables": ["z500", "msl", "t850", "u850", "v850"],
                "lat_cutoff": 8,
                "lon_cutoff": 16,
                "include_dc": True,
                "apply_latitude_weight": True,
                "space": "normalized",
            }
        )
        self.assertTrue(trainer.spectral_loss_enabled)
        self.assertEqual(trainer.spectral_loss_channel_indices.tolist(), [60, 1, 32, 8, 20])
        self.assertEqual(trainer.spectral_loss_metadata["variables"][0]["canonical"], "z500")

    def test_missing_variable_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Could not resolve variable"):
            self._configure({"enabled": True, "variables": ["z500", "missing_variable"]})

    def test_orog_and_tisr_are_rejected_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not include orog"):
            self._configure({"enabled": True, "variables": ["orog"]})
        with self.assertRaisesRegex(ValueError, "must not include tisr"):
            self._configure({"enabled": True, "variables": ["tisr"]})

    def test_absent_spectral_loss_config_disables_feature(self) -> None:
        trainer = self._configure(None)
        self.assertFalse(trainer.spectral_loss_enabled)
        self.assertEqual(trainer.spectral_loss_metadata, {"enabled": False})

    def test_spectral_experiment_configs_load_with_expected_settings(self) -> None:
        params = YParams(
            str(PROJECT_ROOT / "configs" / "weather_dual_resolution_l3_hidden128_spectral_loss_fixed_orog.yaml"),
            "raw_l3_hidden128_spectral_loss_fixed_orog",
            resolution_mode="2p5",
        )
        self.assertEqual(params.hidden_dim, 128)
        self.assertEqual(params.k_neighbors, 8)
        self.assertEqual(params.graph_path, "graphs/graph_2p5_k8_hybrid_row_aware_L3_v3.pt")
        self.assertEqual(params.target_handling["copy_variables"], ["orog"])
        self.assertEqual(params.target_handling["exclude_loss_variables"], ["orog"])
        self.assertTrue(params.spectral_loss["enabled"])
        self.assertAlmostEqual(float(params.spectral_loss["weight"]), 0.005)
        self.assertEqual(params.spectral_loss["variables"], ["z500", "msl", "t850", "u850", "v850"])


class SpectralRolloutIntegrationTest(unittest.TestCase):
    class Adapter:
        def __init__(self, channels: int) -> None:
            self.channels = int(channels)

        def extract_two_steps(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return inp[:, : self.channels], inp[:, self.channels : 2 * self.channels]

    class TinyDeltaModel(torch.nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.output_channels = int(channels)
            self.adapter = SpectralRolloutIntegrationTest.Adapter(channels)
            self.delta = torch.nn.Parameter(torch.full((channels,), 0.1))

        def forward_steps(self, previous: torch.Tensor, current: torch.Tensor, aux_features=None) -> torch.Tensor:
            return current[:, : self.output_channels] + self.delta.view(1, -1, 1, 1)

    class RecordingBadOrogModel(torch.nn.Module):
        output_channels = 2

        def __init__(self) -> None:
            super().__init__()
            self.adapter = SpectralRolloutIntegrationTest.Adapter(2)
            self.bias = torch.nn.Parameter(torch.tensor(0.25))
            self.seen_currents: list[torch.Tensor] = []

        def forward_steps(self, previous: torch.Tensor, current: torch.Tensor, aux_features=None) -> torch.Tensor:
            self.seen_currents.append(current.detach().clone())
            out = current[:, :2].clone()
            out[:, 0] = out[:, 0] + self.bias
            out[:, 1] = -999.0
            return out

    def _tiny_trainer(self, channels: int = 2, rollout_steps: int = 2) -> Trainer:
        model = self.TinyDeltaModel(channels)
        trainer = object.__new__(Trainer)
        trainer.model = model
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-2)
        trainer.scheduler = None
        trainer.gscaler = None
        trainer.device = torch.device("cpu")
        trainer.pin_memory = False
        trainer.amp_enabled = False
        trainer.batch_size = 1
        trainer.gradient_accumulation_steps = 1
        trainer.effective_batch_size = 1
        trainer.max_gradient_norm = None
        trainer.max_train_batches = 2
        trainer.max_rollout_steps = rollout_steps
        trainer.log_timing_breakdown = False
        trainer.log_every_batches = 1
        trainer.rollout_mode = "fixed_full"
        trainer.fixed_train_rollout_steps = rollout_steps
        trainer.rollout_schedule = [rollout_steps]
        trainer.rollout_stage_epochs = [1]
        trainer.epoch = 0
        trainer.iters = 0
        trainer.load_only_current_rollout = False
        trainer._current_train_target_rollout_steps = rollout_steps
        trainer._epoch_lr_values = []
        trainer.lr_schedule_type = "none"
        trainer.graph_gradient_weight = 0.0
        trainer.loss_obj = lambda pred, target: torch.mean((pred - target) ** 2)
        trainer.rollout_loss_weights_name = "uniform"
        trainer.rollout_loss_weights_values = None
        trainer.feature_builder = None
        trainer.target_handler = None
        trainer.params = SimpleNamespace(lead_conditioning={"enabled": False})
        trainer.spectral_loss_enabled = True
        trainer.spectral_loss_weight = 0.005
        trainer.spectral_loss_channel_indices = torch.tensor([0], dtype=torch.long)
        trainer.spectral_loss_lat_cutoff = 2
        trainer.spectral_loss_lon_cutoff = 3
        trainer.spectral_loss_include_dc = True
        trainer.spectral_loss_apply_latitude_weight = True
        trainer.spectral_latitudes_rad = torch.linspace(math.pi / 2.0, -math.pi / 2.0, 4)
        return trainer

    def test_rollout_uses_model_delta_next_state_semantics(self) -> None:
        trainer = self._tiny_trainer(channels=1, rollout_steps=1)
        trainer.spectral_loss_enabled = False
        trainer.model.delta.data.fill_(2.0)
        inp = torch.zeros(1, 2, 1, 1)
        inp[:, 1] = 1.0
        target = torch.full((1, 1, 1, 1, 1), 3.0)
        loss, last_pred = Trainer._rollout_loss(trainer, inp, target, rollout_steps=1)
        self.assertAlmostEqual(float(loss.item()), 0.0)
        self.assertAlmostEqual(float(last_pred.item()), 3.0)

    def test_training_smoke_with_spectral_loss_is_finite(self) -> None:
        trainer = self._tiny_trainer(channels=2, rollout_steps=2)
        inp = torch.zeros(1, 4, 4, 6)
        target = torch.zeros(1, 2, 2, 4, 6)
        target[:, 0, :, :, :] = 0.1
        target[:, 1, :, :, :] = 0.2
        trainer.train_data_loader = [(inp, target), (inp, target)]
        _, logs = Trainer.train_one_epoch(trainer)
        self.assertEqual(int(logs["rollout_steps"]), 2)
        self.assertTrue(math.isfinite(float(logs["train_loss_total"])))
        self.assertIn("train_loss_grid", logs)
        self.assertIn("train_loss_spectral_raw", logs)
        self.assertIn("train_loss_spectral_weighted", logs)

    def test_disabled_spectral_rollout_matches_grid_loss(self) -> None:
        trainer = self._tiny_trainer(channels=2, rollout_steps=1)
        trainer.spectral_loss_enabled = False
        inp = torch.zeros(1, 4, 4, 6)
        target = torch.ones(1, 1, 2, 4, 6)
        loss, _, _, components = Trainer._rollout_loss(
            trainer,
            inp,
            target,
            rollout_steps=1,
            return_lead_losses=True,
            return_loss_components=True,
        )
        self.assertAlmostEqual(float(loss.item()), float(components["grid_loss"].item()))
        self.assertEqual(float(components["spectral_raw_loss"].item()), 0.0)

    def test_fixed_orography_is_applied_before_spectral_loss_and_next_state(self) -> None:
        model = self.RecordingBadOrogModel()
        handler = TargetHandling.from_params(
            SimpleNamespace(
                target_handling={
                    "enabled": True,
                    "copy_variables": ["orog"],
                    "exclude_loss_variables": ["orog"],
                    "known_future_variables": [],
                },
                variable_metadata={"path": None},
                global_means_path="",
                global_stds_path="",
                experiment_dir="",
            ),
            channel_names=["z500", "orog"],
            out_channels=[0, 1],
            logger=_NullLogger(),
        )
        trainer = self._tiny_trainer(channels=2, rollout_steps=2)
        trainer.model = model
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-2)
        trainer.target_handler = handler
        trainer.loss_channel_mask = handler.loss_channel_mask(2)
        trainer.loss_obj = lambda pred, target, channel_mask=None: torch.mean((pred - target) ** 2)
        trainer.spectral_loss_channel_indices = torch.tensor([0], dtype=torch.long)

        inp = torch.zeros(1, 4, 4, 6)
        inp[:, 3] = 42.0
        target = torch.zeros(1, 2, 2, 4, 6)
        target[:, :, 1] = 42.0
        _, last_pred = Trainer._rollout_loss(trainer, inp, target, rollout_steps=2)
        self.assertEqual(len(model.seen_currents), 2)
        self.assertTrue(torch.allclose(last_pred[:, 1], torch.full_like(last_pred[:, 1], 42.0)))
        self.assertTrue(torch.allclose(model.seen_currents[1][:, 1], torch.full_like(model.seen_currents[1][:, 1], 42.0)))


if __name__ == "__main__":
    unittest.main()
