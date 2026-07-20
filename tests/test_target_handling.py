from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.losses import LatitudeWeightedMSE  # noqa: E402
from src.evaluator import EvalConfig, GraphWeatherEvaluator  # noqa: E402
from src.target_handling import TargetHandling, TargetHandlingSettings, target_handling_metadata_matches  # noqa: E402
from src.trainer import Trainer  # noqa: E402
from src.config import DEFAULT_TARGET_HANDLING, YParams, normalize_target_handling_config_dict  # noqa: E402


class _NullLogger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None


def _params(enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        extra_features={"enabled": False},
        target_handling={
            "enabled": enabled,
            "copy_variables": ["orog"] if enabled else [],
            "known_future_variables": ["tisr"] if enabled else [],
            "exclude_loss_variables": ["orog", "tisr"] if enabled else [],
        },
        variable_metadata={"path": None},
        global_means_path="",
        global_stds_path="",
        experiment_dir="",
    )


def _fixed_orog_params(enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        extra_features={"enabled": False},
        target_handling={
            "enabled": enabled,
            "copy_variables": ["orog"] if enabled else [],
            "known_future_variables": [],
            "exclude_loss_variables": ["orog"] if enabled else [],
        },
        variable_metadata={"path": None},
        global_means_path="",
        global_stds_path="",
        experiment_dir="",
    )


def _handler(enabled: bool = True) -> TargetHandling:
    return TargetHandling.from_params(
        _params(enabled=enabled),
        channel_names=["tisr", "orog", "t2m"],
        out_channels=[0, 1, 2],
        logger=_NullLogger(),
    )


class TargetHandlingTest(unittest.TestCase):
    def test_apply_copies_orog_and_uses_future_tisr_for_each_lead(self) -> None:
        handler = _handler()
        current = torch.zeros(1, 3, 1, 1)
        current[:, 1] = 42.0
        initial = current.clone()
        target = torch.zeros(1, 10, 3, 1, 1)
        for lead in range(1, 11):
            target[:, lead - 1, 0] = float(lead)

        for lead in range(1, 11):
            pred = torch.full_like(current, -1000.0)
            pred = handler.apply(
                pred_next=pred,
                current_state=current,
                initial_state=initial,
                target_sequence=target,
                lead=lead,
            )
            self.assertTrue(torch.equal(pred[:, 1], initial[:, 1]))
            self.assertTrue(torch.equal(pred[:, 0], target[:, lead - 1, 0]))
            next_step = current.clone()
            next_step[:, :3] = pred
            current = next_step

    def test_fixed_orography_copy_leaves_other_channels_unchanged_without_future_targets(self) -> None:
        handler = TargetHandling.from_params(
            _fixed_orog_params(),
            channel_names=["tisr", "orog", "t2m"],
            out_channels=[0, 1, 2],
            logger=_NullLogger(),
        )
        current = torch.zeros(1, 3, 1, 1)
        current[:, 1] = 42.0
        initial = current.clone()
        pred = torch.zeros_like(current)
        pred[:, 0] = 3.0
        pred[:, 1] = -999.0
        pred[:, 2] = 5.0
        out = handler.apply(pred_next=pred, current_state=current, initial_state=initial, lead=1)
        self.assertEqual(float(out[:, 0].item()), 3.0)
        self.assertEqual(float(out[:, 1].item()), 42.0)
        self.assertEqual(float(out[:, 2].item()), 5.0)

    def test_loss_mask_excludes_orog_and_tisr_only(self) -> None:
        handler = _handler()
        mask = handler.loss_channel_mask(3)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.tolist(), [0.0, 0.0, 1.0])
        loss_fn = LatitudeWeightedMSE(torch.zeros(2))
        target = torch.zeros(1, 3, 2, 2)
        pred = target.clone()
        pred[:, 0] += 100000.0
        pred[:, 1] += 100000.0
        self.assertAlmostEqual(float(loss_fn(pred, target, channel_mask=mask).item()), 0.0)
        pred = target.clone()
        pred[:, 2] += 1.0
        self.assertGreater(float(loss_fn(pred, target, channel_mask=mask).item()), 0.0)

    def test_fixed_orography_loss_mask_excludes_orog_only(self) -> None:
        handler = TargetHandling.from_params(
            _fixed_orog_params(),
            channel_names=["tisr", "orog", "t2m"],
            out_channels=[0, 1, 2],
            logger=_NullLogger(),
        )
        mask = handler.loss_channel_mask(3)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.tolist(), [1.0, 0.0, 1.0])
        loss_fn = LatitudeWeightedMSE(torch.zeros(2))
        target = torch.zeros(1, 3, 2, 2)
        pred = target.clone()
        pred[:, 1] += 100000.0
        self.assertAlmostEqual(float(loss_fn(pred, target, channel_mask=mask).item()), 0.0)
        pred = target.clone()
        pred[:, 2] += 1.0
        self.assertGreater(float(loss_fn(pred, target, channel_mask=mask).item()), 0.0)

    def test_disabled_target_handling_preserves_old_behavior(self) -> None:
        handler = _handler(enabled=False)
        pred = torch.ones(1, 3, 1, 1)
        current = torch.zeros_like(pred)
        target = torch.zeros(1, 1, 3, 1, 1)
        self.assertIs(handler.apply(pred_next=pred, current_state=current, target_sequence=target, lead=1), pred)
        self.assertIsNone(handler.loss_channel_mask(3))
        self.assertEqual(handler.metadata, {"enabled": False, "copy_variables": [], "known_future_variables": [], "exclude_loss_variables": []})

    def test_missing_enabled_variable_raises_clear_error(self) -> None:
        params = _params(enabled=True)
        with self.assertRaisesRegex(
            ValueError,
            'target_handling requested known_future_variables=\\["tisr"\\], but variable tisr was not found in channel list',
        ):
            TargetHandling.from_params(
                params,
                channel_names=["orog", "t2m"],
                out_channels=[0, 1],
                logger=_NullLogger(),
            )

    def test_checkpoint_target_handling_metadata_mismatch_is_clear(self) -> None:
        handler = _handler()
        ok, reason = target_handling_metadata_matches(handler.metadata, {"target_handling": handler.metadata})
        self.assertTrue(ok)
        self.assertEqual(reason, "")
        ok, reason = target_handling_metadata_matches(handler.metadata, {})
        self.assertFalse(ok)
        self.assertIn("target_handling mismatch", reason)

    def test_trainer_checkpoint_target_handling_mismatch_guard(self) -> None:
        trainer = object.__new__(Trainer)
        trainer.target_handler = _handler()
        with self.assertRaisesRegex(RuntimeError, "Checkpoint target-handling configuration mismatch"):
            Trainer._validate_checkpoint_target_handling(trainer, {})


class TrainerTargetHandlingRolloutTest(unittest.TestCase):
    class Adapter:
        def extract_two_steps(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return inp[:, :3], inp[:, 3:6]

    class TinyModel:
        output_channels = 3
        training = False  # plain stub, not an nn.Module

        def __init__(self) -> None:
            self.adapter = TrainerTargetHandlingRolloutTest.Adapter()

        def forward_steps(self, previous: torch.Tensor, current: torch.Tensor, aux_features=None) -> torch.Tensor:
            out = current[:, :3].clone()
            out[:, 0] = -999.0
            out[:, 1] = 999.0
            return out

    def test_trainer_rollout_target_handling_with_extra_features_disabled(self) -> None:
        handler = _handler()
        trainer = object.__new__(Trainer)
        trainer.model = self.TinyModel()
        trainer.graph_gradient_weight = 0.0
        trainer.loss_obj = LatitudeWeightedMSE(torch.zeros(1))
        trainer.loss_channel_mask = handler.loss_channel_mask(3)
        trainer.target_handler = handler
        trainer.rollout_loss_weights_name = "uniform"
        trainer.rollout_loss_weights_values = None
        trainer.feature_builder = None

        inp = torch.zeros(1, 6, 1, 1)
        inp[:, 4] = 42.0
        inp[:, 5] = 5.0
        target = torch.zeros(1, 10, 3, 1, 1)
        target[:, :, 1] = 42.0
        target[:, :, 2] = 5.0
        for lead in range(1, 11):
            target[:, lead - 1, 0] = float(lead)

        loss, last_pred = Trainer._rollout_loss(trainer, inp, target, rollout_steps=10)
        self.assertAlmostEqual(float(loss.item()), 0.0)
        self.assertTrue(torch.equal(last_pred[:, 1], torch.full((1, 1, 1), 42.0)))
        self.assertTrue(torch.equal(last_pred[:, 0], torch.full((1, 1, 1), 10.0)))


class EvalTargetOverrideConfigTest(unittest.TestCase):
    def test_eval_target_override_defaults_to_disabled(self) -> None:
        cfg = EvalConfig.from_params(SimpleNamespace())
        self.assertFalse(cfg.eval_target_override)
        self.assertEqual(cfg.eval_copy_variables, [])
        self.assertEqual(cfg.eval_known_future_variables, [])
        self.assertEqual(cfg.eval_exclude_loss_variables, [])

    def test_eval_target_override_defaults_to_orog_tisr_when_enabled(self) -> None:
        cfg = EvalConfig.from_params(SimpleNamespace(eval_target_override=True))
        self.assertTrue(cfg.eval_target_override)
        self.assertEqual(cfg.eval_copy_variables, ["orog"])
        self.assertEqual(cfg.eval_known_future_variables, ["tisr"])
        self.assertEqual(cfg.eval_exclude_loss_variables, ["orog", "tisr"])

    def test_eval_target_override_allows_explicit_empty_known_future_variables(self) -> None:
        cfg = EvalConfig.from_params(
            SimpleNamespace(
                eval_target_override=True,
                eval_copy_variables=["orog"],
                eval_known_future_variables=[],
                eval_exclude_loss_variables=["orog"],
            )
        )
        self.assertTrue(cfg.eval_target_override)
        self.assertEqual(cfg.eval_copy_variables, ["orog"])
        self.assertEqual(cfg.eval_known_future_variables, [])
        self.assertEqual(cfg.eval_exclude_loss_variables, ["orog"])

    def test_eval_target_override_handler_is_metadata_independent(self) -> None:
        settings = TargetHandlingSettings(
            enabled=True,
            copy_variables=("orog",),
            known_future_variables=("tisr",),
            exclude_loss_variables=("orog", "tisr"),
        )
        handler = TargetHandling.from_settings(
            settings,
            _params(enabled=False),
            channel_names=["tisr", "orog", "t2m"],
            out_channels=[0, 1, 2],
            logger=_NullLogger(),
            context_name="evaluation target override",
        )
        pred = torch.zeros(1, 3, 1, 1)
        current = torch.zeros_like(pred)
        current[:, 1] = 7.0
        initial = current.clone()
        target = torch.zeros(1, 5, 3, 1, 1)
        target[:, 2, 0] = 3.0
        out = handler.apply(
            pred_next=pred,
            current_state=current,
            initial_state=initial,
            target_sequence=target,
            lead=3,
        )
        self.assertEqual(float(out[:, 1].item()), 7.0)
        self.assertEqual(float(out[:, 0].item()), 3.0)
        ok, reason = target_handling_metadata_matches({"enabled": False}, {})
        self.assertTrue(ok, reason)


class EvalSensitivityModeTest(unittest.TestCase):
    def _evaluator(self, mode: str) -> GraphWeatherEvaluator:
        evaluator = object.__new__(GraphWeatherEvaluator)
        evaluator.cfg = SimpleNamespace(
            eval_sensitivity_mode=mode,
            eval_sensitivity_noise_seed=123,
            debug_eval_sensitivity=False,
            eval_sensitivity_debug_samples=3,
        )
        evaluator.logger = _NullLogger()
        evaluator.sensitivity_channels = {"orog": 1, "tisr": 0}
        evaluator._sensitivity_debug = None
        evaluator._sensitivity_logged_fallbacks = set()
        return evaluator

    def test_eval_sensitivity_defaults_to_normal(self) -> None:
        cfg = EvalConfig.from_params(SimpleNamespace())
        self.assertEqual(cfg.eval_sensitivity_mode, "normal")
        self.assertEqual(cfg.eval_sensitivity_noise_seed, 123)
        self.assertFalse(cfg.debug_eval_sensitivity)

    def test_eval_sensitivity_rejects_unknown_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported eval_sensitivity_mode"):
            EvalConfig.from_params(SimpleNamespace(eval_sensitivity_mode="bad_mode"))

    def test_tisr_zero_changes_only_tisr_channel(self) -> None:
        evaluator = self._evaluator("tisr_zero")
        pred = torch.ones(1, 3, 2, 4)
        current = pred.clone()
        initial = pred.clone()
        target = torch.full_like(pred, 5.0)
        out = GraphWeatherEvaluator._apply_sensitivity_mode(
            evaluator,
            pred_norm=pred,
            current_state=current,
            initial_state=initial,
            target_norm=target,
            lead=1,
            ic=10,
            debug=False,
        )
        self.assertTrue(torch.equal(out[:, 0], torch.zeros_like(out[:, 0])))
        self.assertTrue(torch.equal(out[:, 1:], pred[:, 1:]))

    def test_override_uses_initial_orog_and_future_tisr(self) -> None:
        evaluator = self._evaluator("override_orog_tisr")
        pred = torch.zeros(1, 3, 2, 4)
        initial = torch.zeros_like(pred)
        initial[:, 1] = 7.0
        target = torch.zeros_like(pred)
        target[:, 0] = 3.0
        out = GraphWeatherEvaluator._apply_sensitivity_mode(
            evaluator,
            pred_norm=pred,
            current_state=pred,
            initial_state=initial,
            target_norm=target,
            lead=2,
            ic=10,
            debug=False,
        )
        self.assertTrue(torch.equal(out[:, 1], torch.full_like(out[:, 1], 7.0)))
        self.assertTrue(torch.equal(out[:, 0], torch.full_like(out[:, 0], 3.0)))

    def test_random_is_reproducible_for_same_seed_ic_and_lead(self) -> None:
        pred = torch.zeros(1, 3, 2, 4)
        target = torch.zeros_like(pred)
        first = GraphWeatherEvaluator._apply_sensitivity_mode(
            self._evaluator("orog_random"),
            pred_norm=pred,
            current_state=pred,
            initial_state=pred,
            target_norm=target,
            lead=1,
            ic=10,
            debug=False,
        )
        second = GraphWeatherEvaluator._apply_sensitivity_mode(
            self._evaluator("orog_random"),
            pred_norm=pred,
            current_state=pred,
            initial_state=pred,
            target_norm=target,
            lead=1,
            ic=10,
            debug=False,
        )
        self.assertTrue(torch.equal(first[:, 1], second[:, 1]))
        self.assertGreater(float(first[:, 1].std(unbiased=False).item()), 0.0)
        self.assertTrue(torch.equal(first[:, 0], pred[:, 0]))

    def test_shuffle_batch_one_uses_longitude_roll_fallback(self) -> None:
        evaluator = self._evaluator("orog_shuffle")
        pred = torch.zeros(1, 3, 1, 6)
        initial = torch.zeros_like(pred)
        initial[:, 1, :, :] = torch.arange(6, dtype=torch.float32).view(1, 1, 1, 6)
        out = GraphWeatherEvaluator._apply_sensitivity_mode(
            evaluator,
            pred_norm=pred,
            current_state=pred,
            initial_state=initial,
            target_norm=pred,
            lead=1,
            ic=10,
            debug=False,
        )
        self.assertTrue(torch.equal(out[:, 1], torch.roll(initial[:, 1], shifts=2, dims=-1)))
        self.assertTrue(torch.equal(out[:, 0], pred[:, 0]))


class TargetHandlingConfigTest(unittest.TestCase):
    def test_missing_target_handling_defaults_to_fixed_orography(self) -> None:
        resolved = normalize_target_handling_config_dict({"extra_features": {"enabled": False}})
        self.assertEqual(resolved["target_handling"], DEFAULT_TARGET_HANDLING)

    def test_active_future_configs_enable_fixed_orography(self) -> None:
        specs = [
            ("weather_dual_resolution.yaml", "raw_l3"),
            ("weather_dual_resolution_l3.yaml", "raw_l3"),
            ("weather_dual_resolution_l3_hidden128.yaml", "raw_l3_hidden128"),
            ("weather_dual_resolution_l3_hidden160.yaml", "raw_l3_hidden160"),
            ("weather_dual_resolution_l3_blocks3.yaml", "raw_l3_blocks3"),
            ("weather_dual_resolution_l3_heavy_unet.yaml", "raw_l3_heavy_unet"),
            ("weather_dual_resolution_l3_full_rollout.yaml", "raw_l3_full_rollout"),
            ("weather_dual_resolution_l3_stage_warmup_cosine.yaml", "raw_l3_stage_warmup_cosine"),
            (
                "weather_dual_resolution_l3_hidden128_dense_l3k24_fixed_orog.yaml",
                "raw_l3_hidden128_dense_l3k24_fixed_orog",
            ),
            (
                "weather_dual_resolution_l3_hidden128_scalar_gated_skip_fixed_orog.yaml",
                "raw_l3_hidden128_scalar_gated_skip_fixed_orog",
            ),
            (
                "weather_dual_resolution_l3_hidden128_scalar_gated_pooling_fixed_orog.yaml",
                "raw_l3_hidden128_scalar_gated_pooling_fixed_orog",
            ),
            (
                "weather_dual_resolution_l3_hidden128_lead_conditioned_fixed_orog.yaml",
                "raw_l3_hidden128_lead_conditioned_fixed_orog",
            ),
            (
                "weather_dual_resolution_l4_ratio15_hidden128_fixed_orog.yaml",
                "raw_l4_ratio15_hidden128_fixed_orog",
            ),
            (
                "weather_dual_resolution_l4_hidden128_72_36_24_18_9_fixed_orog.yaml",
                "raw_l4_hidden128_72_36_24_18_9_fixed_orog",
            ),
        ]
        expected = {
            "enabled": True,
            "copy_variables": ["orog"],
            "exclude_loss_variables": ["orog"],
            "known_future_variables": [],
        }
        for filename, root in specs:
            with self.subTest(filename=filename, root=root):
                params = YParams(str(PROJECT_ROOT / "configs" / filename), root, resolution_mode="2p5")
                self.assertEqual(params.target_handling, expected)
                self.assertFalse(params.extra_features["enabled"])
                self.assertEqual(len(params.out_channels), 67)
                self.assertEqual(params.model.get("output_channels"), 67)

    def test_dense_l3k24_fixed_orog_config(self) -> None:
        params = YParams(
            str(PROJECT_ROOT / "configs" / "weather_dual_resolution_l3_hidden128_dense_l3k24_fixed_orog.yaml"),
            "raw_l3_hidden128_dense_l3k24_fixed_orog",
            resolution_mode="2p5",
        )
        self.assertEqual(params.hidden_dim, 128)
        self.assertEqual(params.num_heads, 4)
        self.assertEqual(params.head_dim, 32)
        self.assertEqual(params.level_k_neighbors, [8, 8, 8, 24])
        self.assertEqual(params.edge_counts, [82944, 20736, 5184, 3888])
        self.assertEqual(params.graph_format_version, 4)
        self.assertEqual(params.graph_path, "graphs/graph_2p5_k8_l3k24_hybrid_row_aware_L3_v4.pt")
        self.assertEqual(params.rollout_mode, "curriculum")
        self.assertEqual(params.lr_schedule_type, "warmup_cosine")
        self.assertFalse(params.extra_features["enabled"])
        self.assertEqual(
            params.target_handling,
            {
                "enabled": True,
                "copy_variables": ["orog"],
                "exclude_loss_variables": ["orog"],
                "known_future_variables": [],
            },
        )
        channel_names = [f"var_{idx}" for idx in range(66)] + ["orog"]
        handler = TargetHandling.from_params(
            params,
            channel_names=channel_names,
            out_channels=list(range(67)),
            logger=_NullLogger(),
        )
        mask = handler.loss_channel_mask(67)
        self.assertIsNotNone(mask)
        self.assertEqual(float(mask.sum().item()), 66.0)

    def test_l4_ratio15_hidden128_fixed_orog_config(self) -> None:
        params = YParams(
            str(PROJECT_ROOT / "configs" / "weather_dual_resolution_l4_ratio15_hidden128_fixed_orog.yaml"),
            "raw_l4_ratio15_hidden128_fixed_orog",
            resolution_mode="2p5",
        )
        self.assertEqual(params.hierarchy_type, "ratio15_l4")
        self.assertTrue(params.use_l4_ratio15)
        self.assertTrue(params.use_l3)
        self.assertEqual(params.num_graph_levels, 5)
        self.assertEqual(params.hidden_dim, 128)
        self.assertEqual(params.num_heads, 4)
        self.assertEqual(params.head_dim, 32)
        self.assertEqual(params.input_channels, 134)
        self.assertEqual(params.output_channels, 67)
        self.assertEqual(params.level_shapes, [[72, 144], [48, 96], [32, 64], [21, 42], [14, 28]])
        self.assertEqual(params.node_counts, [10368, 4608, 2048, 882, 392])
        self.assertEqual(params.level_k_neighbors, [8, 8, 8, 16, 24])
        self.assertEqual(params.edge_counts, [82944, 36864, 16384, 14112, 9408])
        self.assertEqual(params.graph_format_version, "ratio15_l4_v1")
        self.assertEqual(params.graph_path, "graphs/graph_2p5_ratio15_L4_k8_8_8_16_24_hybrid_row_aware_v1.pt")
        self.assertEqual(params.pooling["type"], "parent_index_meanmax")
        self.assertEqual(params.pooling["mean_type"], "area_weighted")
        self.assertTrue(params.pooling["include_max"])
        self.assertEqual(params.spectral_loss, {"enabled": False})
        self.assertFalse(params.extra_features["enabled"])
        self.assertEqual(
            params.target_handling,
            {
                "enabled": True,
                "copy_variables": ["orog"],
                "exclude_loss_variables": ["orog"],
                "known_future_variables": [],
            },
        )
        channel_names = [f"var_{idx}" for idx in range(66)] + ["orog"]
        handler = TargetHandling.from_params(
            params,
            channel_names=channel_names,
            out_channels=list(range(67)),
            logger=_NullLogger(),
        )
        mask = handler.loss_channel_mask(67)
        self.assertIsNotNone(mask)
        self.assertEqual(float(mask.sum().item()), 66.0)

    def test_l4_72_36_24_18_9_hidden128_fixed_orog_config(self) -> None:
        params = YParams(
            str(PROJECT_ROOT / "configs" / "weather_dual_resolution_l4_hidden128_72_36_24_18_9_fixed_orog.yaml"),
            "raw_l4_hidden128_72_36_24_18_9_fixed_orog",
            resolution_mode="2p5",
        )
        self.assertEqual(params.experiment_name, "main_raw_2p5_b4_acc3_bf16_delta_l4_hidden128_72_36_24_18_9_fixed_orog")
        self.assertEqual(params.hierarchy_type, "l4_72_36_24_18_9")
        self.assertTrue(params.use_l3)
        self.assertTrue(params.use_l4)
        self.assertFalse(params.use_l4_ratio15)
        self.assertEqual(params.num_graph_levels, 5)
        self.assertEqual(params.hidden_dim, 128)
        self.assertEqual(params.num_heads, 4)
        self.assertEqual(params.head_dim, 32)
        self.assertEqual(params.level_shapes, [[72, 144], [36, 72], [24, 48], [18, 36], [9, 18]])
        self.assertEqual(params.node_counts, [10368, 2592, 1152, 648, 162])
        self.assertEqual(params.level_k_neighbors, [8, 8, 8, 12, 24])
        self.assertEqual(params.edge_counts, [82944, 20736, 9216, 7776, 3888])
        self.assertEqual(params.graph_format_version, "l4_72_36_24_18_9_v1")
        self.assertEqual(params.graph_path, "graphs/graph_2p5_l4_72_36_24_18_9_k8_8_8_12_24_hybrid_row_aware_v1.pt")
        self.assertEqual(params.pooling["type"], "parent_index_meanmax")
        self.assertEqual(params.pooling["mean_type"], "area_weighted")
        self.assertEqual(params.spectral_loss, {"enabled": False})
        self.assertEqual(float(params.weight_decay), 0.0)
        self.assertFalse(params.extra_features["enabled"])
        self.assertEqual(
            params.target_handling,
            {
                "enabled": True,
                "copy_variables": ["orog"],
                "exclude_loss_variables": ["orog"],
                "known_future_variables": [],
            },
        )
        channel_names = [f"var_{idx}" for idx in range(66)] + ["orog"]
        handler = TargetHandling.from_params(
            params,
            channel_names=channel_names,
            out_channels=list(range(67)),
            logger=_NullLogger(),
        )
        mask = handler.loss_channel_mask(67)
        self.assertIsNotNone(mask)
        self.assertEqual(float(mask.sum().item()), 66.0)

    def test_scalar_gated_skip_fixed_orog_config(self) -> None:
        params = YParams(
            str(PROJECT_ROOT / "configs" / "weather_dual_resolution_l3_hidden128_scalar_gated_skip_fixed_orog.yaml"),
            "raw_l3_hidden128_scalar_gated_skip_fixed_orog",
            resolution_mode="2p5",
        )
        self.assertEqual(params.hidden_dim, 128)
        self.assertEqual(params.num_heads, 4)
        self.assertEqual(params.head_dim, 32)
        self.assertEqual(params.level_k_neighbors, [8, 8, 8, 8])
        self.assertEqual(params.edge_counts, [82944, 20736, 5184, 1296])
        self.assertEqual(params.graph_path, "graphs/graph_2p5_k8_hybrid_row_aware_L3_v3.pt")
        self.assertEqual(params.skip_fusion["type"], "scalar_gated")
        self.assertEqual(float(params.skip_fusion["init_scale"]), 1.0)
        self.assertEqual(float(params.skip_fusion["max_scale"]), 2.0)
        self.assertEqual(params.rollout_mode, "curriculum")
        self.assertEqual(params.lr_schedule_type, "warmup_cosine")
        self.assertFalse(params.extra_features["enabled"])
        self.assertEqual(
            params.target_handling,
            {
                "enabled": True,
                "copy_variables": ["orog"],
                "exclude_loss_variables": ["orog"],
                "known_future_variables": [],
            },
        )
        channel_names = [f"var_{idx}" for idx in range(66)] + ["orog"]
        handler = TargetHandling.from_params(
            params,
            channel_names=channel_names,
            out_channels=list(range(67)),
            logger=_NullLogger(),
        )
        mask = handler.loss_channel_mask(67)
        self.assertIsNotNone(mask)
        self.assertEqual(float(mask.sum().item()), 66.0)

    def test_scalar_gated_pooling_fixed_orog_config(self) -> None:
        params = YParams(
            str(PROJECT_ROOT / "configs" / "weather_dual_resolution_l3_hidden128_scalar_gated_pooling_fixed_orog.yaml"),
            "raw_l3_hidden128_scalar_gated_pooling_fixed_orog",
            resolution_mode="2p5",
        )
        self.assertEqual(params.hidden_dim, 128)
        self.assertEqual(params.num_heads, 4)
        self.assertEqual(params.head_dim, 32)
        self.assertEqual(params.level_k_neighbors, [8, 8, 8, 8])
        self.assertEqual(params.edge_counts, [82944, 20736, 5184, 1296])
        self.assertEqual(params.graph_path, "graphs/graph_2p5_k8_hybrid_row_aware_L3_v3.pt")
        self.assertEqual(params.pooling["type"], "scalar_gated_meanmax")
        self.assertEqual(float(params.pooling["init_scale"]), 1.0)
        self.assertEqual(float(params.pooling["max_scale"]), 2.0)
        self.assertEqual(params.rollout_mode, "curriculum")
        self.assertEqual(params.lr_schedule_type, "warmup_cosine")
        self.assertFalse(params.extra_features["enabled"])
        self.assertEqual(
            params.target_handling,
            {
                "enabled": True,
                "copy_variables": ["orog"],
                "exclude_loss_variables": ["orog"],
                "known_future_variables": [],
            },
        )
        channel_names = [f"var_{idx}" for idx in range(66)] + ["orog"]
        handler = TargetHandling.from_params(
            params,
            channel_names=channel_names,
            out_channels=list(range(67)),
            logger=_NullLogger(),
        )
        mask = handler.loss_channel_mask(67)
        self.assertIsNotNone(mask)
        self.assertEqual(float(mask.sum().item()), 66.0)

    def test_lead_conditioned_fixed_orog_config(self) -> None:
        params = YParams(
            str(PROJECT_ROOT / "configs" / "weather_dual_resolution_l3_hidden128_lead_conditioned_fixed_orog.yaml"),
            "raw_l3_hidden128_lead_conditioned_fixed_orog",
            resolution_mode="2p5",
        )
        self.assertEqual(params.hidden_dim, 128)
        self.assertEqual(params.num_heads, 4)
        self.assertEqual(params.head_dim, 32)
        self.assertEqual(params.input_channels, 136)
        self.assertEqual(params.model["input_channels"], 136)
        self.assertEqual(params.output_channels, 67)
        self.assertEqual(params.level_k_neighbors, [8, 8, 8, 8])
        self.assertEqual(params.edge_counts, [82944, 20736, 5184, 1296])
        self.assertEqual(params.graph_path, "graphs/graph_2p5_k8_hybrid_row_aware_L3_v3.pt")
        self.assertEqual(
            params.lead_conditioning,
            {"enabled": True, "type": "sincos_concat", "max_lead": 10, "added_input_channels": 2},
        )
        self.assertEqual(params.skip_fusion["type"], "default")
        self.assertEqual(params.pooling["type"], "default")
        self.assertEqual(params.rollout_mode, "curriculum")
        self.assertEqual(params.lr_schedule_type, "warmup_cosine")
        self.assertFalse(params.extra_features["enabled"])
        self.assertEqual(
            params.target_handling,
            {
                "enabled": True,
                "copy_variables": ["orog"],
                "exclude_loss_variables": ["orog"],
                "known_future_variables": [],
            },
        )
        channel_names = [f"var_{idx}" for idx in range(66)] + ["orog"]
        handler = TargetHandling.from_params(
            params,
            channel_names=channel_names,
            out_channels=list(range(67)),
            logger=_NullLogger(),
        )
        mask = handler.loss_channel_mask(67)
        self.assertIsNotNone(mask)
        self.assertEqual(float(mask.sum().item()), 66.0)

    def test_legacy_5p625_template_enables_fixed_orography(self) -> None:
        params = YParams(str(PROJECT_ROOT / "configs" / "gnn_5p625.yaml"), "raw_5p625", resolution_mode="5p625")
        self.assertEqual(
            params.target_handling,
            {
                "enabled": True,
                "copy_variables": ["orog"],
                "exclude_loss_variables": ["orog"],
                "known_future_variables": [],
            },
        )
        self.assertEqual(len(params.out_channels), 67)

    def test_orog_tisr_fixed_configs_load_expected_settings(self) -> None:
        specs = [
            (
                "weather_dual_resolution_l3_orog_tisr_fixed.yaml",
                "raw_l3_orog_tisr_fixed",
                96,
                4,
                24,
            ),
            (
                "weather_dual_resolution_l3_hidden128_orog_tisr_fixed.yaml",
                "raw_l3_hidden128_orog_tisr_fixed",
                128,
                4,
                32,
            ),
            (
                "weather_dual_resolution_l3_hidden160_orog_tisr_fixed.yaml",
                "raw_l3_hidden160_orog_tisr_fixed",
                160,
                5,
                32,
            ),
        ]
        for filename, root, hidden_dim, num_heads, head_dim in specs:
            with self.subTest(filename=filename):
                params = YParams(str(PROJECT_ROOT / "configs" / filename), root, resolution_mode="2p5")
                self.assertEqual(params.hidden_dim, hidden_dim)
                self.assertEqual(params.num_heads, num_heads)
                self.assertEqual(params.head_dim, head_dim)
                self.assertTrue(params.use_l3)
                self.assertEqual(params.num_graph_levels, 4)
                self.assertEqual(params.graph_path, "graphs/graph_2p5_k8_hybrid_row_aware_L3_v3.pt")
                self.assertEqual(params.rollout_stage_epochs, [3, 3, 8, 10, 12, 14])
                self.assertEqual(params.lr_schedule_type, "warmup_cosine")
                self.assertTrue(params.load_only_current_rollout)
                self.assertFalse(params.extra_features["enabled"])
                self.assertEqual(
                    params.target_handling,
                    {
                        "enabled": True,
                        "copy_variables": ["orog"],
                        "known_future_variables": ["tisr"],
                        "exclude_loss_variables": ["orog", "tisr"],
                    },
                )


class CurriculumTisrfixExperimentConfigTest(unittest.TestCase):
    """The `_tisrfix` / `_w96_tisrfix` experiments treat tisr as a prescribed forcing."""

    CONFIG = str(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "config_2p5_l3_hidden128_dense_l3k24_curriculum_S2toS10_3ep_initckpt.yaml"
    )
    EXPECTED_TARGET_HANDLING = {
        "enabled": True,
        "copy_variables": ["orog"],
        "exclude_loss_variables": ["orog", "tisr"],
        "known_future_variables": ["tisr"],
    }

    def _loss_mask_channels(self, params) -> float:
        channel_names = [f"var_{idx}" for idx in range(65)] + ["orog", "tisr"]
        handler = TargetHandling.from_params(
            params,
            channel_names=channel_names,
            out_channels=list(range(67)),
            logger=_NullLogger(),
        )
        mask = handler.loss_channel_mask(67)
        self.assertIsNotNone(mask)
        self.assertEqual(float(mask[65].item()), 0.0)  # orog
        self.assertEqual(float(mask[66].item()), 0.0)  # tisr
        return float(mask.sum().item())

    def test_base_experiment_is_unchanged(self) -> None:
        params = YParams(self.CONFIG, "dense_l3k24_curriculum_S2toS10_3ep_initckpt", resolution_mode="2p5")
        self.assertEqual(
            params.target_handling,
            {
                "enabled": True,
                "copy_variables": ["orog"],
                "exclude_loss_variables": ["orog"],
                "known_future_variables": [],
            },
        )
        self.assertEqual(params.hidden_dim, 128)
        self.assertTrue(str(params.init_from_checkpoint).endswith("best_ckpt.tar"))
        self.assertTrue(params.init_from_checkpoint_strict)

    EXPECTED_SCHEDULE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    EXPECTED_STAGE_EPOCHS = [80, 1, 1, 1, 1, 2, 2, 4, 4, 4]

    def test_tisrfix_keeps_width_and_trains_two_phase_from_scratch(self) -> None:
        params = YParams(
            self.CONFIG, "dense_l3k24_curriculum_S2toS10_3ep_initckpt_tisrfix", resolution_mode="2p5"
        )
        self.assertEqual(params.target_handling, self.EXPECTED_TARGET_HANDLING)
        self.assertEqual(params.hidden_dim, 128)
        self.assertEqual(params.num_heads, 4)
        self.assertEqual(params.model["input_channels"], 134)
        self.assertEqual(params.model["output_channels"], 67)
        self.assertIsNone(params.init_from_checkpoint)
        self.assertFalse(params.init_from_checkpoint_strict)
        self.assertEqual(params.rollout_schedule, self.EXPECTED_SCHEDULE)
        self.assertEqual(params.rollout_stage_epochs, self.EXPECTED_STAGE_EPOCHS)
        self.assertEqual(params.max_epochs, 100)
        self.assertFalse(params.extra_features["enabled"])
        self.assertEqual(self._loss_mask_channels(params), 65.0)

    def test_w96_tisrfix_sets_width_and_disables_warm_start(self) -> None:
        params = YParams(
            self.CONFIG, "dense_l3k24_curriculum_S2toS10_3ep_initckpt_w96_tisrfix", resolution_mode="2p5"
        )
        self.assertEqual(params.target_handling, self.EXPECTED_TARGET_HANDLING)
        self.assertEqual(params.hidden_dim, 96)
        self.assertEqual(params.num_heads, 3)
        self.assertEqual(params.head_dim, 32)
        self.assertEqual(params.model["hidden_dim"], 96)
        self.assertEqual(params.model["num_heads"], 3)
        self.assertEqual(params.model["head_dim"], 32)
        self.assertEqual(params.model["input_channels"], 134)
        self.assertEqual(params.model["output_channels"], 67)
        self.assertIsNone(params.init_from_checkpoint)
        self.assertFalse(params.init_from_checkpoint_strict)
        self.assertEqual(params.rollout_schedule, self.EXPECTED_SCHEDULE)
        self.assertEqual(params.rollout_stage_epochs, self.EXPECTED_STAGE_EPOCHS)
        self.assertEqual(params.max_epochs, 100)
        self.assertEqual(self._loss_mask_channels(params), 65.0)


class TrainerRolloutTisrFeedbackTest(unittest.TestCase):
    """After every rollout step the state fed back into the model must carry the
    ground-truth tisr for that step's valid time, and the loss must ignore tisr."""

    class Adapter:
        def extract_two_steps(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return inp[:, :3], inp[:, 3:6]

    class RecordingModel:
        output_channels = 3
        training = False  # plain stub, not an nn.Module

        def __init__(self) -> None:
            self.adapter = TrainerRolloutTisrFeedbackTest.Adapter()
            self.seen_currents: list[torch.Tensor] = []

        def forward_steps(self, previous: torch.Tensor, current: torch.Tensor, aux_features=None) -> torch.Tensor:
            self.seen_currents.append(current.clone())
            out = current[:, :3].clone()
            out[:, 0] = -123.0  # garbage tisr prediction; must be overridden + masked
            return out

    def _trainer(self, handler: TargetHandling, model) -> Trainer:
        trainer = object.__new__(Trainer)
        trainer.model = model
        trainer.graph_gradient_weight = 0.0
        trainer.loss_obj = LatitudeWeightedMSE(torch.zeros(1))
        trainer.loss_channel_mask = handler.loss_channel_mask(3)
        trainer.target_handler = handler
        trainer.rollout_loss_weights_name = "uniform"
        trainer.rollout_loss_weights_values = None
        trainer.feature_builder = None
        return trainer

    def test_fed_back_state_carries_ground_truth_tisr_each_step(self) -> None:
        handler = _handler()  # channels: [tisr, orog, t2m], known-future tisr, copy orog
        model = self.RecordingModel()
        trainer = self._trainer(handler, model)

        inp = torch.zeros(1, 6, 2, 2)
        inp[:, 4] = 3.0  # orog in the initial state
        target = torch.zeros(1, 5, 3, 2, 2)
        for lead in range(1, 6):
            target[:, lead - 1, 0] = 10.0 * lead  # time-varying tisr truth

        loss, last_pred = Trainer._rollout_loss(trainer, inp, target, rollout_steps=5)

        self.assertEqual(len(model.seen_currents), 5)
        for step in range(1, 5):
            fed_back = model.seen_currents[step]
            self.assertEqual(fed_back.shape, (1, 3, 2, 2))
            expected_tisr = target[:, step - 1, 0]
            self.assertTrue(
                torch.equal(fed_back[:, 0], expected_tisr),
                f"step {step}: fed-back tisr {fed_back[0, 0, 0, 0]} != truth {expected_tisr[0, 0, 0]}",
            )
            self.assertTrue(torch.equal(fed_back[:, 1], inp[:, 4]))  # orog stays static
        self.assertEqual(last_pred.shape, (1, 3, 2, 2))
        self.assertTrue(torch.equal(last_pred[:, 0], target[:, 4, 0]))

    def test_loss_is_invariant_to_the_tisr_prediction(self) -> None:
        handler = _handler()
        inp = torch.zeros(1, 6, 2, 2)
        target = torch.rand(1, 5, 3, 2, 2)

        class FixedTisrModel(self.RecordingModel):
            tisr_value = 0.0

            def forward_steps(self, previous, current, aux_features=None):
                out = current[:, :3].clone()
                out[:, 0] = self.tisr_value
                return out

        losses = []
        for tisr_value in (0.0, 1.0e6):
            model = FixedTisrModel()
            model.tisr_value = tisr_value
            trainer = self._trainer(handler, model)
            loss, _ = Trainer._rollout_loss(trainer, inp, target.clone(), rollout_steps=5)
            losses.append(float(loss.item()))
        self.assertEqual(losses[0], losses[1])


if __name__ == "__main__":
    unittest.main()
