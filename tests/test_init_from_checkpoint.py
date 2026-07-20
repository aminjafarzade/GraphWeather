from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer import Trainer  # noqa: E402


# Minimal graph/resolution metadata shared by the fake source checkpoint and the
# fake trainer, so resolution and graph-topology checks pass.
GRAPH_META = {
    "resolution_mode": "test",
    "graph_connectivity_strategy": "unit",
    "graph_format_version": 1,
    "graph_k": 2,
    "graph_coordinate_hash": "abc",
}


def _make_bare_trainer(model, optimizer, *, params, graph_meta=GRAPH_META, use_delta=False):
    """Build a Trainer shell with only the attributes initialize_from_checkpoint touches.

    Mirrors the object.__new__(Trainer) pattern used by the restore_checkpoint tests
    so we can exercise the method without a full data/graph pipeline.
    """
    trainer = object.__new__(Trainer)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.scheduler = None
    trainer.device = torch.device("cpu")
    trainer.graph_topology_metadata = graph_meta
    trainer.params = params
    trainer.use_delta_normalization = use_delta
    trainer.feature_builder = None
    trainer.target_handler = None
    # Deliberately non-zero so we can prove the warm-start resets them.
    trainer.iters = 999
    trainer.start_epoch = 77
    trainer.epoch = 77
    trainer.best_score_global = float("inf")
    trainer.best_score_by_stage = {}
    return trainer


def _randomize(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn_like(param))


class InitFromCheckpointTest(unittest.TestCase):
    def test_partial_warm_start_loads_weights_and_resets_training_state(self) -> None:
        source = torch.nn.Linear(3, 2)
        _randomize(source)
        target = torch.nn.Linear(3, 2)
        _randomize(target)  # different weights than source
        optimizer = torch.optim.AdamW(target.parameters(), lr=1.0e-4)
        params = SimpleNamespace(resolution_mode="test", init_from_checkpoint_strict=False)
        trainer = _make_bare_trainer(target, optimizer, params=params)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "src_ckpt.tar"
            torch.save(
                {"iters": 5, "epoch": 42, "model_state": source.state_dict(), "metadata": GRAPH_META},
                path,
            )
            Trainer.initialize_from_checkpoint(trainer, str(path))

        source_params = dict(source.named_parameters())
        for name, param in target.named_parameters():
            self.assertTrue(torch.equal(param, source_params[name]), f"{name} not warm-started")
        # Training state starts fresh (NOT the checkpoint's epoch 42 / iters 5).
        self.assertEqual(trainer.epoch, 0)
        self.assertEqual(trainer.start_epoch, 0)
        self.assertEqual(trainer.iters, 0)

    def test_strict_warm_start_loads_all_matching_weights(self) -> None:
        source = torch.nn.Linear(3, 2)
        _randomize(source)
        target = torch.nn.Linear(3, 2)
        _randomize(target)
        optimizer = torch.optim.AdamW(target.parameters(), lr=1.0e-4)
        params = SimpleNamespace(resolution_mode="test", init_from_checkpoint_strict=True)
        trainer = _make_bare_trainer(target, optimizer, params=params)

        # The architecture validator has its own coverage and needs full model metadata;
        # stub it here so we exercise this method's strict weight-load + reset branch.
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "src.trainer.validate_checkpoint_architecture"
        ) as arch_check:
            path = Path(tmp) / "src_ckpt.tar"
            torch.save({"epoch": 3, "model_state": source.state_dict(), "metadata": GRAPH_META}, path)
            Trainer.initialize_from_checkpoint(trainer, str(path))
            arch_check.assert_called_once()  # strict path really validated architecture

        source_params = dict(source.named_parameters())
        for name, param in target.named_parameters():
            self.assertTrue(torch.equal(param, source_params[name]))
        self.assertEqual(trainer.epoch, 0)

    def test_strict_warm_start_rejects_missing_tensor(self) -> None:
        """Strict mode is all-or-nothing: a checkpoint missing a weight must error,
        rather than silently leaving it initialized (the failure mode of the
        allow_partial path)."""
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(target.parameters(), lr=1.0e-4)
        params = SimpleNamespace(resolution_mode="test", init_from_checkpoint_strict=True)
        trainer = _make_bare_trainer(target, optimizer, params=params)

        incomplete_state = {k: v for k, v in source.state_dict().items() if k != "bias"}
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "src.trainer.validate_checkpoint_architecture"
        ):
            path = Path(tmp) / "src_ckpt.tar"
            torch.save({"epoch": 1, "model_state": incomplete_state, "metadata": GRAPH_META}, path)
            with self.assertRaises(RuntimeError):
                Trainer.initialize_from_checkpoint(trainer, str(path))

    def test_allow_partial_overrides_strict_flag(self) -> None:
        """Legacy init_from_checkpoint_allow_partial forces the non-strict loader even
        when init_from_checkpoint_strict is left at its default True."""
        source = torch.nn.Linear(3, 2)
        _randomize(source)
        target = torch.nn.Linear(3, 2)
        _randomize(target)
        optimizer = torch.optim.AdamW(target.parameters(), lr=1.0e-4)
        params = SimpleNamespace(
            resolution_mode="test",
            init_from_checkpoint_strict=True,
            init_from_checkpoint_allow_partial=True,
        )
        trainer = _make_bare_trainer(target, optimizer, params=params)

        # No architecture stub: if the strict branch were taken it would call the real
        # validator and fail on the tiny model. Reaching a clean load proves partial won.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "src_ckpt.tar"
            torch.save({"epoch": 9, "model_state": source.state_dict(), "metadata": GRAPH_META}, path)
            Trainer.initialize_from_checkpoint(trainer, str(path))

        source_params = dict(source.named_parameters())
        for name, param in target.named_parameters():
            self.assertTrue(torch.equal(param, source_params[name]))

    def test_strict_warm_start_warns_but_loads_on_target_handling_mismatch(self) -> None:
        """A target-handling mismatch (e.g. the new run prescribes tisr as a known
        forcing while the source checkpoint predicted it) must not block a strict
        warm-start: target handling carries no weights. It warns and loads."""
        from types import SimpleNamespace as NS

        from src.target_handling import TargetHandling

        source = torch.nn.Linear(3, 2)
        _randomize(source)
        target = torch.nn.Linear(3, 2)
        _randomize(target)
        optimizer = torch.optim.AdamW(target.parameters(), lr=1.0e-4)
        params = NS(resolution_mode="test", init_from_checkpoint_strict=True)
        trainer = _make_bare_trainer(target, optimizer, params=params)

        class _Quiet:
            def info(self, *a, **k):
                return None

            def warning(self, *a, **k):
                return None

        handler_params = NS(
            extra_features={"enabled": False},
            target_handling={
                "enabled": True,
                "copy_variables": ["orog"],
                "known_future_variables": ["tisr"],
                "exclude_loss_variables": ["orog", "tisr"],
            },
            variable_metadata={"path": None},
            global_means_path="",
            global_stds_path="",
            experiment_dir="",
        )
        trainer.target_handler = TargetHandling.from_params(
            handler_params,
            channel_names=["tisr", "orog", "t2m"],
            out_channels=[0, 1, 2],
            logger=_Quiet(),
        )

        checkpoint_meta = dict(
            GRAPH_META,
            target_handling={
                "enabled": True,
                "copy_variables": ["orog"],
                "known_future_variables": [],
                "exclude_loss_variables": ["orog"],
            },
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "src.trainer.validate_checkpoint_architecture"
        ):
            path = Path(tmp) / "src_ckpt.tar"
            torch.save({"epoch": 3, "model_state": source.state_dict(), "metadata": checkpoint_meta}, path)
            with self.assertLogs(level="WARNING") as logs:
                Trainer.initialize_from_checkpoint(trainer, str(path))
        self.assertTrue(any("target_handling mismatch" in msg for msg in logs.output))
        self.assertTrue(any("Proceeding with the warm-start" in msg for msg in logs.output))

        source_params = dict(source.named_parameters())
        for name, param in target.named_parameters():
            self.assertTrue(torch.equal(param, source_params[name]))
        self.assertEqual(trainer.epoch, 0)

    def test_missing_path_raises(self) -> None:
        target = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(target.parameters(), lr=1.0e-4)
        params = SimpleNamespace(resolution_mode="test")
        trainer = _make_bare_trainer(target, optimizer, params=params)
        with self.assertRaises(FileNotFoundError):
            Trainer.initialize_from_checkpoint(trainer, "/no/such/checkpoint.tar")

    def test_resolution_mismatch_raises(self) -> None:
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(target.parameters(), lr=1.0e-4)
        params = SimpleNamespace(resolution_mode="2p5", init_from_checkpoint_strict=False)
        trainer = _make_bare_trainer(target, optimizer, params=params)
        foreign_meta = dict(GRAPH_META, resolution_mode="5p625")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "src_ckpt.tar"
            torch.save({"epoch": 1, "model_state": source.state_dict(), "metadata": foreign_meta}, path)
            with self.assertRaises(RuntimeError):
                Trainer.initialize_from_checkpoint(trainer, str(path))


if __name__ == "__main__":
    unittest.main()
