"""P1 tests: gw-run/1 contract + ingest (dashboard/contract.py, dashboard/ingest.py).

Two layers, per dashboard/docs/04-BUILD-PLAN.md:
- real-repo assertions (the initckpt canary proves values are read verbatim);
- synthetic fixtures for every failure path (a fabricated value is a bug —
  the correct output is always a Problem).

unittest-style, runnable by pytest (repo convention).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard import ingest  # noqa: E402
from dashboard.contract import SCHEMA_VERSION  # noqa: E402

RUNS_ROOT = PROJECT_ROOT / "runs"
CANARY_RUN = "dense_l3k24_curriculum_S2toS10_3ep_initckpt"


# ---------------------------------------------------------------------------
# real repo
# ---------------------------------------------------------------------------

@unittest.skipUnless((RUNS_ROOT / CANARY_RUN).is_dir(), "real runs/ not available")
class RealRepoIngestTest(unittest.TestCase):
    records: list = []
    by_id: dict = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = ingest.ingest_all(RUNS_ROOT)
        cls.by_id = {r.run_id: r for r in cls.records}

    def test_expected_evaluated_run_count(self) -> None:
        evaluated = [r for r in self.records if r.status == "evaluated"]
        self.assertGreaterEqual(len(evaluated), 11,
                                [r.run_id for r in evaluated])

    def test_initckpt_canary(self) -> None:
        run = self.by_id[CANARY_RUN]
        self.assertEqual(run.status, "evaluated")
        primary = [e for e in run.evaluations if e.is_primary]
        self.assertEqual(len(primary), 1)
        ev = primary[0]
        self.assertEqual(ev.eval_id, "evaluation_test_weekly52")
        self.assertTrue(ev.is_valid)
        self.assertEqual(ev.horizon, 10)
        self.assertEqual(ev.lead_times, list(range(1, 11)))
        self.assertEqual(len(ev.variables), 6)
        self.assertIn("persistence", ev.baselines)
        self.assertEqual(ev.checkpoint["epoch"], 27)
        # The canary: the value is read VERBATIM from the JSON (invariant I2).
        z500_last = ev.series["z500"].rmse.mean[-1]
        self.assertAlmostEqual(z500_last, 683.826, delta=0.01)
        # persistence has the same axis
        self.assertEqual(len(ev.baselines["persistence"]["z500"]["rmse"].mean), 10)

    def test_initckpt_has_two_first_class_evaluations(self) -> None:
        run = self.by_id[CANARY_RUN]
        eval_ids = sorted(e.eval_id for e in run.evaluations)
        self.assertIn("evaluation_test_weekly52", eval_ids)
        self.assertIn("evaluation_test_weekly52_S5ckpt", eval_ids)
        self.assertEqual(sum(1 for e in run.evaluations if e.is_primary), 1)
        for e in run.evaluations:
            self.assertTrue(e.is_valid, f"{e.eval_id} unexpectedly invalid")

    def test_runs_without_eval_have_no_fabricated_fields(self) -> None:
        no_eval = [r for r in self.records if not r.evaluations]
        self.assertTrue(no_eval, "expected at least one run without evaluations")
        for r in no_eval:
            self.assertIn(r.status, ("trained", "in_progress"))
            d = r.to_dict()
            self.assertNotIn("training", d)          # Q5: never ingested
            self.assertEqual(d["evaluations"], [])

    def test_every_record_dumps_to_strict_json(self) -> None:
        for r in self.records:
            json.dumps(r.to_dict(), allow_nan=False)

    def test_artifact_relpaths_exist(self) -> None:
        for r in self.records:
            for e in r.evaluations:
                for key in ("summary_csv", "summary_txt", "log", "s_dir"):
                    rel = e.artifacts.get(key)
                    if rel is not None:
                        self.assertTrue((Path(r.path) / rel).exists(), rel)
                for rel in e.artifacts.get("plots", {}).values():
                    self.assertTrue((Path(r.path) / rel).is_file(), rel)

    def test_junk_entries_ignored(self) -> None:
        ids = set(self.by_id)
        self.assertNotIn("config_resolved.yaml", ids)
        # the empty nested <name>/<name> dir must not appear as a run
        for r in self.records:
            nested = Path(r.path) / r.run_id
            if nested.is_dir():
                self.assertNotIn(f"{r.run_id}/{r.run_id}", ids)


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------

def _series(horizon: int, base: float = 1.0) -> dict:
    return {
        "mean": [base * i for i in range(1, horizon + 1)],
        "ci_lower": [base * i - 0.1 for i in range(1, horizon + 1)],
        "ci_upper": [base * i + 0.1 for i in range(1, horizon + 1)],
    }


def make_payload(horizon: int = 10, variables=("z500", "t2m"),
                 include_persistence: bool = True,
                 drop_optional: bool = False) -> dict:
    def metrics_block() -> dict:
        return {
            v: {"name": v, "variable_idx": i, "channel": i,
                "rmse": _series(horizon, 10.0), "acc": _series(horizon, 0.05)}
            for i, v in enumerate(variables)
        }

    payload = {
        "eval_fixed_rollout_steps": horizon,
        "lead_times": list(range(1, horizon + 1)),
        "selection": {"split": "test", "n_initial_conditions": 51, "stride": 7},
        "climatology": {"climatology_path": "data/stats/clim.nc"},
        "bootstrap": {"samples": 1000, "confidence_level": 0.95, "seed": 42},
        "features": {},
        "evaluation_target_override": False,
        "orog_tisr_sensitivity": None,
        "affine_calibration": None,
        "external_baselines": [],
        "external_baseline_fairness_note": "",
        "checkpoints": {
            "global_best": {"label": "model", "epoch": 27,
                            "train_rollout_steps": horizon, "params_m": 2.47,
                            "metrics": metrics_block()},
        },
    }
    if include_persistence:
        payload["checkpoints"]["persistence"] = {
            "label": "persistence", "epoch": None, "train_rollout_steps": horizon,
            "metrics": metrics_block(),
        }
    if drop_optional:
        for k in ("features", "evaluation_target_override", "orog_tisr_sensitivity",
                  "affine_calibration", "external_baselines",
                  "external_baseline_fairness_note"):
            payload.pop(k, None)
    return payload


def make_run(root: Path, name: str, *, payload=None, raw_json: str = None,
             eval_dirs=("evaluation_test_weekly52",), done: bool = True,
             horizon: int = 10) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "config_resolved.yaml").write_text(yaml.safe_dump({
        "resolution_mode": "2p5", "hidden_dim": 128, "num_heads": 4,
        "grid_shape": [72, 144], "level_k_neighbors": [8, 8, 8, 24],
        "lr": 5e-5, "min_lr": 1e-6, "lr_schedule_type": "warmup_cosine",
        "warmup_epochs": 1, "max_epochs": 27,
        "rollout_schedule": [2, 3], "rollout_stage_epochs": [3, 3],
        "target_handling": {"known_future_variables": [], "copy_variables": ["orog"]},
        "wandb": {"tags": ["fixture"]},
    }))
    (run / "model_summary.txt").write_text("trainable_parameters: 2473847\n")
    (run / "out.log").write_text("start\nDONE rank 0\n" if done else "start\n")
    (run / "ckpt.tar").write_text("x")
    (run / "last_ckpt.tar").write_text("x")
    for ed in eval_dirs:
        d = run / ed
        d.mkdir()
        (d / f"S{horizon}").mkdir()
        if raw_json is not None:
            (d / f"fixed{horizon}_global_best_metrics.json").write_text(raw_json)
        elif payload is not None:
            (d / f"fixed{horizon}_global_best_metrics.json").write_text(
                json.dumps(payload, allow_nan=True))
    return run


class SyntheticFixtureTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _ingest_one(self, name: str):
        records = ingest.ingest_all(self.root)
        return next(r for r in records if r.run_id == name)

    def test_valid_fixture_is_evaluated(self) -> None:
        make_run(self.root, "ok", payload=make_payload())
        run = self._ingest_one("ok")
        self.assertEqual(run.status, "evaluated")
        ev = run.evaluations[0]
        self.assertTrue(ev.is_valid and ev.is_primary)
        self.assertEqual(ev.horizon, 10)
        self.assertEqual(ev.series["z500"].rmse.mean[0], 10.0)
        self.assertEqual(run.architecture.params_millions, 2.474)
        self.assertFalse(any(p.severity == "error" for p in run.problems))

    def test_corrupt_json_is_exactly_one_parse_error(self) -> None:
        make_run(self.root, "corrupt", raw_json="{ this is not json")
        run = self._ingest_one("corrupt")
        parse_errors = [p for p in run.problems if p.reason == "parse_error"]
        self.assertEqual(len(parse_errors), 1)
        self.assertEqual(parse_errors[0].severity, "error")
        self.assertEqual(run.status, "invalid")
        ev = run.evaluations[0]
        self.assertFalse(ev.is_valid)
        self.assertEqual(ev.series, {})          # nothing fabricated

    def test_truncated_mean_is_axis_mismatch(self) -> None:
        payload = make_payload()
        payload["checkpoints"]["global_best"]["metrics"]["z500"]["rmse"]["mean"] = [1.0] * 9
        make_run(self.root, "trunc", payload=payload)
        run = self._ingest_one("trunc")
        mismatches = [p for p in run.problems if p.reason == "axis_mismatch"]
        self.assertTrue(mismatches)
        self.assertIn("9 values", mismatches[0].found)
        self.assertIn("10", mismatches[0].expected)
        ev = run.evaluations[0]
        self.assertFalse(ev.is_valid)            # error severity on model series
        self.assertNotIn("z500", ev.series)      # dropped, not padded
        self.assertIn("t2m", ev.series)          # the healthy variable survives

    def test_missing_optional_keys_is_schema_drift_warning(self) -> None:
        make_run(self.root, "olddrift", payload=make_payload(drop_optional=True))
        run = self._ingest_one("olddrift")
        drift = [p for p in run.problems if p.reason == "schema_drift"]
        self.assertEqual(len(drift), 1)
        self.assertEqual(drift[0].severity, "warning")
        self.assertEqual(run.status, "evaluated")      # ingest continues
        self.assertTrue(run.evaluations[0].is_valid)

    def test_nan_passes_through_as_null_with_warning(self) -> None:
        payload = make_payload()
        payload["checkpoints"]["global_best"]["metrics"]["t2m"]["acc"]["mean"][3] = float("nan")
        make_run(self.root, "nanrun", payload=payload)
        run = self._ingest_one("nanrun")
        ev = run.evaluations[0]
        self.assertTrue(ev.is_valid)
        self.assertIsNone(ev.series["t2m"].acc.mean[3])   # null, not interpolated
        self.assertEqual(ev.series["t2m"].acc.mean[2], 0.05 * 3)
        nan_warnings = [p for p in run.problems
                        if p.severity == "warning" and "NaN" in p.found]
        self.assertTrue(nan_warnings)
        json.dumps(run.to_dict(), allow_nan=False)        # strict JSON survives

    def test_missing_global_best_is_error(self) -> None:
        payload = make_payload()
        del payload["checkpoints"]["global_best"]
        make_run(self.root, "nogb", payload=payload)
        run = self._ingest_one("nogb")
        self.assertEqual(run.status, "invalid")
        self.assertTrue(any(p.reason == "missing_key" and p.severity == "error"
                            for p in run.problems))

    def test_two_eval_dirs_one_primary(self) -> None:
        make_run(self.root, "twoeval", payload=make_payload(),
                 eval_dirs=("evaluation_test_weekly52", "evaluation_test_weekly52_S5ckpt"))
        run = self._ingest_one("twoeval")
        self.assertEqual(len(run.evaluations), 2)
        primaries = [e for e in run.evaluations if e.is_primary]
        self.assertEqual(len(primaries), 1)
        self.assertEqual(primaries[0].eval_id, "evaluation_test_weekly52")

    def test_eval_dir_without_json_is_missing_file_warning(self) -> None:
        run_dir = make_run(self.root, "midway", payload=None,
                           eval_dirs=("evaluation_test_weekly52",))
        (run_dir / "evaluation_test_weekly52" / "evaluation.log").write_text("running\n")
        run = self._ingest_one("midway")
        self.assertEqual(run.status, "trained")   # absence is never invalidity
        self.assertEqual(run.evaluations, [])
        missing = [p for p in run.problems if p.reason == "missing_file"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].severity, "warning")

    def test_stale_run_without_done_gets_stale_partial(self) -> None:
        run_dir = make_run(self.root, "stale", payload=None, eval_dirs=(), done=False)
        old = 10_000
        for name in ("out.log", "ckpt.tar", "last_ckpt.tar"):
            os.utime(run_dir / name, (old, old))
        run = self._ingest_one("stale")
        self.assertEqual(run.status, "in_progress")
        self.assertTrue(any(p.reason == "stale_partial" for p in run.problems))

    def test_horizon_seven_never_assumes_ten(self) -> None:
        make_run(self.root, "h7", payload=make_payload(horizon=7), horizon=7)
        run = self._ingest_one("h7")
        ev = run.evaluations[0]
        self.assertTrue(ev.is_valid)
        self.assertEqual(ev.horizon, 7)
        self.assertEqual(ev.lead_times, list(range(1, 8)))
        self.assertEqual(len(ev.series["z500"].rmse.mean), 7)
        self.assertEqual(ev.artifacts["s_dir"], "evaluation_test_weekly52/S7")

    def test_junk_root_entries_ignored(self) -> None:
        make_run(self.root, "realrun", payload=make_payload())
        (self.root / "config_resolved.yaml").write_text("stray: file\n")
        (self.root / "realrun" / "realrun").mkdir()      # empty nested dir
        (self.root / "wandb").mkdir()
        records = ingest.ingest_all(self.root)
        self.assertEqual([r.run_id for r in records], ["realrun"])

    def test_run_without_run_json_still_parses(self) -> None:
        """run.json is optional/additive: a run lacking it ingests normally.
        The scanner keys on config_resolved.yaml, never on run.json."""
        run_dir = make_run(self.root, "norunjson", payload=make_payload())
        self.assertFalse((run_dir / "run.json").exists())
        run = self._ingest_one("norunjson")
        self.assertEqual(run.status, "evaluated")
        self.assertTrue(run.evaluations[0].is_valid)

    def test_run_json_manifest_is_ignored_by_scanner(self) -> None:
        """The run.json manifest written by run_full_pipeline.sh is additive:
        its presence must not change run detection, status, or add problems."""
        run_dir = make_run(self.root, "withrunjson", payload=make_payload())
        (run_dir / "run.json").write_text(json.dumps({
            "schema": "gw-run-manifest/1", "status": "complete",
            "resolution": "2p5", "config_name": "withrunjson", "config": "c.yaml",
            "seed": 777, "horizon": 10, "git_sha": "abc1234",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T01:00:00+00:00",
        }))
        run = self._ingest_one("withrunjson")
        self.assertEqual(run.run_id, "withrunjson")
        self.assertEqual(run.status, "evaluated")
        self.assertTrue(run.evaluations[0].is_valid)
        self.assertFalse(any(p.severity == "error" for p in run.problems))

    def test_diagnostics_csvs_with_string_columns_parse(self) -> None:
        """Regression: identifier columns (phase, layer_name, backend) must never
        crash the diagnostics reader; malformed rows are skipped, not fatal."""
        run_dir = make_run(self.root, "diagrun", payload=make_payload())
        diag = run_dir / "diagnostics_full_eval"
        diag.mkdir()
        (diag / "epoch_0010_attention_metrics.csv").write_text(
            "epoch,phase,layer_name,entropy,entropy_norm\n"
            "10,valid,model/encoder.0,1.93,0.927\n"
            "10,valid,processor/l0_blocks.0,1.86,0.894\n")
        (diag / "epoch_0010_variance_ratio.csv").write_text(
            "epoch,phase,lead,variable,variance_ratio,backend\n"
            "10,valid,1,z500,0.99,rfft2\n"
            "10,valid,not_a_number,z500,0.98,rfft2\n"   # malformed row
            "10,valid,2,z500,0.97,rfft2\n")
        (diag / "epoch_0010_rollout_curve.csv").write_text(
            "epoch,phase,horizon,step,loss\n"
            "10,valid,2,1,0.03\n10,valid,2,2,0.05\n")
        run = self._ingest_one("diagrun")
        self.assertFalse(any(p.reason == "parse_error" and p.artifact == "diagnostics"
                             for p in run.problems),
                         [f"{p.reason}:{p.found}" for p in run.problems])
        d = run.diagnostics
        self.assertEqual(len(d.attention_table), 2)
        self.assertEqual(d.attention_table[0]["phase"], "valid")
        self.assertEqual(d.attention_table[0]["entropy_norm"], 0.927)
        self.assertEqual(d.variance_ratio["z500"], [0.99, 0.97])   # bad row skipped
        self.assertEqual(d.rollout_curve["losses"], [0.03, 0.05])
        json.dumps(run.to_dict(), allow_nan=False)

    def test_cli_json_output(self) -> None:
        make_run(self.root, "cliok", payload=make_payload())
        out = subprocess.run(
            [sys.executable, "-m", "dashboard.ingest", "--once", "--json",
             "--root", str(self.root)],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), check=True)
        payload = json.loads(out.stdout)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["run_id"], "cliok")
        self.assertEqual(payload[0]["schema_version"], SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
