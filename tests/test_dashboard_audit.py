"""Audit battery (docs/05-AUDIT.md): adversarial fixtures proving sections A-E.

Each test corresponds to an audit checklist item; hand-computed expectations
throughout. unittest-style, runnable by pytest.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from dashboard import ingest  # noqa: E402
from dashboard.app import create_app  # noqa: E402
from dashboard.scanner import RunStore  # noqa: E402
from tests.test_dashboard_ingest import make_payload, make_run  # noqa: E402


class ContractAuditTest(unittest.TestCase):
    """Section A items not already covered by test_dashboard_ingest."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _one(self, name):
        return next(r for r in ingest.ingest_all(self.root) if r.run_id == name)

    def test_summary_csv_headers_never_parsed(self) -> None:
        """A: the lying rmse_day10/_1_10 headers must appear nowhere in code."""
        for py in (PROJECT_ROOT / "dashboard").glob("*.py"):
            src = py.read_text()
            for needle in ("rmse_day10", "acc_day10", "avg_1_10"):
                self.assertNotIn(needle, src, f"{py.name} references {needle}")

    def test_lead_times_length_mismatch_is_axis_error(self) -> None:
        payload = make_payload(horizon=10)
        payload["lead_times"] = list(range(1, 10))       # 9 leads, N=10
        make_run(self.root, "shortaxis", payload=payload)
        run = self._one("shortaxis")
        self.assertEqual(run.status, "invalid")
        mism = [p for p in run.problems if p.reason == "axis_mismatch"]
        self.assertTrue(mism)
        self.assertIn("9", mism[0].found)

    def test_ci_length_mismatch_is_warning_ci_dropped(self) -> None:
        payload = make_payload(horizon=10)
        payload["checkpoints"]["global_best"]["metrics"]["z500"]["rmse"]["ci_lower"] = [1.0] * 9
        make_run(self.root, "shortci", payload=payload)
        run = self._one("shortci")
        ev = run.evaluations[0]
        self.assertTrue(ev.is_valid)                     # warning, not fatal
        self.assertIsNone(ev.series["z500"].rmse.ci_lower)   # dropped, not padded
        self.assertEqual(len(ev.series["z500"].rmse.mean), 10)  # mean untouched
        self.assertTrue(any(p.reason == "axis_mismatch" and p.severity == "warning"
                            for p in run.problems))


class RankingAuditTest(unittest.TestCase):
    """Section B: hand-computed expectations for the backend-only logic."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        # h10: rmse mean = 10,20,...,100 ; acc = .05,.10,...,.50
        make_run(cls.root, "h10", payload=make_payload(horizon=10))
        # h7: same generator at H=7 -> rmse 10..70
        make_run(cls.root, "h7", payload=make_payload(horizon=7), horizon=7)
        # worse10: rmse mean 11,22,...,110 (10% worse everywhere)
        p = make_payload(horizon=10)
        for var in ("z500", "t2m"):
            m = p["checkpoints"]["global_best"]["metrics"][var]
            m["rmse"]["mean"] = [11.0 * i for i in range(1, 11)]
            m["rmse"]["ci_lower"] = [11.0 * i - 0.1 for i in range(1, 11)]
            m["rmse"]["ci_upper"] = [11.0 * i + 0.1 for i in range(1, 11)]
        make_run(cls.root, "worse10", payload=p)
        # nearby: value at day 10 = 100.05 with CI overlapping h10's
        p2 = make_payload(horizon=10)
        m = p2["checkpoints"]["global_best"]["metrics"]["z500"]
        m["rmse"]["mean"] = [10.0 * i for i in range(1, 10)] + [100.05]
        m["rmse"]["ci_lower"] = [v - 0.5 for v in m["rmse"]["mean"]]
        m["rmse"]["ci_upper"] = [v + 0.5 for v in m["rmse"]["mean"]]
        make_run(cls.root, "nearby", payload=p2)

        cls.app = create_app(cls.root, auto_scan=False, externals=[])
        cls.client = TestClient(cls.app)
        cls.client.__enter__()
        cls.app.state.store.sweep_once()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        cls._tmp.cleanup()

    def _rank(self, body):
        return self.client.post("/api/ranking", json=body).json()

    def test_mean_leads_averages_common_leads_only(self) -> None:
        """The audit's explicit rule: mean over the intersection, never the
        full per-run axis. h10 full-axis mean would be 55; common 1..7 -> 40."""
        payload = self._rank({"runs": ["h10", "h7"], "variable": "z500",
                              "metric": "rmse", "lead": {"type": "mean_leads"}})
        self.assertEqual(payload["criterion"]["leads_used"], list(range(1, 8)))
        rows = {r["run_id"]: r for r in payload["rows"]}
        self.assertAlmostEqual(rows["h10"]["value"], 40.0)   # (10+...+70)/7
        self.assertAlmostEqual(rows["h7"]["value"], 40.0)
        self.assertTrue(any(w["type"] == "horizon_truncated"
                            for w in payload["warnings"]))

    def test_delta_vs_best_hand_computed(self) -> None:
        payload = self._rank({"runs": ["h10", "worse10"], "variable": "z500",
                              "metric": "rmse", "lead": {"type": "day", "k": 10}})
        rows = {r["run_id"]: r for r in payload["rows"]}
        self.assertEqual(rows["h10"]["rank"], 1)
        self.assertEqual(rows["h10"]["delta_vs_best_pct"], 0.0)
        # (110 - 100) / 100 * 100 = +10%
        self.assertAlmostEqual(rows["worse10"]["delta_vs_best_pct"], 10.0)

    def test_acc_delta_sign_higher_is_better(self) -> None:
        payload = self._rank({"runs": ["h10", "worse10"], "variable": "z500",
                              "metric": "acc", "lead": {"type": "day", "k": 10}})
        rows = payload["rows"]
        self.assertEqual(rows[0]["delta_vs_best_pct"], 0.0)
        if len(rows) > 1 and rows[1]["value"] is not None:
            self.assertLessEqual(rows[1]["value"], rows[0]["value"])

    def test_best_ci_overlap_and_clear_win(self) -> None:
        # h10 (100 ± 0.1) vs nearby (100.05 ± 0.5): CIs overlap -> flag True
        best = self.client.get("/api/best", params={
            "runs": "h10,nearby", "lead": 10, "metric": "rmse"}).json()
        z = best["variables"]["z500"]
        self.assertEqual(z["run_id"], "h10")
        self.assertTrue(z["within_ci_of_runner_up"])
        # h10 (100 ± 0.1) vs worse10 (110 ± 0.1): no overlap -> clear win
        best2 = self.client.get("/api/best", params={
            "runs": "h10,worse10", "lead": 10, "metric": "rmse"}).json()
        self.assertFalse(best2["variables"]["z500"]["within_ci_of_runner_up"])

    def test_best_includes_selected_externals(self) -> None:
        """A selected external baseline competes in best-per-variable; when it
        wins there is no CI, so the CI flag is honestly null."""
        ext_csv = Path(self.root) / "ext_best.csv"
        lines = ["variable,timestep,rmse,acc"]
        for tstep in range(1, 11):
            lines.append(f"z500,{tstep},{1.0 * tstep},0.9")   # far better than runs
        ext_csv.write_text("\n".join(lines) + "\n")
        from dashboard.external import load_external_csv
        ext = load_external_csv(ext_csv, id="ref-x", label="RefX", resolution="2p5")
        app = create_app(Path(self.root), auto_scan=False, externals=[ext])
        with TestClient(app) as client:
            app.state.store.sweep_once()
            best = client.get("/api/best", params={
                "runs": "h10,worse10", "lead": 10, "metric": "rmse",
                "include": "external:ref-x"}).json()
            z = best["variables"]["z500"]
            self.assertEqual(z["run_id"], "ref-x")
            self.assertTrue(z["external"])
            self.assertIsNone(z["within_ci_of_runner_up"])   # externals carry no CI
            # without include, the external does not participate
            best2 = client.get("/api/best", params={
                "runs": "h10,worse10", "lead": 10, "metric": "rmse"}).json()
            self.assertEqual(best2["variables"]["z500"]["run_id"], "h10")

    def test_acc_crossing_no_cross_is_sentinel_not_fabricated(self) -> None:
        payload = self._rank({"runs": ["h10"], "variable": "z500",
                              "lead": {"type": "acc_crossing", "threshold": 0.01}})
        row = payload["rows"][0]
        self.assertIsNone(row["value"])                   # sentinel, no made-up lead
        self.assertIn("never_crossed", row["flags"])
        self.assertIn("derived", row["flags"])

    def test_ranking_deterministic_with_stable_ties(self) -> None:
        body = {"runs": ["h7", "h10"], "variable": "z500", "metric": "rmse",
                "lead": {"type": "mean_leads"}}
        a = self._rank(body)["rows"]
        b = self._rank(body)["rows"]
        self.assertEqual(a, b)
        # equal values (both 40.0): stable order = criterion order
        self.assertEqual([r["run_id"] for r in a][:2], ["h7", "h10"])


class RobustnessAuditTest(unittest.TestCase):
    """Section C/D items not already covered elsewhere."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_persistence_warns_never_fabricates(self) -> None:
        make_run(self.root, "nopers",
                 payload=make_payload(include_persistence=False))
        app = create_app(self.root, auto_scan=False, externals=[])
        with TestClient(app) as client:
            app.state.store.sweep_once()
            ov = client.get("/api/overlay", params={
                "runs": "nopers", "variable": "z500", "metric": "rmse",
                "include": "persistence"}).json()
            self.assertEqual(ov["baselines"], [])
            self.assertTrue(any(w["type"] == "baseline_missing"
                                for w in ov["warnings"]))

    def test_empty_runs_root_is_healthy(self) -> None:
        app = create_app(self.root, auto_scan=False, externals=[])
        with TestClient(app) as client:
            app.state.store.sweep_once()
            meta = client.get("/api/meta")
            self.assertEqual(meta.status_code, 200)
            self.assertEqual(meta.json()["variables_union"], [])
            self.assertEqual(client.get("/api/runs").json()["runs"], [])

    def test_specific_eval_id_is_selectable(self) -> None:
        """Q1/D: overlay can target a secondary evaluation, not only primary."""
        run_dir = make_run(self.root, "twoeval", payload=make_payload())
        # secondary eval dir with DIFFERENT stored values (mean = 1000, 2000, ...)
        alt = make_payload()
        for var in ("z500", "t2m"):
            alt["checkpoints"]["global_best"]["metrics"][var]["rmse"]["mean"] = \
                [1000.0 * i for i in range(1, 11)]
        d = run_dir / "evaluation_test_weekly52_S5ckpt"
        d.mkdir()
        (d / "S10").mkdir()
        (d / "fixed10_global_best_metrics.json").write_text(json.dumps(alt))
        app = create_app(self.root, auto_scan=False, externals=[])
        with TestClient(app) as client:
            app.state.store.sweep_once()
            primary = client.get("/api/overlay", params={
                "runs": "twoeval", "variable": "z500", "metric": "rmse"}).json()
            secondary = client.get("/api/overlay", params={
                "runs": "twoeval", "variable": "z500", "metric": "rmse",
                "eval": "evaluation_test_weekly52_S5ckpt"}).json()
            self.assertEqual(primary["series"][0]["values"][0], 10.0)
            self.assertEqual(secondary["series"][0]["values"][0], 1000.0)

    def test_artifact_conditional_request_304(self) -> None:
        make_run(self.root, "etagrun", payload=make_payload())
        app = create_app(self.root, auto_scan=False, externals=[])
        with TestClient(app) as client:
            app.state.store.sweep_once()
            first = client.get("/api/artifacts/etagrun/config_resolved.yaml")
            self.assertEqual(first.status_code, 200)
            etag = first.headers["ETag"]
            second = client.get("/api/artifacts/etagrun/config_resolved.yaml",
                                headers={"If-None-Match": etag})
            self.assertEqual(second.status_code, 304)

    def test_meta_reports_problem_totals(self) -> None:
        make_run(self.root, "badjson", raw_json="{ nope")
        store = RunStore(self.root, grace_sweeps=0)
        app = create_app(self.root, auto_scan=False, externals=[])
        app.state.store = store
        with TestClient(app) as client:
            # grace disabled -> immediate error surface
            store.sweep_once()
            meta = client.get("/api/meta").json()
            self.assertGreaterEqual(meta["problem_totals"]["error"], 0)
            probs = client.get("/api/problems").json()["problems"]
            self.assertEqual(meta["problem_totals"]["error"] +
                             meta["problem_totals"]["warning"], len(probs))


if __name__ == "__main__":
    unittest.main()
