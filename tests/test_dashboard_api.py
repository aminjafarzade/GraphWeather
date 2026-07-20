"""P2 tests: polling scanner, ranking engine, FastAPI endpoints.

unittest-style, runnable by pytest. Uses synthetic fixture runs (helpers from
test_dashboard_ingest) plus a fixture external-baseline CSV; no GPU, no writes
into the real runs/.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from dashboard.app import create_app  # noqa: E402
from dashboard.contract import SCHEMA_VERSION  # noqa: E402
from dashboard.external import load_external_csv  # noqa: E402
from dashboard.scanner import RunStore  # noqa: E402
from tests.test_dashboard_ingest import make_payload, make_run  # noqa: E402


def _bump_mtime(path: Path, offset_ns: int = 5_000_000_000) -> None:
    st = path.stat()
    ns = st.st_mtime_ns + offset_ns
    os.utime(path, ns=(ns, ns))


def _write_external_csv(path: Path, horizon: int = 10) -> None:
    lines = ["variable,timestep,rmse,acc"]
    for var in ("z500", "t2m"):
        for t in range(1, horizon + 1):
            lines.append(f"{var},{t},{5.0 * t},{max(0.99 - 0.09 * t, 0.01):.3f}")
    path.write_text("\n".join(lines) + "\n")


class ScannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_add_update_remove_events(self) -> None:
        make_run(self.root, "alpha", payload=make_payload())
        store = RunStore(self.root)

        events = store.sweep_once()
        self.assertEqual([e["type"] for e in events], ["run_added"])

        # unchanged fingerprint -> no events, no re-parse
        self.assertEqual(store.sweep_once(), [])

        # touch the metrics JSON -> run_updated with the new content
        json_path = self.root / "alpha" / "evaluation_test_weekly52" / \
            "fixed10_global_best_metrics.json"
        payload = make_payload()
        payload["checkpoints"]["global_best"]["epoch"] = 99
        json_path.write_text(json.dumps(payload))
        _bump_mtime(json_path)
        events = store.sweep_once()
        self.assertEqual([e["type"] for e in events], ["run_updated"])
        self.assertEqual(store.get("alpha").evaluations[0].checkpoint["epoch"], 99)

        # removal -> tombstone + run_removed
        shutil.rmtree(self.root / "alpha")
        events = store.sweep_once()
        self.assertEqual([e["type"] for e in events], ["run_removed"])
        tomb = store.get("alpha")
        self.assertEqual(tomb.status, "removed")
        self.assertTrue(any(p.reason == "missing_file" for p in tomb.problems))

    def test_parse_error_grace_window(self) -> None:
        make_run(self.root, "beta", payload=make_payload())
        store = RunStore(self.root, grace_sweeps=2)
        store.sweep_once()
        self.assertEqual(store.get("beta").status, "evaluated")

        json_path = self.root / "beta" / "evaluation_test_weekly52" / \
            "fixed10_global_best_metrics.json"
        json_path.write_text("{ mid-write garbage")
        _bump_mtime(json_path)

        # sweeps 1 and 2: withheld (warning, not invalid)
        for expect_grace in (1, 2):
            store.sweep_once()
            record = store.get("beta")
            self.assertNotEqual(record.status, "invalid",
                                f"grace sweep {expect_grace} published invalid")
            self.assertTrue(any(p.reason == "stale_partial" for p in record.problems))
        # sweep 3: grace exhausted -> invalid with the parse_error
        store.sweep_once()
        record = store.get("beta")
        self.assertEqual(record.status, "invalid")
        self.assertTrue(any(p.reason == "parse_error" and p.severity == "error"
                            for p in record.problems))

        # recovery: valid JSON again -> evaluated, counters reset
        json_path.write_text(json.dumps(make_payload()))
        _bump_mtime(json_path, 10_000_000_000)
        store.sweep_once()
        self.assertEqual(store.get("beta").status, "evaluated")


class ApiTest(unittest.TestCase):
    """One fixture tree, several runs, full endpoint coverage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        # h10: two variables, CI bands, persistence
        make_run(cls.root, "h10", payload=make_payload(horizon=10))
        # h7: shorter horizon
        make_run(cls.root, "h7", payload=make_payload(horizon=7), horizon=7)
        # neverdrop: ACC stays high (never crosses 0.6)
        payload = make_payload(horizon=10)
        for var in ("z500", "t2m"):
            payload["checkpoints"]["global_best"]["metrics"][var]["acc"]["mean"] = \
                [0.95 - 0.01 * i for i in range(10)]
        make_run(cls.root, "neverdrop", payload=payload)
        # awaiting: trained, no eval
        make_run(cls.root, "awaiting", payload=None, eval_dirs=())

        ext_csv = cls.root / "kai_fixture.csv"
        _write_external_csv(ext_csv)
        cls.external = load_external_csv(ext_csv, id="kai-2p5",
                                         label="KAI fixture", resolution="1p5")
        # NOTE: fixture runs declare resolution_mode 2p5; the external is
        # deliberately tagged 1p5 to exercise resolution_mismatch.

        cls.app = create_app(cls.root, auto_scan=False, externals=[cls.external])
        cls.client = TestClient(cls.app)
        cls.client.__enter__()
        cls.app.state.store.sweep_once()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        cls._tmp.cleanup()

    def test_meta(self) -> None:
        meta = self.client.get("/api/meta").json()
        self.assertEqual(meta["schema_version"], SCHEMA_VERSION)
        self.assertEqual(meta["variables_union"], ["t2m", "z500"])
        self.assertEqual(meta["units"]["source"], "declared")
        self.assertEqual(meta["external_baselines"][0]["id"], "kai-2p5")

    def test_runs_list_and_detail(self) -> None:
        rows = self.client.get("/api/runs").json()["runs"]
        by_id = {r["run_id"]: r for r in rows}
        self.assertEqual(by_id["h10"]["status"], "evaluated")
        self.assertEqual(by_id["awaiting"]["status"], "trained")
        self.assertIsNone(by_id["awaiting"]["headline"])
        self.assertEqual(by_id["h7"]["headline"]["lead_last"], 7)
        detail = self.client.get("/api/runs/h10").json()
        self.assertEqual(detail["schema_version"], SCHEMA_VERSION)
        self.assertEqual(detail["evaluations"][0]["horizon"], 10)
        self.assertEqual(self.client.get("/api/runs/nope").status_code, 404)

    def test_architecture_diagnostics_maps(self) -> None:
        arch = self.client.get("/api/runs/h10/architecture").json()["architecture"]
        self.assertEqual(arch["hidden_dim"], 128)
        self.assertEqual(arch["params_millions"], 2.474)
        diag = self.client.get("/api/runs/h10/diagnostics").json()
        self.assertIsNone(diag["diagnostics"])      # fixtures have none
        maps = self.client.get("/api/runs/h10/maps").json()
        self.assertIsNone(maps["qualitative"])

    def test_artifact_passthrough_and_traversal(self) -> None:
        resp = self.client.get("/api/artifacts/h10/config_resolved.yaml")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("ETag", resp.headers)
        # path traversal is rejected at the server (encoded forms bypass the
        # client's own URL normalization and reach the handler)
        bad = self.client.get("/api/artifacts/h10/..%2F..%2Fetc%2Fpasswd")
        self.assertEqual(bad.status_code, 404)
        bad2 = self.client.get("/api/artifacts/h10/%2e%2e/h7/config_resolved.yaml")
        self.assertEqual(bad2.status_code, 404)
        bad3 = self.client.get("/api/artifacts/h10//etc/passwd")
        self.assertEqual(bad3.status_code, 404)
        # a literal ../ is normalized by the CLIENT before the request is sent;
        # it arrives as /api/artifacts/h7/... and legitimately serves that run.
        normalized = self.client.get("/api/artifacts/h10/../h7/config_resolved.yaml")
        self.assertEqual(normalized.status_code, 200)

    def test_overlay_intersects_mixed_horizons(self) -> None:
        payload = self.client.get(
            "/api/overlay",
            params={"runs": "h10,h7", "variable": "z500", "metric": "rmse",
                    "include": "persistence"}).json()
        self.assertEqual(payload["lead_times"], list(range(1, 8)))
        types = [w["type"] for w in payload["warnings"]]
        self.assertIn("horizon_truncated", types)
        h10 = next(s for s in payload["series"] if s["run_id"] == "h10")
        self.assertEqual(len(h10["values"]), 7)
        self.assertEqual(h10["values"][0], 10.0)      # verbatim stored value
        self.assertTrue(payload["baselines"])          # persistence included

    def test_overlay_external_resolution_mismatch(self) -> None:
        payload = self.client.get(
            "/api/overlay",
            params={"runs": "h10", "variable": "z500", "metric": "rmse",
                    "include": "external:kai-2p5"}).json()
        types = [w["type"] for w in payload["warnings"]]
        self.assertIn("resolution_mismatch", types)   # warn-and-allow (Q3)
        ext = next(b for b in payload["baselines"] if b.get("external"))
        self.assertEqual(ext["values"][0], 5.0)

    def test_best_per_variable(self) -> None:
        payload = self.client.get(
            "/api/best", params={"runs": "h10,neverdrop", "lead": 10,
                                 "metric": "rmse"}).json()
        self.assertTrue(payload["derived"])
        z500 = payload["variables"]["z500"]
        self.assertIn(z500["run_id"], ("h10", "neverdrop"))
        self.assertEqual(z500["value"], 100.0)         # stored, not recomputed
        self.assertIsNotNone(z500["within_ci_of_runner_up"])

    def test_ranking_day_k(self) -> None:
        body = {"runs": ["h10", "h7", "awaiting"], "variable": "z500",
                "metric": "rmse", "lead": {"type": "day", "k": 10}}
        payload = self.client.post("/api/ranking", json=body).json()
        rows = {r["run_id"]: r for r in payload["rows"]}
        self.assertIn("not_comparable", rows["h7"]["flags"])       # H=7 has no day 10
        self.assertIn("lead_outside_horizon", rows["h7"]["flags"])
        self.assertIn("no_valid_evaluation", rows["awaiting"]["flags"])
        self.assertEqual(rows["h10"]["rank"], 1)
        self.assertEqual(rows["h10"]["delta_vs_best_pct"], 0.0)
        self.assertIsNone(rows["h7"]["rank"])
        # determinism
        again = self.client.post("/api/ranking", json=body).json()
        self.assertEqual(payload["rows"], again["rows"])

    def test_ranking_mean_leads_is_derived(self) -> None:
        body = {"runs": ["h10", "h7"], "variable": "z500", "metric": "rmse",
                "lead": {"type": "mean_leads"}}
        payload = self.client.post("/api/ranking", json=body).json()
        self.assertIn("value", payload["derived"])
        for row in payload["rows"]:
            self.assertIn("derived", row["flags"])
        types = [w["type"] for w in payload["warnings"]]
        self.assertIn("horizon_truncated", types)     # means over different horizons

    def test_ranking_acc_crossing(self) -> None:
        body = {"runs": ["h10", "neverdrop"], "variable": "z500",
                "lead": {"type": "acc_crossing", "threshold": 0.6}}
        payload = self.client.post("/api/ranking", json=body).json()
        rows = {r["run_id"]: r for r in payload["rows"]}
        self.assertIn("never_crossed", rows["neverdrop"]["flags"])
        self.assertEqual(rows["neverdrop"]["rank"], 1)     # never crossing = best
        self.assertIsNotNone(rows["h10"]["value"])          # crossed at some stored lead

    def test_ranking_external_row(self) -> None:
        body = {"runs": ["h10"], "variable": "z500", "metric": "rmse",
                "lead": {"type": "day", "k": 10}, "include_external": True}
        payload = self.client.post("/api/ranking", json=body).json()
        ext = next(r for r in payload["rows"] if r["run_id"] == "kai-2p5")
        self.assertIn("external", ext["flags"])
        self.assertIn("resolution_mismatch", ext["flags"])
        self.assertEqual(ext["value"], 50.0)

    def test_problems_endpoint(self) -> None:
        payload = self.client.get("/api/problems").json()
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertIsInstance(payload["problems"], list)

    def test_sse_stream_hello(self) -> None:
        # once=1 bounds the stream (this TestClient buffers whole responses,
        # so the default infinite SSE stream would deadlock the test).
        resp = self.client.get("/api/events", params={"once": 1})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.headers["content-type"].startswith("text/event-stream"))
        self.assertIn("event: hello", resp.text)


if __name__ == "__main__":
    unittest.main()
