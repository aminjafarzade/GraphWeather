"""P3 smoke test: every endpoint answers with a schema-shaped payload, and the
static SPA is served. Runs against the REAL repo runs/ when present (read-only).

unittest-style, runnable by pytest.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from dashboard.app import create_app  # noqa: E402
from dashboard.contract import SCHEMA_VERSION  # noqa: E402

RUNS_ROOT = PROJECT_ROOT / "runs"


@unittest.skipUnless(RUNS_ROOT.is_dir(), "real runs/ not available")
class SmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app(RUNS_ROOT, auto_scan=False)
        cls.client = TestClient(cls.app)
        cls.client.__enter__()
        cls.app.state.store.sweep_once()
        cls.runs = cls.client.get("/api/runs").json()["runs"]
        cls.evaluated = [r for r in cls.runs if r["status"] == "evaluated"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_static_spa_served(self) -> None:
        index = self.client.get("/")
        self.assertEqual(index.status_code, 200)
        self.assertIn("gw-dashboard", index.text)
        self.assertEqual(self.client.get("/main.js").status_code, 200)
        self.assertEqual(self.client.get("/styles.css").status_code, 200)

    def test_every_endpoint_answers(self) -> None:
        self.assertTrue(self.evaluated, "no evaluated runs on disk")
        rid = self.evaluated[0]["run_id"]
        variable = next(iter(self.evaluated[0]["headline"]["variables"]))
        endpoints = [
            ("GET", "/api/meta", None),
            ("GET", "/api/runs", None),
            ("GET", f"/api/runs/{rid}", None),
            ("GET", f"/api/runs/{rid}/architecture", None),
            ("GET", f"/api/runs/{rid}/diagnostics", None),
            ("GET", f"/api/runs/{rid}/maps", None),
            ("GET", f"/api/overlay?runs={rid}&variable={variable}&metric=rmse"
                    "&include=persistence", None),
            ("GET", f"/api/best?runs={rid}&lead={self.evaluated[0]['headline']['lead_last']}"
                    "&metric=rmse", None),
            ("POST", "/api/ranking",
             {"runs": [rid], "variable": variable, "metric": "rmse",
              "lead": {"type": "mean_leads"}}),
            ("GET", "/api/problems", None),
            ("GET", "/api/events?once=1", None),
        ]
        for method, url, body in endpoints:
            resp = self.client.post(url, json=body) if method == "POST" \
                else self.client.get(url)
            self.assertEqual(resp.status_code, 200, f"{method} {url}: {resp.text[:200]}")
            if "event-stream" not in resp.headers.get("content-type", ""):
                payload = resp.json()
                self.assertEqual(payload.get("schema_version", SCHEMA_VERSION),
                                 SCHEMA_VERSION, url)

    def test_invariants_hold_end_to_end(self) -> None:
        # I4: the lead axis is discovered, not hardcoded
        rid = self.evaluated[0]["run_id"]
        detail = self.client.get(f"/api/runs/{rid}").json()
        ev = next(e for e in detail["evaluations"] if e["is_primary"])
        self.assertEqual(len(ev["lead_times"]), ev["horizon"])
        # I2: overlay serves the same verbatim values as the record
        variable = ev["variables"][0]
        ov = self.client.get(f"/api/overlay?runs={rid}&variable={variable}"
                             "&metric=rmse").json()
        run_series = next(s for s in ov["series"] if s["run_id"] == rid)
        self.assertEqual(run_series["values"],
                         ev["series"][variable]["rmse"]["mean"])
        # I3: problems are structured and verbatim
        problems = self.client.get("/api/problems").json()["problems"]
        for p in problems:
            for key in ("run_id", "artifact", "path", "expected", "found",
                        "reason", "severity", "detected_at", "schema_version"):
                self.assertIn(key, p)


if __name__ == "__main__":
    unittest.main()
