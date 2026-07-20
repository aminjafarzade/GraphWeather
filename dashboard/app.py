"""gw-dashboard FastAPI application.

Read-only over runs/; every number in every response is backend-produced
(invariant I1). Serve with:

    uvicorn dashboard.app:app --host 127.0.0.1 --port 8677

or programmatically via create_app() (tests pass auto_scan=False and sweep
manually).
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import ranking as ranking_mod
from .contract import DECLARED_UNITS, SCHEMA_VERSION
from .external import default_externals
from .ingest import DEFAULT_RUNS_ROOT
from .scanner import RunStore
from .tree import TreeStore

SSE_HEARTBEAT_S = 15.0


def _run_summary(record) -> dict:
    arch = record.architecture
    headline = None
    primary = ranking_mod.select_eval(record, "primary")
    if primary is not None:
        lt = primary.lead_times
        headline = {
            "eval_id": primary.eval_id,
            "lead_first": lt[0],
            "lead_last": lt[-1],
            "variables": {
                var: {
                    "rmse_first": primary.series[var].rmse.mean[0],
                    "rmse_last": primary.series[var].rmse.mean[-1],
                    "acc_first": primary.series[var].acc.mean[0],
                    "acc_last": primary.series[var].acc.mean[-1],
                }
                for var in primary.variables
            },
        }
    return {
        "run_id": record.run_id,
        "name": record.run_id,
        "status": record.status,
        "tags": arch.tags,
        "architecture": {
            "resolution_mode": arch.resolution_mode,
            "hidden_dim": arch.hidden_dim,
            "params_millions": arch.params_millions,
        },
        "evaluations": [
            {"eval_id": e.eval_id, "is_primary": e.is_primary,
             "is_valid": e.is_valid, "horizon": e.horizon}
            for e in record.evaluations
        ],
        "headline": headline,
        "stale": any(p.reason == "stale_partial" for p in record.problems),
        "problem_count": {
            "error": sum(1 for p in record.problems if p.severity == "error"),
            "warning": sum(1 for p in record.problems if p.severity == "warning"),
        },
        "schema_version": record.schema_version,
    }


def create_app(runs_root: Optional[Path] = None, *, scan_interval_s: float = 20.0,
               auto_scan: bool = True, externals: Optional[list] = None,
               tree_path: Optional[Path] = None) -> FastAPI:
    store = RunStore(runs_root or DEFAULT_RUNS_ROOT)
    ext_list = default_externals() if externals is None else externals
    tree_store = TreeStore(tree_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = None
        if auto_scan:
            await asyncio.to_thread(store.sweep_once)   # first snapshot before serving
            task = asyncio.create_task(store.run_forever(scan_interval_s))
        else:
            store.attach_loop(asyncio.get_running_loop())
        yield
        if task is not None:
            task.cancel()

    app = FastAPI(title="gw-dashboard", lifespan=lifespan)
    app.state.store = store
    app.state.externals = ext_list

    # -- meta -----------------------------------------------------------------

    @app.get("/api/meta")
    def meta() -> dict:
        records = store.snapshot()
        variables: set = set()
        for r in records:
            for e in r.evaluations:
                if e.is_valid:
                    variables.update(e.variables)
        problems = store.problems()
        return {
            "schema_version": SCHEMA_VERSION,
            "problem_totals": {
                "error": sum(1 for p in problems if p.severity == "error"),
                "warning": sum(1 for p in problems if p.severity == "warning"),
            },
            "variables_union": sorted(variables),
            "metrics": ["rmse", "acc"],
            "units": {"source": "declared", "table": DECLARED_UNITS},
            "external_baselines": [x.to_meta() for x in ext_list],
            "scan": {"root": str(store.runs_root),
                     "interval_s": scan_interval_s if auto_scan else None,
                     "last_sweep": store.last_sweep,
                     "sweep_count": store.sweep_count},
        }

    # -- runs -----------------------------------------------------------------

    @app.get("/api/runs")
    def list_runs() -> dict:
        rows = sorted((_run_summary(r) for r in store.snapshot()),
                      key=lambda r: r["run_id"])
        return {"schema_version": SCHEMA_VERSION, "runs": rows}

    def _get_record(run_id: str):
        record = store.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        return record

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        return _get_record(run_id).to_dict()

    @app.get("/api/runs/{run_id}/architecture")
    def get_architecture(run_id: str) -> dict:
        record = _get_record(run_id)
        return {"schema_version": SCHEMA_VERSION, "run_id": run_id,
                "architecture": asdict(record.architecture)}

    @app.get("/api/runs/{run_id}/diagnostics")
    def get_diagnostics(run_id: str) -> dict:
        record = _get_record(run_id)
        return {"schema_version": SCHEMA_VERSION, "run_id": run_id,
                "diagnostics": asdict(record.diagnostics) if record.diagnostics else None}

    @app.get("/api/runs/{run_id}/maps")
    def get_maps(run_id: str) -> dict:
        record = _get_record(run_id)
        return {"schema_version": SCHEMA_VERSION, "run_id": run_id,
                "qualitative": asdict(record.qualitative) if record.qualitative else None}

    # -- artifact passthrough ---------------------------------------------------

    @app.get("/api/artifacts/{run_id}/{relpath:path}")
    def get_artifact(run_id: str, relpath: str, request: Request):
        record = _get_record(run_id)
        base = Path(record.path).resolve()
        target = (base / relpath).resolve()
        if base != target and base not in target.parents:
            raise HTTPException(status_code=404, detail="path escapes the run directory")
        if not target.is_file():
            raise HTTPException(status_code=404, detail="no such artifact")
        stat = target.stat()
        etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag})
        return FileResponse(target, headers={
            "ETag": etag,
            "Cache-Control": "no-cache",
        })

    # -- comparison endpoints (backend-computed; invariant I1) ------------------

    @app.get("/api/overlay")
    def get_overlay(
        runs: str = Query(..., description="comma-separated run ids"),
        variable: str = Query(...),
        metric: str = Query("rmse"),
        eval: str = Query("primary"),
        include: str = Query("", description="comma list: persistence,external:<id>"),
    ) -> dict:
        if metric not in ranking_mod.METRICS:
            raise HTTPException(status_code=422, detail=f"metric must be one of {ranking_mod.METRICS}")
        payload = ranking_mod.overlay(
            store.snapshot(),
            [r for r in runs.split(",") if r],
            variable, metric,
            eval_choice=eval,
            include=[t for t in include.split(",") if t],
            externals=app.state.externals,
        )
        payload["schema_version"] = SCHEMA_VERSION
        return payload

    @app.get("/api/best")
    def get_best(
        runs: str = Query(...),
        lead: int = Query(...),
        metric: str = Query("rmse"),
        eval: str = Query("primary"),
        include: str = Query("", description="comma list: external:<id>"),
    ) -> dict:
        if metric not in ranking_mod.METRICS:
            raise HTTPException(status_code=422, detail=f"metric must be one of {ranking_mod.METRICS}")
        payload = ranking_mod.best_per_variable(
            store.snapshot(), [r for r in runs.split(",") if r],
            lead=lead, metric=metric, eval_choice=eval,
            externals=app.state.externals,
            include=[t for t in include.split(",") if t])
        payload["schema_version"] = SCHEMA_VERSION
        return payload

    @app.post("/api/ranking")
    async def post_ranking(request: Request) -> dict:
        criterion = await request.json()
        for key in ("runs", "variable"):
            if key not in criterion:
                raise HTTPException(status_code=422, detail=f"criterion missing {key!r}")
        payload = ranking_mod.rank(store.snapshot(), criterion,
                                   externals=app.state.externals)
        payload["schema_version"] = SCHEMA_VERSION
        return payload

    # -- ideas (analysis notes authored in dashboard/ideas.json; read per request
    #    so edits appear without a restart) ----------------------------------------

    @app.get("/api/ideas")
    def get_ideas() -> dict:
        path = Path(__file__).resolve().parent / "ideas.json"
        if not path.is_file():
            return {"schema_version": SCHEMA_VERSION, "ideas": []}
        try:
            payload = json.loads(path.read_text(errors="replace"))
        except Exception as exc:  # noqa: BLE001 — surface, never guess (I3)
            raise HTTPException(status_code=500, detail=f"ideas.json unreadable: {exc}")
        ideas = payload.get("ideas") if isinstance(payload, dict) else None
        return {"schema_version": SCHEMA_VERSION, "ideas": ideas if isinstance(ideas, list) else []}

    # -- experiment lineage tree (user-authored; persisted OUTSIDE runs/) ---------

    app.state.tree = tree_store

    @app.get("/api/tree")
    def get_tree() -> dict:
        return {"schema_version": SCHEMA_VERSION, "nodes": tree_store.nodes()}

    @app.post("/api/tree/nodes")
    async def create_tree_node(request: Request) -> dict:
        fields = await request.json()
        try:
            node = tree_store.create(fields if isinstance(fields, dict) else {})
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"schema_version": SCHEMA_VERSION, "node": node}

    @app.patch("/api/tree/nodes/{node_id}")
    async def update_tree_node(node_id: str, request: Request) -> dict:
        fields = await request.json()
        try:
            node = tree_store.update(node_id, fields if isinstance(fields, dict) else {})
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown node {node_id!r}")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"schema_version": SCHEMA_VERSION, "node": node}

    @app.delete("/api/tree/nodes/{node_id}")
    def delete_tree_node(node_id: str) -> dict:
        try:
            node = tree_store.delete(node_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown node {node_id!r}")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"schema_version": SCHEMA_VERSION, "deleted": node}

    # -- problems ----------------------------------------------------------------

    @app.get("/api/problems")
    def get_problems(since: str = Query("")) -> dict:
        problems = [asdict(p) for p in store.problems()]
        if since:
            problems = [p for p in problems if p["detected_at"] > since]
        return {"schema_version": SCHEMA_VERSION, "problems": problems}

    # -- SSE ---------------------------------------------------------------------

    @app.get("/api/events")
    async def events(once: bool = False):
        queue = store.subscribe()

        async def gen():
            try:
                yield "event: hello\ndata: {}\n\n"
                if once:
                    # bounded mode: drain whatever is already queued and close
                    # (used by tests; browsers use the default infinite stream)
                    while not queue.empty():
                        event = queue.get_nowait()
                        yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                    return
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=SSE_HEARTBEAT_S)
                        yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"
            finally:
                store.unsubscribe(queue)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    # -- static frontend -----------------------------------------------------------

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir() and any(static_dir.iterdir()):
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


def _app_from_env() -> FastAPI:
    """Configuration via environment (P4 / open Q2):
    GW_DASH_RUNS_ROOT      runs root to watch (default: repo runs/)
    GW_DASH_SCAN_INTERVAL  sweep interval seconds (default 20)
    GW_DASH_EXTERNAL_CSV   extra external baselines, ';'-separated entries of
                           "path,id,label,resolution" (kai_2p5.csv is built in)
    """
    runs_root = os.environ.get("GW_DASH_RUNS_ROOT")
    interval = float(os.environ.get("GW_DASH_SCAN_INTERVAL", "20"))
    externals = None
    spec = os.environ.get("GW_DASH_EXTERNAL_CSV", "").strip()
    if spec:
        from .external import default_externals as _defaults, load_external_csv
        externals = _defaults()
        for entry in spec.split(";"):
            parts = [p.strip() for p in entry.split(",")]
            if len(parts) == 4:
                ext = load_external_csv(Path(parts[0]), id=parts[1],
                                        label=parts[2], resolution=parts[3])
                if ext is not None:
                    externals.append(ext)
    return create_app(Path(runs_root) if runs_root else None,
                      scan_interval_s=interval, externals=externals)


app = _app_from_env()


def main() -> None:
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="gw-dashboard server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8677)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
