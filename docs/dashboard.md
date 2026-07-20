# Results dashboard

`dashboard/` is a standalone FastAPI + static-SPA app that scans `runs/` and
presents each run's metrics, diagnostics, and qualitative maps. It **never
imports `src/`** (no torch) and **writes nothing** into `runs/` — it only reads.

## Run it

```bash
pip install -r dashboard/requirements.txt      # separate from the main deps
python -m dashboard.ingest --once --json        # dump the ingested run records
# or serve the API + SPA (see dashboard/docs for the server entry point)
```

## The `gw-run/1` data contract (must never regress)

The dashboard's detailed, self-contained contract lives in
[`../dashboard/docs/`](../dashboard/docs/) (`00`–`04`, with
`02-DATA-CONTRACT.md` as the schema). The invariants any run-storage change must
preserve (`CONTRIBUTING.md §10`):

1. **A run = a directory containing `config_resolved.yaml`** at its root. That
   file is the run-detection key.
2. **Evaluations are subdirs named `evaluation*`** containing
   `fixed{N}_global_best_metrics.json`. All `evaluation*` dirs are first-class.
3. **Horizon `N` is per-run data** — never hardcode 10.
4. Metrics are read **verbatim** from the JSON; the dashboard fabricates nothing.

Any restructure touching these ships the matching `dashboard/` change (scanner
path or `dashboard/docs/02-DATA-CONTRACT.md`) in the **same commit**.

## Tests

Six no-torch tests (`tests/test_dashboard_*.py`) are the dashboard-contract smoke
and run in the CI fast lane (`CONTRIBUTING.md §9`). A run **without** the optional
`run.json` manifest still ingests normally — `run.json` is additive and the
scanner keys only on `config_resolved.yaml`.
