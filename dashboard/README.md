# gw-dashboard

Read-only results dashboard over `runs/`. See `CLAUDE.md` and `docs/` for the
contract and architecture.

## Run

    ~/miniconda3/envs/graphweather-cu128/bin/python -m dashboard.app --port 8677
    # then open http://127.0.0.1:8677  (SSH tunnel: ssh -L 8677:127.0.0.1:8677 <host>)

Configuration (env): `GW_DASH_RUNS_ROOT`, `GW_DASH_SCAN_INTERVAL`,
`GW_DASH_EXTERNAL_CSV="path,id,label,resolution[;...]"`.

## Test

    python -m unittest tests.test_dashboard_ingest tests.test_dashboard_api tests.test_dashboard_smoke

## CLI ingest (no server)

    python -m dashboard.ingest --once            # table
    python -m dashboard.ingest --once --json     # full gw-run/1 records
