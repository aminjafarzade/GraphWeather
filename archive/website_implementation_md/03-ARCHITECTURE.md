# 03 — Architecture

## Stack

- **Backend:** Python 3.10, **FastAPI + uvicorn** (optionally `sse-starlette`
  for the event stream). New deps go in `dashboard/requirements.txt`, **not** the
  repo's `requirements.txt`. The service **imports nothing from `src/`** (no
  torch, no GPU) — it reads artifacts only, which is what guarantees read-only /
  no-eval-re-execution.
- **Frontend:** a no-build static SPA — plain ES modules + **Plotly via CDN** —
  served by FastAPI `StaticFiles` under `dashboard/static/`. No build step; this
  matches the HPC no-toolchain reality and the single-user scope. Reuse the
  palette, theming, and tab structure of
  `experiments/experiment_dashboard.html` (8-slot categorical palette,
  light/dark theme, Problems styling).

Rationale: the repo is pure Python with numpy/pandas/PyYAML already pinned and
no JS toolchain; the ingest logic can be lifted from the already-working
`scripts/build_dashboard_data.py`. A React/Vite app would add a build step for
no benefit at one-user scale.

## Package layout

```
dashboard/
  CLAUDE.md
  requirements.txt          # fastapi, uvicorn, (sse-starlette)
  docs/                     # these files
  contract.py               # gw-run/1 dataclasses + validators
  ingest.py                 # RunRecord builder (lifts build_dashboard_data.py:60-99)
  scanner.py                # polling sweep + fingerprint + cache + SSE fan-out
  ranking.py                # NEW backend logic: ranking / best / delta / intersection
  app.py                    # FastAPI app, routes, StaticFiles
  static/                   # ES-module SPA + Plotly CDN
tests/                      # unittest-style, runnable by pytest
```

The service writes only to its own cache/log location **outside `runs/`**. Run
directories are opened strictly read-only.

## API surface

Every number in every response is backend-produced; the frontend renders
verbatim. Every response includes `schema_version`.

| Feature | Endpoint | In → Out |
|---|---|---|
| Meta / discovery | `GET /api/meta` | → `{schema_version, variables_union, metrics:["rmse","acc"], external_baselines:[{id,label,resolution,leads}], scan:{root,interval_s,last_sweep}}` |
| List runs | `GET /api/runs` | → summary rows: `{run_id, status, name, tags, architecture summary, primary-eval headline, problem_count}` |
| One run | `GET /api/runs/{id}` | → full `RunRecord` |
| Architecture | `GET /api/runs/{id}/architecture` | → the architecture block, in display groups |
| Diagnostics | `GET /api/runs/{id}/diagnostics` | → `DiagRecord` |
| Maps | `GET /api/runs/{id}/maps` | → metadata + list of map relpaths |
| Artifact passthrough | `GET /api/artifacts/{id}/{relpath}` | → read-only file (PNG, summary.txt), `ETag=mtime`, path-traversal-proof |
| Overlay N runs | `GET /api/overlay?runs=a,b,c&variable=z500&metric=rmse&eval=primary&include=persistence,external:kai-2p5` | → `{lead_times:[common], series:[{run_id,label,values,ci_lower?,ci_upper?}], baselines:[...], warnings:[horizon_truncated, resolution_mismatch, ...]}` |
| Best per variable | `GET /api/best?runs=...&lead=10&metric=rmse` | → `{<var>: {run_id, value, ci, within_ci_of_runner_up}}` |
| Ranking table | `POST /api/ranking` `{runs[], variable, metric, lead, include_external}` | → `[{rank, run_id, value, ci, delta_vs_best_pct, flags}]` + criterion echo |
| Problems feed | `GET /api/problems?since=` | → `[Problem, ...]`, newest first |
| Live updates | `GET /api/events` (SSE) | → `run_added \| run_updated \| run_removed \| problem` |

`lead` in ranking is one of:
`{type:"day", k}` · `{type:"mean_leads"}` · `{type:"acc_crossing", threshold}`.

`eval` (overlay) is `primary` or a specific `eval_id`. Every `evaluation*` dir
is a first-class, independently selectable evaluation per run (Q1), so overlay,
ranking, and best can target any of a run's evaluations, not just the primary.

### Derived-but-allowed backend logic (label it as such)

Ranking order, best-per-variable, `delta_vs_best_pct`, the common-lead
intersection, `mean_leads` (arithmetic mean of stored per-lead values), and
`acc_crossing` (first lead where stored ACC crosses a threshold). All of these
operate **only on stored values** and are labeled `derived` in the response.
`best` uses argmin for RMSE / argmax for ACC over stored values; the
`within_ci_of_runner_up` flag compares stored CI bounds only.

### Forbidden and absent

Any metric recomputation; any interpolation; any averaging beyond what the
criterion explicitly names; running the model; writing into `runs/`.

## Discovery & live update

- **Polling scanner, not inotify** (Lustre). Sweep `runs/*/` every ~20 s
  (configurable). Per run compute a cheap **fingerprint** = `mtime_ns + size` of
  `config_resolved.yaml`, `out.log`, every
  `evaluation*/fixed*_global_best_metrics.json`, and the diag/qual dir mtimes.
  Re-parse only what changed.
- **Partial / still-writing runs:** a run without the metrics JSON is
  `in_progress`/`trained`, never `invalid` for mere absence. A JSON that fails
  to parse gets one `parse_error` Problem and a **grace window (2 sweeps)**
  before being surfaced as invalid — the evaluator writes it in one pass near
  the end, and a scan can race that write. Never render a half-parsed eval.
- **Propagation:** SSE events (`run_added` / `run_updated` / `run_removed` /
  `problem`); the frontend also refetches on reconnect. Removal ⇒ `run_removed`
  + a tombstone Problem (Open Q8).

## Caching

In-process dict keyed `(abs_path, mtime_ns, size)` for parsed JSON/CSV; the
scanner invalidates on fingerprint change. Metrics JSONs are ~100–300 KB and
there are ~11 runs today, so this is trivially memory-resident. Artifact (PNG)
responses use file `ETag`s. No persistent cache needed at this scale; add a disk
cache only if `runs × vars` grows ~100×.

## Frontend

- **Views:**
  - **Runs** — table from `/api/runs` with status chips (`in_progress` →
    "training in progress", `trained` → "awaiting evaluation", `evaluated`,
    `invalid`), problem counts, click → detail.
  - **Run detail** — an **evaluation selector** when the run has more than one
    `evaluation*` dir (all first-class, Q1); RMSE/ACC curves for the selected
    evaluation (persistence dashed, CI bands as stored) + architecture panel +
    diagnostics panels (with "not recorded for this run" placeholders for
    missing conditional artifacts) + maps gallery. A run with **no valid
    evaluation** shows an **"awaiting evaluation"** state — architecture panel
    only, no curves and no training plots.
  - **Compare** — run multiselect chips + variable/metric/lead controls →
    `/api/overlay`; one ranking table → `/api/ranking`; best-per-variable strip
    → `/api/best`; truncation/mismatch warnings as banners.
  - **Problems** — verbatim `Problem` objects, filterable, header badge count.
- **State model:** `{selectedRuns[], evalChoice, variable, metric, leadSelection,
  criterion, includeBaselines[], theme}`. All option lists (variables, leads,
  baselines) come from `/api/meta` + per-run `lead_times` — **nothing hardcoded,
  including the horizon.**
- **Frontend computes nothing.** Every cell, delta, rank, and "best" badge
  arrives from the API; charts plot arrays as received.

## Risks / edge cases

1. Horizon mismatch → intersection + explicit truncation warning.
2. Missing persistence block → baselines omitted + warning, never fabricated.
3. NaNs in stored metrics → gaps in the plot + a `Problem` warning.
4. Same run re-evaluated in place → fingerprint catches the JSON mtime change,
   record updated, `run_updated` emitted.
5. Schema drift (older JSON variants) → tolerant reader + `schema_drift`
   warning.
6. Empty `runs/` → empty list + healthy `/api/meta` + UI empty-state.
7. Junk entries (root files, empty nested dir, `wandb/`) → a dir needs
   `config_resolved.yaml` to count as a run; else ignored silently.
8. Eval dir with `N != 10` → everything keys off JSON `lead_times`; the only
   10-specific strings on disk (summary CSV headers) are never parsed.
9. External baseline resolution mismatch → warn-and-allow with a
   `resolution_mismatch` warning; never block (decided, Q3).
10. Lustre slowness → the sweep is time-boxed; a stale `last_sweep` is shown in
    the UI footer.
