# 04 — Build plan

Build in phases. **Each phase is independently testable and must pass its tests
before the next begins.** Tests are `unittest`-style under `tests/`, runnable by
`pytest` (repo convention: no `conftest.py`, no `import pytest`). Run tests
against the **real repo runs** where noted, and against small synthetic fixtures
for the failure paths.

## P1 — Contract + ingest

**Deliverables:** `dashboard/contract.py` (dataclasses + validators for
`gw-run/1`, including the `Problem` object) and `dashboard/ingest.py` (the
`RunRecord` builder; lift the config extractor from
`scripts/build_dashboard_data.py:60-99`).

**CLI:** `python -m dashboard.ingest --once --json` dumps all records to stdout.

**Tests (against the real repo):**

- Expect **≥ 11 evaluated runs** today, including
  `dense_l3k24_curriculum_S2toS10_3ep_initckpt` — assert it has **horizon 10**,
  **6 variables**, a **persistence** block, and **epoch 27**.
- Assert its z500 series: `series.z500.rmse.mean[-1] ≈ 683.826` (last lead).
  *(This is the canary that proves values are read verbatim from the JSON.)*
- A run lacking eval dirs (e.g. the in-progress 1p5 run) ⇒ status
  `trained`/`in_progress`, **no training record, zero fabricated fields**; the
  frontend will surface it as "awaiting evaluation".
- A run with two `evaluation*` dirs ⇒ **two `EvalRecord`s**, exactly one flagged
  `is_primary`, both selectable (Q1).

**Tests (synthetic fixtures for failure paths):**

- Corrupt-JSON fixture ⇒ exactly **one `parse_error` Problem**, run marked
  `invalid`, no fabricated series.
- Truncated-array fixture (a `mean` array shorter than `lead_times`) ⇒
  **`axis_mismatch`**.
- Older-JSON fixture missing top-level keys ⇒ **`schema_drift` warning**, ingest
  continues.
- NaN in a metric ⇒ **`warning`**, value passed through as `null`, not
  interpolated.

**Acceptance:** every real run ingests to a `RunRecord`; every fabricated-value
temptation instead produces a `Problem`.

## P2 — Scanner + API

**Deliverables:** the polling `scanner.py` (fingerprint + cache + grace window),
`ranking.py` (the new backend ranking / best / delta / intersection logic), the
FastAPI `app.py` with all endpoints + SSE + the path-traversal-proof artifact
passthrough.

**Tests:**

- **Fingerprint invalidation:** copy a run into a tmp fixture tree, touch its
  metrics JSON → a `run_updated` event fires and the record reflects the change.
- **Removal tombstone:** remove a fixture run → `run_removed` + tombstone
  Problem.
- **Artifact passthrough:** correct `ETag` from mtime; a `../` traversal attempt
  is rejected; only files inside the resolved run dir are served.
- **Overlay intersection:** synthetic mixed-horizon fixtures (H=10 vs H=7) →
  common leads `1..7` + a `horizon_truncated` warning listing both horizons.
- **Ranking determinism:** stable order for a fixed criterion; a `day-k`
  criterion with `k` beyond a run's horizon flags that run `not_comparable`;
  `mean_leads` and `acc_crossing` are labeled `derived`.
- **Best-per-variable:** argmin for RMSE / argmax for ACC over stored values;
  `within_ci_of_runner_up` reflects stored CI bounds.

**Acceptance:** the API answers every endpoint from real runs; ranking / best /
delta are computed only from stored values; no endpoint recomputes a metric or
touches `runs/`.

## P3 — Frontend

**Deliverables:** the static SPA under `dashboard/static/` wired to the API —
Runs table (status chips incl. "awaiting evaluation" for `trained` runs), Run
detail (an evaluation selector when a run has >1 evaluation; curves for the
selected eval + architecture + diagnostics + maps gallery; an "awaiting
evaluation" state with architecture only — no curves, no training plots — when
there is no valid eval), Compare (overlay + one ranking table + best strip +
warning banners), Problems panel with header badge. All option lists come from
the API; the frontend performs zero arithmetic. SSE-driven refresh;
empty-`runs/` empty-state.

**Tests:** a manual checklist plus a Python smoke test that hits **every**
endpoint via FastAPI `TestClient` and asserts 200 + schema-shaped payloads.

**Acceptance:** the four invariants hold end-to-end; horizon and variable lists
are discovered from the backend, never hardcoded in the UI; a run with a
`Problem` renders as invalid with the reason shown verbatim.

## P4 — Polish

**Deliverables:**

- `/api/best` CI-overlap flags refined.
- `acc_crossing` criterion (Open Q4).
- `resolution_mismatch` warnings for `kai-2p5` overlaid on non-2.5° runs
  (decided: warn-and-allow, Q3).
- Config for scan roots (default `[runs/]`, Open Q2) and additional external
  reference CSVs.

**Acceptance:** the provisional defaults for the open questions are implemented
and configurable; nothing in P1–P3 regressed.

## Definition of done

- All four invariants hold, demonstrably, in the P3 smoke test and the P1 canary
  assertions.
- No file under `src/`, `scripts/`, `configs/`, or `runs/` was modified.
- New dependencies live only in `dashboard/requirements.txt`.
- Every requested feature (per-run results, overlay, best-per-variable, ranking
  table, diagnostics, qualitative maps, architecture info, Problems, live
  updates) is served by a named endpoint and rendered by a view.
