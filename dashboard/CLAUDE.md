# gw-dashboard — build instructions for Claude

You are building **gw-dashboard**: a **read-only** results dashboard over the
GraphWeather repo at `/lustre/home/ziya/GNN/GraphWeather5p625`. It is a new
top-level package `dashboard/` (FastAPI backend + a no-build static frontend).
It reads experiment artifacts under `runs/` and presents evaluation results,
comparisons, diagnostics, qualitative maps, and architecture info.

## The four invariants (a violation is a bug, not a style choice)

- **I1 — Backend is the single source of truth.** The frontend never computes,
  derives, averages, rescales, or ranks a number. If a value appears in the UI,
  the backend produced it and the frontend rendered it verbatim.
- **I2 — Never re-implement or alter the metrics.** All RMSE / ACC / CI /
  persistence values are read **verbatim** from
  `runs/<run>/<evaldir>/fixed{N}_global_best_metrics.json`. Do not recompute
  metrics, do not run the model, do not write into `runs/`, `src/`, `scripts/`,
  or `configs/`.
- **I3 — Strict validated contract, no hallucination.** Every run is validated
  against schema `gw-run/1` on ingest. On any mismatch, emit a structured
  `Problem` object and surface the run as *invalid* with that reason. Never
  drop a bad run silently, never guess/interpolate/fill a missing value.
  **Reporting a format problem is always preferred over inventing data.**
- **I4 — Horizon is per-run data, never a constant.** The lead-time axis is the
  JSON `lead_times` field. Nothing assumes 10. Multi-run views intersect the
  common leads and attach a truncation warning.

## Read these in order before writing any code

1. `docs/00-OVERVIEW.md` — what this is, features, non-goals, open questions.
2. `docs/01-GROUND-TRUTH.md` — verified facts about the artifacts, with file
   citations. **This is the anti-hallucination reference. Trust it, and
   re-verify against the live repo before relying on any detail.**
3. `docs/02-DATA-CONTRACT.md` — the `gw-run/1` schema and validation rules.
4. `docs/03-ARCHITECTURE.md` — stack, package layout, API surface, scanner,
   caching, frontend.
5. `docs/04-BUILD-PLAN.md` — the phased build order and its tests.

## Rules of engagement

- **Verify, don't assume.** The ground-truth doc was produced by a repo analysis
  and is believed correct, but the repo is live. Before you depend on a fact
  (a file path, a JSON key, a run's numbers), open the real file and confirm.
- **If the repo contradicts these docs, the repo wins** — stop and report the
  contradiction rather than coding around it.
- **The open questions in `docs/00-OVERVIEW.md`:** Q1, Q3, and Q5 are now
  decided (marked there — first-class selectable evaluations; warn-only on
  baseline resolution mismatch; "awaiting evaluation" only, no training curves).
  The rest keep their provisional defaults; if an unanswered one would change
  your design, ask before locking it in.
- Do not add dependencies to the repo's `requirements.txt`; put new deps in
  `dashboard/requirements.txt`. Import nothing from `src/` (no torch).
- Tests are `unittest`-style under `tests/`, runnable by `pytest` (repo
  convention: no `conftest.py`, no `import pytest`).
