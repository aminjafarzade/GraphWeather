# 00 — Overview

## What we are building

A dashboard for reviewing GraphCast-style hierarchical GNN weather-forecasting
experiments. Each experiment is an autoregressive model evaluated with a
multi-day rollout (currently 10 daily steps, but the horizon is variable). The
dashboard has two parts:

- a **backend service** that watches the `runs/` folder, ingests each experiment
  read-only, validates it against a strict contract, and serves its results
  through an API;
- a **frontend** that displays runs individually or overlaid, shows best score
  per variable, rollout plots over the (variable) forecast horizon, an
  architecture panel, diagnostics, qualitative maps, and one ranking table
  driven by the current selection.

## Features

- **Per-run results** — RMSE and ACC per variable per lead time, with bootstrap
  confidence bands when present, and the persistence baseline.
- **Comparisons** — overlay N selected runs on RMSE-vs-lead and ACC-vs-lead
  charts; a comparison table with the best value per column highlighted and a
  percentage delta versus a chosen reference run.
- **Best-per-variable** — for the current selection, which run wins each
  variable at a chosen lead/metric.
- **Ranking table** — one ranked table given a criterion (variable, metric, and
  a lead selector: day-k, mean-over-leads, or ACC-threshold-crossing).
- **Diagnostics** — per-run deep dive: rollout curve, variance ratio, power
  spectrum (model vs truth, to reveal over-smoothing at long lead times), plus
  the raw diagnostic tables where present.
- **Qualitative maps** — serve the pre-rendered bias/rollout map PNGs and their
  metadata where a run has them.
- **Architecture info** — resolution, grid/graph structure, hidden dim, heads,
  connectivity, block counts, rollout curriculum, learning-rate schedule,
  forcings, parameter count, tags — cleanly formatted.
- **Problems panel** — every validation failure as a first-class, verbatim
  entry, with a badge count in the header.
- **Live updates** — new / changed / removed runs propagate to the UI.

## The four invariants

See `CLAUDE.md`. In short: backend is the single source of truth (I1); never
alter or recompute the metrics and never run the model (I2); strict validated
contract with structured format-errors instead of hallucination (I3); horizon
is per-run data, never hardcoded to 10 (I4).

## Two honest limitations (consequences of "don't touch eval")

1. **No climatology-forecast baseline curve exists in the artifacts.** Only the
   per-run **persistence** baseline and the external reference CSV are stored;
   climatology appears only as the anomaly reference inside ACC. The UI shows
   persistence + external references, plus the implicit reading that "ACC = 0
   corresponds to climatology skill." A climatology RMSE/ACC curve would require
   new eval computation and is out of scope.
2. **Qualitative maps are PNG-only in v1.** The evaluator saves no field arrays,
   only scalar metrics and line-plot PNGs. Interactive re-colormapped maps would
   require an offline, user-triggered re-run of the qualitative stage with
   `--save_arrays`; that is out of scope for v1. Where the pipeline already ran
   the qualitative stage, the run contains pre-rendered PNGs + metadata that we
   serve directly.

## Non-goals (do not build)

- Authentication or multi-user state (assume localhost + SSH tunnel, single
  user, unless told otherwise).
- Running any model inference or writing into `runs/`.
- Re-computing metrics, or adding a climatology-forecast baseline curve.
- Interactive map re-rendering from arrays (PNG serving only in v1).

## Open questions

Q1, Q3, and Q5 are **decided** (marked below). The rest keep their provisional
defaults — safe to start with; confirm with the user where noted. None change
the core contract.

1. **Multiple eval dirs per run** (e.g. `evaluation_test_weekly52` and
   `evaluation_test_weekly52_S5ckpt`): treat each as a first-class,
   selectable evaluation, or only the canonical one?
   **DECIDED (yes):** ingest every `evaluation*` dir as a first-class,
   selectable evaluation; primary = priority-list match; the run detail and the
   overlay/ranking views let you pick which evaluation to use.
2. **Scan roots**: watch only `runs/`, or also `experiments/`?
   *Default: `runs/` only, configurable.*
3. **External baseline resolution mismatch**: `kai_2p5.csv` is 2.5°-only —
   when overlaid on a 1.5° run, block or warn-and-allow?
   **DECIDED (warn):** warn-and-allow — attach a `resolution_mismatch` warning
   and still overlay the baseline; never block it.
4. **Ranking criteria**: beyond {metric, variable, day-k / mean-over-leads}, do
   you want derived criteria like "first day ACC < 0.6"?
   *Default: include it, flagged as `derived`.*
5. **In-progress runs**: show training-time curves for runs without eval, or
   list them as "awaiting evaluation" only?
   **DECIDED (awaiting eval only):** do not build a training-curve view and do
   not ingest training metrics. A run without a valid evaluation is surfaced
   only by status — `trained` shows as "awaiting evaluation", `in_progress` as
   "training in progress". Its architecture panel still renders.
6. **Map arrays**: is the offline `--save_arrays` re-run acceptable later as a
   documented optional step, or is PNG-serving permanent?
   *Default: PNGs only in v1.*
7. **Deployment**: single user on the login/GPU node via SSH tunnel, no auth?
   *Default: localhost + tunnel, no auth.*
8. **Run deletion / rename**: tombstone in Problems, or drop silently?
   *Default: tombstone for one scanner lifetime.*
9. **WeatherBench2 backend**: no run on disk currently has
   `weatherbench2_rollout_*.csv`; support it in the contract now or defer?
   *Default: reserve a schema slot now, flag the path as untested.*
