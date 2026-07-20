# GraphWeather

A direct-grid spherical **Graph U-Net** for medium-range weather forecasting on
latitude/longitude grids. The model works on the native grid (no mesh regridding),
predicts a tendency `delta` and returns `x_current + delta`, and is trained with an
autoregressive rollout curriculum.

Current work runs at **2.5° (`2p5`)** and **1.5° (`1p5`)** resolution with the
**L3/L4** depth variants (4–5 graph levels), **hidden dim 128 or 160**, **dense
row-aware graphs**, and a rollout **curriculum** (S1 base → S2…S10, often
warm-started from an S1 checkpoint). The original 3-level, hidden-96, ~1.13M-param
5.625° model is the historical baseline, not the current model.

> **Repo name.** The directory is still `GraphWeather5p625` for historical reasons;
> the active resolutions are 2p5/1p5. A rename is deferred (high blast radius).

## Documentation

| Guide | Covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Graph U-Net, graph levels, L3/L4 variants, delta prediction |
| [docs/data.md](docs/data.md) | Resolutions, NetCDF layout, stats, KAI baselines, the 1p5 pole-less trap |
| [docs/pipeline.md](docs/pipeline.md) | Environment, the `run_full_pipeline.sh` golden path, training & curriculum |
| [docs/evaluation.md](docs/evaluation.md) | RMSE/ACC eval, WeatherBench2 backend, stage comparison, zone dominance |
| [docs/dashboard.md](docs/dashboard.md) | The results dashboard + its `gw-run/1` data contract |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Repo conventions: naming, layout, config layering, run/dashboard invariants |

## Quickstart

```bash
# Python 3.10+. Install the package (editable) + dev tools.
pip install -e '.[dev]'
```

For GPU training, install the torch build matching your CUDA driver first. On the
sm_120 box use the pinned `graphweather-cu128` interpreter — do not assume bare
`python` (it silently misbehaves under the wrong env). See
[docs/pipeline.md](docs/pipeline.md).

Run a full experiment (train → eval → plot → maps → diagnostics) via the launcher:

```bash
CONFIG=configs/experiments/<id>.yaml \
CONFIG_NAME=<id> \
bash scripts/run_full_pipeline.sh
```

Every stage is individually toggleable (`RUN_TRAIN`, `RUN_EVAL`, `RUN_PLOT`,
`RUN_QUALITATIVE`, `RUN_DIAGNOSTICS`); `DRY_RUN=1` prints commands without running
them. The result lands in `runs/<id>/` (see [docs/pipeline.md](docs/pipeline.md)).

## Results dashboard

`dashboard/` is a standalone FastAPI + SPA app that scans `runs/` and presents each
run's metrics, diagnostics, and qualitative maps. It reads runs verbatim (never
imports `src/`, never writes into `runs/`).

```bash
pip install -r dashboard/requirements.txt   # separate from the main deps
python -m dashboard.ingest --once --json     # inspect ingested run records
```

Its data contract (`gw-run/1`) is documented in
[dashboard/docs/](dashboard/docs/) and summarized in
[docs/dashboard.md](docs/dashboard.md). A directory is a "run" iff it contains
`config_resolved.yaml`; that invariant must be preserved by any run-storage change.

## Repository layout

```text
src/          library code (import as the `src` package)
scripts/      CLIs: train.py, evaluate.py, plot_*, visualize_*, run_full_pipeline.sh
configs/      YAML configs (base + experiments)
tests/        regression net (fast no-GPU lane + torch-heavy lane; see CONTRIBUTING §9)
dashboard/    results dashboard (FastAPI + SPA) + dashboard/docs (data contract)
docs/         reference guides (this table) + docs/experiments (specs)
runs/         experiment outputs (git-ignored except .gitkeep)
data/         stats/ (regenerable, ignored) + baselines/ (tracked KAI CSVs)
graphs/       auto-built graph bundles (ignored)
```

## Tests

```bash
# Fast lane (no GPU): dashboard contract + config/resolution invariants
pytest tests/test_dashboard_*.py tests/test_config_*.py \
       tests/test_resolution*.py tests/test_zone_rmse_dominance.py

# Full lane (needs a GPU box for the torch-heavy tests)
pytest tests/
```

CI runs the fast lane on every push (`.github/workflows/ci.yml`).
