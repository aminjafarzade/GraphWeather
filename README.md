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

[CONTRIBUTING.md](CONTRIBUTING.md) covers the repo conventions: naming, layout,
config layering, and the run/dashboard invariants.

The reference guides below live in the working tree but are **not published in
this repository**, so the paths are given as plain text rather than links:

| Guide | Covers |
|---|---|
| `docs/architecture.md` | Graph U-Net, graph levels, L3/L4 variants, delta prediction, the mesh encoder |
| `docs/data.md` | Resolutions, NetCDF layout, stats, KAI baselines, the 1p5 pole-less trap |
| `docs/pipeline.md` | Environment, the `run_pipeline.sh` golden path, training & curriculum |
| `docs/evaluation.md` | RMSE/ACC eval, WeatherBench2 backend, stage comparison, zone dominance |
| `docs/dashboard.md` | The results dashboard + its `gw-run/1` data contract |

The dashboard's own data contract *is* published, under
[dashboard/docs/](dashboard/docs/).

## Quickstart

```bash
# Python 3.10+. Install the package (editable) + dev tools.
pip install -e '.[dev]'
```

For GPU training, install the torch build matching your CUDA driver first. On the
sm_120 box use the pinned `graphweather-cu128` interpreter — do not assume bare
`python` (it silently misbehaves under the wrong env). See
`docs/pipeline.md`.

Run a full experiment (graph → train → clim → eval → plot → maps → diagnostics)
via the single launcher:

```bash
bash scripts/run_pipeline.sh \
  --resolution 2p5 \
  --config configs/experiments/<id>.yaml \
  --config-name <id> \
  --stages train,eval,plot
```

`--stages` selects any subset of `graph,train,clim,eval,plot,qual,diag,dashmaps`;
it resets every stage toggle first, so a stale `RUN_*` in the environment cannot
leak in and retrain over a finished run. `--dry-run` prints the commands without
executing them. The result lands in `runs/<id>/` (see
`docs/pipeline.md`).

`scripts/run_full_pipeline.sh` still exists as a thin shim over this launcher;
new work should call `run_pipeline.sh` directly.

## Results

Best runs at each resolution, read from the eval outputs under
`runs/<id>/evaluation_test_weekly52/`. RMSE is latitude-weighted; variables are in
model units (see `docs/evaluation.md`). Lower RMSE and higher
ACC are better.

### 2.5° — best run

`2p5_l3_h160_densel3k24_s1x100lr1e3_currS2toS10x3_flatlr1e6_perfctl_compilefull_ema_itv`

L3 / hidden 160 / dense-L3K24, **3.84 M parameters**, best checkpoint at epoch 127.
S1 for 100 epochs at lr 1e-3, then an S2…S10 curriculum at 3 epochs per horizon with
a flat lr of 1e-6, EMA, `compile_scope: full`, and inverse-tendency-variance loss
weighting. Evaluated on the **test** split over 52 initial conditions at stride 7,
10-day rollout, WeatherBench2-compatible RMSE.

| Variable | RMSE d3 | RMSE d5 | RMSE d10 | ACC d5 | ACC d10 | KAI 2.5° RMSE d3 / d5 / d10 |
|---|---:|---:|---:|---:|---:|---:|
| z500 | 203.5 | 374.2 | 672.5 | 0.876 | 0.521 | 175 / 335 / 640 |
| t850 | 1.171 | 1.750 | 2.889 | 0.833 | 0.469 | 1.05 / 1.70 / 3.10 |
| t2m  | 1.024 | 1.413 | 2.216 | 0.805 | 0.459 | 1.30 / 1.75 / 2.95 |
| msl  | 207.0 | 358.6 | 609.3 | 0.842 | 0.458 | 200 / 345 / 600 |
| u850 | 2.052 | 3.099 | 4.621 | 0.789 | 0.428 | — |

Against the KAI 2.5° reference the picture is **mixed, not a clean win**: better on
`t2m` at every lead and on `t850` by day 10, behind on `z500` at every lead and on
`msl` at short leads.

> Only RMSE is compared above. The ACC column in `data/baselines/kai_2p5.csv` uses a
> different definition from this repo's evaluator (0.75 at day 3 versus 0.96 here),
> so the two ACC numbers are not comparable and are deliberately not put side by side.

### 1.5° — best run

`1p5_l4_h160_densel4k24_currS2toS10x3_initckpt_s1x150`

L4 / hidden 160 / dense-L4K24, best checkpoint at epoch 19, S1 for 150 epochs then an
S2…S10 curriculum warm-started from that S1 checkpoint. Evaluated over 51 initial
conditions spanning 2020-01-02 … 2020-12-17.

| Variable | RMSE avg d1–10 | RMSE d10 | ACC avg d1–10 | ACC d10 | KAI 1.5° RMSE d10 | Persistence d10 |
|---|---:|---:|---:|---:|---:|---:|
| z500 | 421.0 | 697.6 | 0.803 | 0.530 | 799.4 | 1088.9 |
| t850 | 2.131 | 3.057 | 0.723 | 0.437 | 3.437 | 4.524 |
| t2m  | 1.768 | 2.463 | 0.693 | 0.429 | 2.615 | 3.259 |
| msl  | 394.1 | 630.6 | 0.774 | 0.481 | 730.1 | 952.3 |

At 1.5° the model **beats the KAI reference at day 10 on all four headline
variables**, and beats persistence by a wide margin. The L3 / hidden-128 variant
(`1p5_l3_h128_densel3k24_currS2toS10x3_initckpt`) is behind it on z500
(442.8 average, 692.1 at day 10), so the extra graph level and width are earning
their cost here.

> The 1.5° table comes from this repo's fixed-10 summary, not the WeatherBench2
> backend used for the 2.5° table, so the two tables are **not** directly comparable
> to each other. Each is internally consistent and each is compared only against a
> baseline evaluated the same way.

### Work in progress

An optional GraphCast-style **mesh encoder** (grid → icosphere → grid) is implemented
but not yet reflected in the numbers above; see
`docs/architecture.md` and `mesh_encoder` in the config schema.
Grid and mesh checkpoints are separate families and the loader refuses to mix them.

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
`docs/dashboard.md`. A directory is a "run" iff it contains
`config_resolved.yaml`; that invariant must be preserved by any run-storage change.

## Repository layout

```text
src/          library code (import as the `src` package)
scripts/      CLIs: train.py, evaluate.py, plot_*, visualize_*, run_pipeline.sh
scripts/dev/  debug/diagnostic/profiling helpers (not part of the pipeline)
configs/      YAML configs (base + experiments)
tests/        regression net (fast no-GPU lane + torch-heavy lane; see CONTRIBUTING §9)
dashboard/    results dashboard (FastAPI + SPA) + dashboard/docs (data contract)
docs/         reference guides + docs/experiments (specs) -- local only, not tracked
archive/      retired configs/scripts + generated decks -- local only, not tracked
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
