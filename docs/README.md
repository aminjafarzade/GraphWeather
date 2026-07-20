# GraphWeather — Documentation

Reference guides for the direct-grid spherical Graph U-Net. Start with the
top-level [`README.md`](../README.md) for a quickstart, then dive in here.

| Guide | Covers |
|---|---|
| [architecture.md](architecture.md) | The Graph U-Net (levels, graphs, delta prediction, L3/L4 variants, hidden dims) |
| [data.md](data.md) | Expected NetCDF layout, resolutions (2p5 / 1p5 / 5p625), stats, KAI baselines |
| [pipeline.md](pipeline.md) | Environment, the `run_full_pipeline.sh` golden path, training, curriculum |
| [evaluation.md](evaluation.md) | RMSE/ACC evaluation, WeatherBench2 backend, stage comparison, zone dominance |
| [dashboard.md](dashboard.md) | The results dashboard and its `gw-run/1` data contract |
| [experiments/](experiments/) | One-off experiment specs |

Conventions that govern the repo layout, naming, configs, and the dashboard
contract live in [`../CONTRIBUTING.md`](../CONTRIBUTING.md). The dashboard's own
detailed docs are under [`../dashboard/docs/`](../dashboard/docs/).

> **Note.** These guides are seeded from the (previously monolithic) README and
> reflect the current 2p5/1p5, L3/L4, hidden-128/160 work. Some structural items
> described in `CONTRIBUTING.md` (config layering, tracked baselines, run-naming
> grammar) are the target state and are being rolled out in phases.
