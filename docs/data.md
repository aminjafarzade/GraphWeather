# Data

## Resolutions

Three resolutions are supported (`src/resolution.py`, canonical `Np5` tags):

| Mode | Grid (lat×lon) | Poles? | Notes |
|---|---|---|---|
| `5p625` | 32×64 | no | legacy 5.625° baseline |
| `2p5` | 72×144 | no | primary current resolution |
| `1p5` | 121×240 | **yes** | kai_1p5; latitudes −90…+90 inclusive at 1.5° |

Accepted aliases resolve to the canonical form: `5.625`/`5deg625` → `5p625`,
`2.5`/`2deg5` → `2p5`, `1.5`/`1deg5` → `1p5`. Use the `Np5` form in all
identifiers; the dotted form is only for human-facing plot labels
(`CONTRIBUTING.md §1.1`).

## NetCDF layout

Each NetCDF file contains a field array shaped:

```
fields[time, channel, latitude, longitude]
```

The KAI/ERA5 files are daily. The loader validates grid shape and channel-count
compatibility against the active resolution profile — **do not reuse 5.625°
statistics for a 2.5° run**, etc.

Data/graph/stats paths are set per resolution profile in the config. Graph and
checkpoint paths are repo-root-relative, so the checkout can be moved freely. Data
roots are still absolute: the 2.5°/5.625° sets resolve to `/home/amin/KAI_5/…` on
this machine, and the 1.5° KAI set to `/lustre/home/mahmed/Hydro/kai_1p5_data`
(another user's home — the one remaining external dependency). The target state
moves both behind `${GW_DATA_ROOT}`-style environment interpolation + a
`configs/base/paths.yaml` layer (`CONTRIBUTING.md §3.2`); `.env.example` documents
the variables.

## Normalization stats & climatology

- Normalization statistics (`global_mean.npy` / `global_std.npy`) and day-of-year
  climatology live under `data/stats/` and are **regenerable** (git-ignored).
- Climatology is rebuilt via `scripts/build_climatology.py` or computed on the fly
  by the evaluator (`--compute_climatology`), cached alongside the eval output.

## Graphs

Row-aware graph bundles are cached under `graphs/` (git-ignored, auto-built at
runtime when `auto_build_graph: true`, or built explicitly with
`scripts/build_graph.py`). Rebuild only when the grid, coordinates, or graph
settings change.

> **1p5 pole-less trap.** Some legacy 1p5 configs reference a 120×240 pole-less
> graph while the kai_1p5 grid is 121×240. Never auto-rebuild a 1p5 run onto a
> pole-less graph; use the `graph_1p5_121x240_*` bundles. See `architecture.md`.

## KAI baselines (external comparison)

The KAI reference curves (`kai_2p5.csv`, `kai_1p5.csv`) are **tracked baseline
inputs** and belong under `data/baselines/` (`CONTRIBUTING.md §2`, §1.1 for the
`2.5→2p5` rename). They are consumed by the plotter, the pipeline, and the
dashboard's external-baseline view.
