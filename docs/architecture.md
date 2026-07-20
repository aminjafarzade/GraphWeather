# Architecture

A direct-grid spherical **Graph U-Net**. The model operates on the native
latitude/longitude grid (no regridding to a mesh), predicts a tendency `delta`,
and returns `x_current + delta`. Autoregressive rollout is trained with a
curriculum that grows the rollout length.

Library code lives in `src/` (import as the `src` package). Key modules:

| Module | Role |
|---|---|
| `architecture.py` | Graph U-Net assembly |
| `processor.py` / `layers.py` | message-passing processor + local graph-attention blocks |
| `pooling.py` | mean/max pool + parent unpool between levels |
| `graph_builder.py` / `graph_bundle.py` | hybrid row-aware kNN graph construction + container |
| `models.py` | model definitions |
| `features.py` / `solar.py` / `lead_conditioning.py` | input feature assembly (solar, orography, TISR, lead-time) |
| `trainer.py` | training loop, curriculum/rollout, BPTT, checkpointing |
| `evaluator.py` | weekly-52 eval, latitude-weighted RMSE/ACC, persistence, bootstrap CIs |
| `resolution.py` | `RESOLUTION_SPECS` — grid specs per resolution |
| `losses.py` | latitude-weighted MSE + spectral loss |
| `lr_schedulers.py` | warmup-cosine + rollout-stage schedulers |

## Graph levels

The graph is built directly on the grid with **directed hybrid row-aware
spherical kNN edges**. Level shapes per resolution (`src/resolution.py`):

| Resolution | Native grid | 3-level (default) shapes |
|---|---|---|
| `5p625` | 32×64 = 2048 | (32,64) → (16,32) → (8,16) |
| `2p5` | 72×144 = 10368 | (72,144) → (36,72) → (18,36) |
| `1p5` | 121×240 = 29040 | (121,240) → (61,120) → (31,60) |

> **1p5 includes the poles.** The kai_1p5 grid is 121 rows (latitudes −90…+90
> inclusive at 1.5°), unlike 2p5/5p625 which are pole-less. Do not pair a 1p5
> config with a 120×240 pole-less graph — see `data.md` and `CONTRIBUTING.md §3`.

## L3 / L4 depth variants

The original model is a 3-level U-Net. Deeper variants add coarser global
propagation levels and refine finer levels after unpooling:

- **L3 (4 graph levels)** — enable with `model.num_graph_levels: 4`, `use_l3: true`.
  For 2p5 this is `72×144 → 36×72 → 18×36 → 9×18`; it adds a coarse global level
  and then refines L2 after unpooling from L3.
- **L4** — an additional level (`l4_72_36_24_18_9` hierarchy for 2p5); see
  `src/resolution.py` for the exact level shapes and graph-path naming.

Checkpoints record `use_l3` and `num_graph_levels`; a 3-level checkpoint loads
into a 3-level model only, unless `--init_from_checkpoint_allow_partial` is used
explicitly for weight initialization.

## Capacity

Current experiments run **L3/L4** at **hidden dim 128 or 160** with **dense
graphs** (`densel3k24` / `densel4k24`, i.e. k=24 on the coarse level) and a
rollout **curriculum** (S1 base → S2…S10). The exact parameter counts and level
configuration for a given run are recorded in that run's `config_resolved.yaml`
and `model_summary.txt`. (The original 3-level, hidden-96, ~1.13M-parameter
5p625/2p5 baseline is the historical starting point, not the current model.)

## Prediction & rollout

- Predicts a tendency `delta`, returns `x_current + delta`.
- Unchunked autoregressive rollout training, daily steps (`dt: 1` = one forecast
  day; `S=10` = 10-day rollout), capped at `S=10` by default.
- Multi-horizon validation logs `valid_S1`, `valid_S2`, `valid_S4`, … `valid_S10`.
- Rollout-stage-aware LR scheduling lets LR decrease (or restart warmup+cosine)
  as rollout length grows. See `pipeline.md` for schedule details.
