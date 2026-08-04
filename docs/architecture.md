# Architecture

A spherical **Graph U-Net**. By default the model operates directly on the
native latitude/longitude grid, predicts a tendency `delta`, and returns
`x_current + delta`. An optional `mesh_encoder.enabled: true` path encodes the
grid onto an icosphere, runs the same Graph U-Net processor on that mesh, and
decodes back to the grid before the unchanged tendency head and residual.
Autoregressive rollout is trained with a curriculum that grows the rollout
length.

Library code lives in `src/` (import as the `src` package). Key modules:

| Module | Role |
|---|---|
| `architecture.py` | Graph U-Net assembly |
| `processor.py` / `layers.py` | message-passing processor + local graph-attention blocks |
| `pooling.py` | mean/max pool + parent unpool between levels |
| `graph_builder.py` / `graph_bundle.py` | hybrid row-aware kNN graph construction + container |
| `mesh_builder.py` / `mesh_layers.py` | optional native icosphere bundle + bipartite grid/mesh message passing |
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

## Optional icosphere encode/decode

The mesh path is opt-in. Configurations without this block retain the legacy
direct-grid path:

```yaml
mesh_encoder:
  enabled: true
  refinement: 5
  g2m_radius_factor: 0.6
  mlp_hidden_ratio: 2
graph_path: graphs/graph_2p5_icosphere_r5_l3.pt
```

`L0` is the finest native icosphere triangulation and successively coarser
levels come from the same midpoint-subdivision hierarchy. By default, native
degree-5/6 connectivity is retained. Each native level is stored as six incoming
slots per node; the 12 original icosahedron vertices have one masked dummy self
slot. The mask makes that slot contribute exactly zero while keeping the
existing dense fixed-k attention implementation.

For the two 2.5° L3 experiments:

| Finest mesh | U-Net mesh levels |
|---|---|
| M5 | 10242 → 2562 → 642 → 162 |
| M4 | 2562 → 642 → 162 → 42 |

Grid→mesh edges use the configured radius factor times the maximum native edge
length of the finest mesh. Mesh→grid edges are the three vertices of the native
finest-mesh triangle containing each grid point. The existing processor and
pool/unpool modules contain no mesh-specific branches.

The embedded grid already bypasses the complete mesh processor: Mesh2Grid uses
it as the destination latent and applies a residual update
`h_grid + update(h_grid, aggregated_mesh_messages)`. Therefore a second
post-decoder addition of the same embedding would duplicate the identity path.
The opt-in `grid_skip_mlp: true` instead follows the GraphCast encoder more
closely by carrying
`h_grid_skip = h_grid + MLP+LayerNorm(h_grid)` to Mesh2Grid. Grid2Mesh messages
still use `h_grid`, so the skip update and mesh update are parallel outputs of
the encoder step. This option requires `boundary_type: graphcast_mlp`:

```yaml
mesh_encoder:
  boundary_type: graphcast_mlp
  grid_skip_mlp: true
```

For a dense-L0-style spatial boundary, mesh mode also supports an opt-in grid
attention encoder and decoder:

```yaml
mesh_encoder:
  boundary_type: graphcast_mlp
  grid_attention_encoder_blocks: 2
  grid_attention_decoder_blocks: 2
  grid_attention_k_neighbors: 8
```

The grid encoder runs after grid embedding and before Grid2Mesh. Its output is
both the Grid2Mesh source and the long grid latent carried to Mesh2Grid. The
grid decoder runs after the residual Mesh2Grid update and before the output
head. These are true spatial `LocalGraphAttentionBlock`s; unlike
`grid_skip_mlp`, they exchange information over grid-grid edges.

The dense-H160 comparison config embeds the exact L0 tensors from
`graph_2p5_k8_l3k24_hybrid_row_aware_L3_v4.pt` in the mesh bundle. This avoids
changing tie-broken polar neighbors when independently rebuilding kNN from
near-identical floating-point coordinates. Its effective hierarchy is
grid→M4→M3→M2 (`10368→2562→642→162`), so M1 is omitted to match the dense
model's four scales:

```text
grid embedding -> grid attention x2 -> Grid2Mesh
-> M4/M3/M2 U-Net -> Mesh2Grid -> grid attention x2 -> head
```

`mesh_encoder.coarse_level_connectivity` defaults to `native_icosphere`: no
extra long-range edges are added, and global mixing relies on U-Net depth. The
opt-in `full_m1` value is accepted only when the coarsest level is M1. It makes
that 42-node level a complete directed graph without self-edges (`k=41`,
1,722 edge slots), while every finer level remains native. Graph bundles record
the choice and use separate cache/checkpoint families for the two topologies.

Grid and mesh checkpoints are separate families. Mesh checkpoints record
`graph_mode: mesh` and the resolved `mesh_encoder` settings. Strict resume,
initialization, evaluation, and visualization reject a checkpoint from the
other family. Existing grid checkpoints continue to load in grid mode.

The combined 150-epoch S1 plus three-epochs-per-later-horizon configs are:

- `configs/experiments/2p5_l3_h160_icomeshm5_s1x150_currs2tos10x3.yaml`
- `configs/experiments/2p5_l3_h160_icomeshm4_s1x150_currs2tos10x3.yaml`

Build and train either one with:

```bash
python scripts/build_graph.py \
  --config configs/experiments/2p5_l3_h160_icomeshm5_s1x150_currs2tos10x3.yaml \
  --config_name 2p5_l3_h160_icomeshm5_s1x150_currs2tos10x3

python scripts/train.py \
  --config configs/experiments/2p5_l3_h160_icomeshm5_s1x150_currs2tos10x3.yaml \
  --config_name 2p5_l3_h160_icomeshm5_s1x150_currs2tos10x3
```

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
