# README for Codex: L4 Ratio-1.5 Graph U-Net Experiment

## 1. Goal

Implement a new clean architecture experiment for the GraphWeather / GNN weather forecasting repository:

```text
Base Graph U-Net
2.5° grid
hidden_dim = 128
fixed orography target handling
new 5-level hierarchy with approximately 1.5× coarsening
coarsest level = 14 × 28
```

The purpose is to test whether the current 2× Graph U-Net hierarchy loses too much mid-scale weather information during pooling.

Current hierarchy:

```text
L0: 72 × 144
L1: 36 × 72
L2: 18 × 36
L3: 9 × 18
```

New experiment hierarchy:

```text
L0: 72 × 144
L1: 48 × 96
L2: 32 × 64
L3: 21 × 42
L4: 14 × 28
```

This is not mainly a “more global compression” experiment. The old L3 `9×18 = 162` nodes is more compressed than the new L4 `14×28 = 392` nodes. This experiment tests whether gentler coarsening preserves mid-scale structures better.

Expected possible benefits:

```text
better mid-latitude storm-track structure
better z500 / msl / u850 / t850 propagation
less information loss between encoder and decoder
better fronts / cyclone / regional pressure-system structure
```

Expected possible risks:

```text
weaker global compression than old 9×18 L3
higher compute and memory
nonuniform pooling groups
possible pooling/unpooling artifacts
```

Keep this experiment clean. Do not combine with dense L3, skip gates, gated pooling, lead conditioning, spectral loss, static features, LSM, or hidden160 in this first run.

---

## 2. Hard constraints

Do not change unrelated behavior.

Use:

```yaml
resolution_mode: 2p5
hidden_dim: 128
num_heads: 4
input_channels: 134
output_channels: 67
use_l4_ratio15: true
num_graph_levels: 5
```

Do not use:

```text
hidden_dim=160
heavy U-Net
L3 k=24 dense graph from earlier experiment
scalar gated skip
scalar gated pooling
lead conditioning
spectral loss
full-rollout training from scratch
static auxiliary features
land-sea mask
TISR known-future override
```

Fixed orography must be active:

```yaml
target_handling:
  enabled: true
  copy_variables: ["orog"]
  exclude_loss_variables: ["orog"]
  known_future_variables: []
```

Model still outputs 67 channels, but loss uses 66/67 channels because `orog` is excluded.

---

## 3. New hierarchy and graph metadata

Add support for this hierarchy:

```yaml
model:
  hierarchy_type: ratio15_l4
  num_graph_levels: 5
  level_shapes:
    - [72, 144]
    - [48, 96]
    - [32, 64]
    - [21, 42]
    - [14, 28]
  level_k_neighbors: [8, 8, 8, 16, 24]
```

Expected node counts:

```text
L0: 72 × 144 = 10368
L1: 48 × 96  = 4608
L2: 32 × 64  = 2048
L3: 21 × 42  = 882
L4: 14 × 28  = 392
```

Expected directed edge counts with the above k values:

```text
L0: 10368 × 8  = 82944
L1: 4608 × 8   = 36864
L2: 2048 × 8   = 16384
L3: 882 × 16   = 14112
L4: 392 × 24   = 9408
```

Use a separate graph cache. Do not overwrite old graph files.

Suggested graph path:

```text
graphs/graph_2p5_ratio15_L4_k8_8_8_16_24_hybrid_row_aware_v1.pt
```

Graph metadata should include:

```python
{
    "hierarchy_type": "ratio15_l4",
    "num_graph_levels": 5,
    "use_l4_ratio15": True,
    "level_shapes": [[72,144], [48,96], [32,64], [21,42], [14,28]],
    "node_counts": [10368, 4608, 2048, 882, 392],
    "level_k_neighbors": [8, 8, 8, 16, 24],
    "edge_counts": [82944, 36864, 16384, 14112, 9408],
    "connectivity_strategy": "hybrid_row_aware_knn",
    "pooling_map_strategy": "proportional_parent_index",
    "graph_format_version": "ratio15_l4_v1"
}
```

When loading graph cache, validate:

```text
resolution mode
level shapes
node counts
level k values
edge counts
connectivity strategy
pooling map strategy
graph format version
coordinate/grid hash if available
```

If metadata does not match config, rebuild or raise a clear error.

---

## 4. Replace hardcoded 2× pooling with general parent-index pooling

The existing code may assume 2× pooling:

```python
parent_i = child_i // 2
parent_j = child_j // 2
```

This will not work for ratio-1.5 levels.

Implement general parent-index maps for arbitrary fine/coarse shapes.

For fine grid `Hf × Wf` and coarse grid `Hc × Wc`:

```python
parent_i = floor(i * Hc / Hf)
parent_j = floor(j * Wc / Wf)
parent_i = min(parent_i, Hc - 1)
parent_j = min(parent_j, Wc - 1)
parent_id = parent_i * Wc + parent_j
```

This creates:

```text
parent_index: LongTensor [N_fine]
```

Every fine node maps to exactly one coarse parent.

Required pool maps:

```text
pool_l0_to_l1: length 10368, parent IDs in [0, 4607]
pool_l1_to_l2: length 4608,  parent IDs in [0, 2047]
pool_l2_to_l3: length 2048,  parent IDs in [0, 881]
pool_l3_to_l4: length 882,   parent IDs in [0, 391]
```

Validate:

```text
every parent ID is valid
every fine node has exactly one parent
every coarse parent has at least one child
child counts are saved/logged
```

The child groups will be variable-sized. Some parents may have one child, others two or four. This is expected.

---

## 5. Pooling operation

Keep the first L4 experiment simple and close to current behavior.

Use the existing pooling idea:

```text
mean_pool + max_pool → concat → linear projection
```

But make it work with arbitrary parent maps and variable-size groups.

Pseudo-code:

```python
def pool_parent_index(h_fine, parent_index, num_coarse):
    # h_fine: [B, N_fine, D]
    # parent_index: [N_fine]
    # num_coarse: int

    mean = scatter_mean(h_fine, index=parent_index, dim=1, dim_size=num_coarse)
    maxv = scatter_max(h_fine, index=parent_index, dim=1, dim_size=num_coarse)
    pooled = pool_proj(torch.cat([mean, maxv], dim=-1))
    return pooled
```

Prefer area-weighted mean if easy:

```text
child_weight = cos(latitude_child)
weighted_mean = scatter_sum(h_fine * child_weight) / scatter_sum(child_weight)
```

If area-weighted mean is implemented, log it:

```yaml
pooling:
  type: parent_index_meanmax
  mean_type: area_weighted
  include_max: true
```

If area-weighted mean is too invasive, use normal mean for this first experiment, but keep the code structured so area-weighted mean can be added later.

Do not implement dynamic top-k gPool. Dense weather prediction needs fixed spatial alignment. Use deterministic static pooling.

---

## 6. Unpooling operation

Unpooling should be the deterministic reverse of parent-index pooling.

Pseudo-code:

```python
def unpool_parent_index(h_coarse, parent_index):
    # h_coarse: [B, N_coarse, D]
    # parent_index: [N_fine]
    return h_coarse[:, parent_index, :]
```

Then fuse with the encoder skip at that level:

```python
h_up = unpool_parent_index(h_coarse, parent_index)
h_fine = fuse(skip_fine, h_up)
```

Do not zero-fill unselected nodes. This is not top-k pooling. Every fine node receives the feature of its assigned coarse parent.

---

## 7. Build graphs at every level

For every level, build a real graph using the existing spherical/hybrid row-aware kNN construction.

Do not derive graph edges only from pooling maps.

For each level:

```text
create lat-lon coordinates
convert to sphere coordinates if current graph builder does this
build hybrid row-aware kNN graph
compute edge features as before
validate connectedness
validate no self-loops
validate no duplicate neighbors per target
save metadata
```

Use:

```text
L0 k = 8
L1 k = 8
L2 k = 8
L3 k = 16
L4 k = 24
```

Reason: the new coarsest level `14×28` is less compressed than old `9×18`, so top levels need denser connectivity to avoid weak long-range communication.

---

## 8. Processor architecture

Add a new 5-level processor path, while keeping existing 3-level and 4-level models backward compatible.

Suggested forward flow:

```text
L0 processing
↓ pool L0 → L1
L1 processing
↓ pool L1 → L2
L2 processing
↓ pool L2 → L3
L3 processing
↓ pool L3 → L4
L4 processing
↑ unpool L4 → L3 + L3 skip
L3 refinement after L4
↑ unpool L3 → L2 + L2 skip
L2 refinement after L3
↑ unpool L2 → L1 + L1 skip
L1 refinement
↑ unpool L1 → L0 + L0 skip
L0 refinement
```

First experiment block counts:

```yaml
model:
  l0_blocks: 2
  l1_blocks: 2
  l2_blocks: 1
  l3_blocks: 1
  l4_blocks: 1

  l3_refine_after_l4_blocks: 1
  l2_refine_after_l3_blocks: 1
  l1_refine_blocks: 1
  l0_refine_blocks: 1
```

Use the same existing graph attention block class. Do not change attention logic.

Do not add extra activations between blocks.

---

## 9. New config file

Create:

```text
configs/weather_dual_resolution_l4_ratio15_hidden128_fixed_orog.yaml
```

Experiment name:

```yaml
experiment_name: main_raw_2p5_b4_acc3_bf16_delta_l4_ratio15_hidden128_fixed_orog
```

Config content should include:

```yaml
resolution_mode: 2p5

model:
  input_channels: 134
  output_channels: 67

  hidden_dim: 128
  num_heads: 4

  hierarchy_type: ratio15_l4
  use_l4_ratio15: true
  num_graph_levels: 5

  level_shapes:
    - [72, 144]
    - [48, 96]
    - [32, 64]
    - [21, 42]
    - [14, 28]

  k_neighbors: 8
  level_k_neighbors: [8, 8, 8, 16, 24]

  l0_blocks: 2
  l1_blocks: 2
  l2_blocks: 1
  l3_blocks: 1
  l4_blocks: 1

  l3_refine_after_l4_blocks: 1
  l2_refine_after_l3_blocks: 1
  l1_refine_blocks: 1
  l0_refine_blocks: 1

  pooling:
    type: parent_index_meanmax
    mean_type: area_weighted
    include_max: true

resolution_profiles:
  2p5:
    graph_path: graphs/graph_2p5_ratio15_L4_k8_8_8_16_24_hybrid_row_aware_v1.pt

extra_features:
  enabled: false

target_handling:
  enabled: true
  copy_variables: ["orog"]
  exclude_loss_variables: ["orog"]
  known_future_variables: []

batch_size: 4
gradient_accumulation_steps: 3
max_epochs: 50

enable_amp: true
amp_dtype: auto_bf16_fp16

use_delta_normalization: true

lr_schedule_type: warmup_cosine
lr: 1.0e-4
min_lr: 3.0e-6
warmup_epochs: 2
warmup_start_factor: 0.1

single_pass_multi_horizon_validation: true
load_only_current_rollout: true

max_rollout_steps: 10
rollout_schedule: [1, 2, 4, 6, 8, 10]
rollout_stage_epochs: [3, 3, 8, 10, 12, 14]

checkpoint_metric: valid_S10
checkpoint_mode: min
stage_checkpoint_metric_mode: final_horizon

log_every_batches: 100
```

If `area_weighted` pooling is not implemented in the first pass, change `mean_type` to `mean`, but document it in the logs.

---

## 10. Expected size and memory

Expected parameter count:

```text
~2.9M–3.1M parameters
```

Expected memory:

```text
~28–35 GB at late rollout stages with batch_size=4, bf16, gradient_accumulation_steps=3
```

If OOM occurs, fallback:

```yaml
batch_size: 2
gradient_accumulation_steps: 6
```

Expected runtime:

```text
~1.3× to 1.6× slower than base h128 L3
```

---

## 11. Logging

At startup, log:

```text
Graph hierarchy:
  hierarchy_type: ratio15_l4
  num_graph_levels: 5
  level_shapes: [[72,144], [48,96], [32,64], [21,42], [14,28]]
  node_counts: [10368, 4608, 2048, 882, 392]
  level_k_neighbors: [8, 8, 8, 16, 24]
  edge_counts: [82944, 36864, 16384, 14112, 9408]
  graph_path: graphs/graph_2p5_ratio15_L4_k8_8_8_16_24_hybrid_row_aware_v1.pt

Pooling:
  type: parent_index_meanmax
  mean_type: area_weighted or mean
  include_max: true
  pool maps:
    L0->L1 child_count min/mean/max: ...
    L1->L2 child_count min/mean/max: ...
    L2->L3 child_count min/mean/max: ...
    L3->L4 child_count min/mean/max: ...

Model:
  hidden_dim: 128
  num_heads: 4
  head_dim: 32
  trainable_parameters: ...

Target handling:
  copy_variables: orog -> channel ...
  exclude_loss_variables: orog -> channel ...
  loss channels: 66/67
```

Save this information to:

```text
config_resolved.yaml
model_summary.txt
checkpoint metadata
```

Checkpoint metadata should include:

```python
{
    "hierarchy_type": "ratio15_l4",
    "num_graph_levels": 5,
    "level_shapes": [[72,144], [48,96], [32,64], [21,42], [14,28]],
    "level_k_neighbors": [8, 8, 8, 16, 24],
    "pooling": {
        "type": "parent_index_meanmax",
        "mean_type": "area_weighted",
        "include_max": True
    },
    "target_handling": {
        "enabled": True,
        "copy_variables": ["orog"],
        "exclude_loss_variables": ["orog"],
        "known_future_variables": []
    }
}
```

---

## 12. Backward compatibility

Existing model configs must continue to work:

```text
3-level old models
4-level L3 models
4-level dense L3 models
hidden128/hidden160 configs
skip gate / pooling / lead-conditioned configs
```

Do not make ratio-1.5 hierarchy the default. It must only activate when:

```yaml
model:
  hierarchy_type: ratio15_l4
  use_l4_ratio15: true
  num_graph_levels: 5
```

Checkpoint mismatch should raise a clear error:

```text
Checkpoint architecture mismatch:
checkpoint hierarchy_type=standard_l3,
current hierarchy_type=ratio15_l4.
Train from scratch or use explicit partial initialization.
```

For this experiment, train from scratch.

---

## 13. Tests

Add/update tests.

### Graph construction tests

Validate:

```text
level shapes
node counts
edge counts
level k values
one connected component per level
no self-loops
no duplicate neighbors per target
```

Expected:

```text
L0 nodes=10368 edges=82944 k=8
L1 nodes=4608  edges=36864 k=8
L2 nodes=2048  edges=16384 k=8
L3 nodes=882   edges=14112 k=16
L4 nodes=392   edges=9408  k=24
```

### Parent map tests

Validate:

```text
L0->L1 length=10368 parent range 0..4607
L1->L2 length=4608 parent range 0..2047
L2->L3 length=2048 parent range 0..881
L3->L4 length=882 parent range 0..391
```

Every coarse parent must have at least one child.

### Pool/unpool tests

Use toy tensors to verify:

```text
pool output shape = [B, N_coarse, D]
unpool output shape = [B, N_fine, D]
unpool gathers the correct parent feature for each child
```

### Forward shape test

```python
x = torch.randn(1, 134, 72, 144)
y = model(x)
assert y.shape == (1, 67, 72, 144)
```

### Target handling test

Verify:

```text
orog is copied during rollout
loss uses 66/67 channels
```

### Smoke training

Run tiny training:

```text
max_train_batches=2
max_valid_batches=2
max_epochs=2
```

Confirm:

```text
no NaN
no shape mismatch
checkpoint saves
metadata records hierarchy_type=ratio15_l4
```

---

## 14. Commands

### Build graph

```bash
python scripts/build_graph.py \
  --config configs/weather_dual_resolution_l4_ratio15_hidden128_fixed_orog.yaml \
  --resolution_mode 2p5 \
  --force_rebuild
```

### Train

```bash
python scripts/train.py \
  --config configs/weather_dual_resolution_l4_ratio15_hidden128_fixed_orog.yaml \
  --resolution_mode 2p5 \
  --experiment_name main_raw_2p5_b4_acc3_bf16_delta_l4_ratio15_hidden128_fixed_orog
```

### Evaluate

Prefer WeatherBench2-compatible evaluation if available:

```bash
EXP=experiments/main_raw_2p5_b4_acc3_bf16_delta_l4_ratio15_hidden128_fixed_orog
CFG=configs/weather_dual_resolution_l4_ratio15_hidden128_fixed_orog.yaml
CLIM=data/stats/2p5_train_dayofyear_climatology.nc
KAI=data/external/kai_2.5.csv

python scripts/evaluate.py \
  --config "$CFG" \
  --resolution_mode 2p5 \
  --checkpoint "$EXP/best_ckpt.tar" \
  --split test \
  --fixed_rollout_steps 10 \
  --selection stride \
  --stride 7 \
  --variables z500 t2m t850 msl q700 u850 \
  --include_persistence \
  --climatology_path "$CLIM" \
  --bootstrap_samples 1000 \
  --confidence_level 0.95 \
  --bootstrap_seed 42 \
  --rmse_backend weatherbench2 \
  --external_baseline_csv "$KAI" \
  --external_baseline_label "Kai 7M" \
  --output_dir "$EXP/evaluation_test_weekly_weatherbench2"
```

If WeatherBench2 backend is not available yet, use the current evaluation command.

---

## 15. Comparison after training

Compare against:

```text
h128 fixed-orog control
h128 dense L3 k=24
h128 scalar gated skip
h128 scalar gated pooling
h128 lead conditioned
h160 fixed-orog
Kai 7M
persistence
```

Main questions:

```text
1. Does L4 ratio-1.5 improve valid_S10?
2. Does it improve WeatherBench2 day-10 RMSE/ACC?
3. Are improvements strongest in z500 / msl / u850 / t850?
4. Does it reduce mid-latitude error dominance?
5. Does it improve S=4/S=6 but fail at S=10?
6. Is the improvement worth extra compute and memory?
```

Success criteria:

```text
valid_S10 improvement over h128 control > 0.003
and/or
mean day-10 WeatherBench2 RMSE improvement > 1%
```

If it only improves all-channel validation loss but not selected WeatherBench2 metrics, do not treat it as successful.

---

## 16. Acceptance criteria

This task is complete when:

```text
1. New ratio15_l4 hierarchy is implemented.
2. Existing old hierarchies remain backward compatible.
3. General parent-index pooling/unpooling works for arbitrary level shapes.
4. New graph cache exists and has correct metadata.
5. L0/L1/L2/L3/L4 graphs are connected and have expected edge counts.
6. Model forward pass works with input [B,134,72,144] and output [B,67,72,144].
7. Fixed-orography target handling is active and loss uses 66/67 channels.
8. New config exists.
9. Smoke training passes.
10. Full training command starts correctly.
```
