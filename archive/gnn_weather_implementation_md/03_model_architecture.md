# 03 — Model Architecture

## First raw prototype

The first implementation should be simple and direct:

```text
Input weather grid
    ↓
Batch adapter: grid → nodes
    ↓
Linear embedding
    ↓
Encoder: local graph attention on L0
    ↓
Processor: Graph U-Net L0 → L1 → L2 → L1 → L0
    ↓
Decoder: local graph refinement on L0
    ↓
Output head: hidden → Δx
    ↓
Forecast: x_next = x_current + Δx
    ↓
Batch adapter: nodes → grid
```

No shared latent space for now.
No AIFS O96 grid for now.
No adaptive or learned long-range edges for now.
No chunked rollout training for the raw prototype.

---

# Input and output

## Input node feature

For each grid node `i`:

```text
node_feature_i = [
    x_{t-12h, i},
    x_{t, i},
    static_i,
    time_features
]
```

If there are `67` dynamic variables:

```text
C_in ≈ 2 × 67 + static/time features ≈ 145–150
C_out = 67
```

## Output

The model predicts a 12h tendency:

```text
Δx = x_{t+12h} - x_t
```

Final forecast:

```text
x_pred = x_t + Δx
```

For precipitation or accumulated variables, consider direct prediction later. For the first raw prototype, keep a simple uniform output head unless the baseline already treats precipitation separately.

---

# Recommended first 5° config

For the 5° prototype:

```text
resolution: 5°
N0 = 2,664
N1 = 684
N2 = 180
hidden dimension d = 96 or 128
neighbors k = 8
attention heads H = 4
physical batch size B = 2
```

Start with `d=96` if you want maximum speed. Use `d=128` if memory is comfortable.

---

# Encoder

## Structure

```text
Linear embedding: C_in → d
Local graph-attention block × 1 or 2 on L0
```

Recommended for raw prototype:

```text
encoder_blocks = 1
```

Increase to `2` only after the basic model is working.

## Why encoder is simple

The encoder should only convert raw weather variables into geometry-aware features. It should not pool/downsample yet.

Reason:

```text
Pooling too early may lose fine details before the model has learned useful features.
```

---

# Local graph-attention block

One block:

```text
Input h
    ↓
LayerNorm
    ↓
Local graph attention over kNN edges
    ↓
Residual connection
    ↓
LayerNorm
    ↓
MLP / feed-forward network
    ↓
Residual connection
    ↓
Output h
```

Formula:

```text
h1 = h + LocalGraphAttention(LayerNorm(h), edge_index, edge_attr)
h2 = h1 + MLP(LayerNorm(h1))
```

Attention is local, not global:

```text
each node attends only to its k nearest spherical neighbors
```

---

# Processor: Graph U-Net

## Purpose

The processor is the main weather reasoning module.

It solves a weakness of flat local GNNs:

```text
local message passing alone moves information slowly across the globe
```

The Graph U-Net gives:

```text
L0: local details
L1: regional context
L2: large-scale context
```

## Raw prototype processor structure

Use a small version first:

```text
L0 native graph blocks: 2
pool L0 → L1
L1 coarse graph blocks: 2
pool L1 → L2
L2 coarsest graph blocks: 1
unpool L2 → L1 + skip fusion
L1 refinement block: 1
unpool L1 → L0 + skip fusion
L0 refinement block: 1
```

This is smaller than the final research version and easier to train.

## Later larger processor

After the prototype works:

```text
L0 blocks: 4
L1 blocks: 3
L2 blocks: 2
L1 refinement: 1
L0 refinement: 1
```

---

# Pooling

Pooling maps fine features to coarse features:

```text
L0 → L1
L1 → L2
```

Use:

```text
mean pooling + max pooling
```

Then project:

```text
h_coarse = Linear([mean_pool(h_fine), max_pool(h_fine)])
```

Why both:

```text
mean = regional background state
max = strong/extreme local signal
```

---

# Unpooling

For the first prototype, use parent unpooling:

```text
h_unpooled[fine_node] = h_coarse[parent(fine_node)]
```

Then fuse with skip connection:

```text
h_fused = MLP([h_skip, h_unpooled])
```

Why skip:

```text
skip preserves fine local information
unpooled feature adds coarse/global context
```

Later, replace parent unpooling with learned coarse-to-fine attention if needed.

---

# Decoder

## Structure

```text
Local graph-refinement block × 1 on L0
Linear output head: d → C_out
```

The decoder does not need heavy global reasoning. The processor already handled multiscale communication.

Decoder purpose:

```text
restore local consistency
clean up unpooled features
produce Δx at every native grid node
```

---

# Parameter count rough ranges

For 5° prototype:

| Config | Hidden dim | Processor size | Approx params |
|---|---:|---|---:|
| tiny debug | 64 | 2-1-1 | 0.5M–1.0M |
| raw prototype | 96 | 2-2-1 | 1M–2M |
| stronger 5° | 128 | 2-2-1 | 2M–4M |
| later 2.5° | 128–160 | 4-3-2 | 4M–8M |
| later 1.5° | 160 | 4-3-2 | 5M–9M |

Node count affects activation memory much more than parameter count.

---

# Module layout suggestion

```text
models/gnn_weather/
    graph_weather_model.py      # top-level model wrapper
    batch_adapter.py            # converts baseline batch ↔ node tensors
    graph_builder.py            # precompute graph bundles
    graph_bundle.py             # load graph files
    layers.py                   # LocalGraphAttentionBlock, MLP, edge encoding
    pooling.py                  # mean+max pool, parent unpool, skip fusion
    processor.py                # GraphUNetProcessor
    losses.py                   # optional GNN losses
```
