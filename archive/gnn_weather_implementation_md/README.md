# Lightweight Graph U-Net Weather Backbone — Implementation Guide

This folder contains implementation instructions for the first prototype of the graph-based weather forecasting backbone we designed.

The goal is to implement a **direct-grid spherical graph model** that:

- starts at **5° resolution** for fast prototyping,
- later scales to **2.5°** and **1.5°**,
- keeps the **same dataset preparation/interface** as the existing non-GNN baseline project,
- predicts a **12-hour tendency** `Δx`,
- supports autoregressive rollout training with rollout length gradually increased from `S=1` to `S=20`,
- does **not** use chunked rollout/backpropagation in the first raw prototype,
- uses a small physical batch size, initially `B=2`.

The model is not using AIFS's O96 latent processor grid. Instead, it uses the native grid as the input/output graph and builds a small graph pyramid:

```text
L0: native graph
L1: coarse graph
L2: coarsest graph
```

For the 5° prototype, assuming inclusive latitudes and regular longitude spacing:

```text
L0: 37 × 72  = 2,664 nodes
L1: 19 × 36  =   684 nodes
L2: 10 × 18  =   180 nodes
```

The core model:

```text
Input batch from existing dataset
        ↓
GNN batch adapter: grid tensor → node tensor
        ↓
Linear embedding
        ↓
Encoder: 1–2 local graph-attention blocks on L0
        ↓
Processor: Graph U-Net pyramid L0 → L1 → L2 → L1 → L0
        ↓
Decoder: 1 local graph-refinement block on L0
        ↓
Output head: hidden → Δx
        ↓
Forecast: x_next = x_current + Δx
        ↓
Output adapter: node tensor → original grid tensor format
```

## Markdown files

Read in this order:

1. [`01_interface_contract.md`](01_interface_contract.md) — how to keep the same dataset/model interface as the existing non-GNN project.
2. [`02_graph_creation.md`](02_graph_creation.md) — how to create L0/L1/L2 graphs and graph features.
3. [`03_model_architecture.md`](03_model_architecture.md) — exact encoder/processor/decoder design.
4. [`04_training_plan.md`](04_training_plan.md) — rollout training stages without chunking for the raw prototype.
5. [`05_losses_and_metrics.md`](05_losses_and_metrics.md) — latitude-weighted MSE, rollout loss, graph-gradient loss, metrics.
6. [`06_memory_and_compute.md`](06_memory_and_compute.md) — compute/memory formulas and 5°/2.5°/1.5° estimates.
7. [`07_implementation_checklist.md`](07_implementation_checklist.md) — coding checklist.
8. [`08_config_templates.md`](08_config_templates.md) — suggested config templates.
9. [`09_pseudocode.md`](09_pseudocode.md) — model/training pseudocode.

## First implementation target

Implement the 5° prototype first:

```text
resolution: 5°
physical batch size: 2
hidden size: 96 or 128
neighbors: k=8
rollout stages: 1 → 2 → 4 → 8 → 12 → 20
prediction step: 12h
output: Δx tendency
```

Use the existing non-GNN dataset preparation exactly as much as possible. The GNN model should adapt the batch after loading, not require a separate dataset.
