# 08 — Config Templates

These are templates. Adapt names/paths to the existing non-GNN project.

---

# 5° raw prototype

```yaml
experiment_name: gnn_graph_unet_5deg_raw

model:
  name: GraphWeatherModel
  resolution: 5.0
  graph_path: graphs/graph_5deg.pt
  input_mode: two_state_12h
  predict: tendency_delta_x
  hidden_dim: 128
  edge_dim: 6
  num_heads: 4
  k_neighbors: 8

  encoder:
    type: local_graph_attention
    num_blocks: 1

  processor:
    type: graph_unet
    l0_blocks: 2
    l1_blocks: 2
    l2_blocks: 1
    l1_refine_blocks: 1
    l0_refine_blocks: 1
    pooling: mean_max
    unpooling: parent

  decoder:
    type: local_graph_refinement
    num_blocks: 1

  output_head:
    out_channels: 67

training:
  forecast_step_hours: 12
  physical_batch_size: 2
  mixed_precision: true
  optimizer: AdamW
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  rollout_schedule: [1, 2, 4, 8, 12, 20]
  chunked_rollout: false
  gradient_accumulation: 1

loss:
  main: latitude_weighted_mse
  graph_gradient_loss: false
  graph_gradient_weight: 0.0

interface:
  use_existing_dataset: true
  preserve_baseline_batch_keys: true
  adapter: GNNBatchAdapter
```

---

# 5° tiny debug

```yaml
experiment_name: gnn_graph_unet_5deg_tiny_debug

model:
  name: GraphWeatherModel
  resolution: 5.0
  graph_path: graphs/graph_5deg.pt
  hidden_dim: 64
  num_heads: 2
  k_neighbors: 8

  encoder:
    num_blocks: 1

  processor:
    l0_blocks: 1
    l1_blocks: 1
    l2_blocks: 1
    l1_refine_blocks: 1
    l0_refine_blocks: 1
    pooling: mean_max
    unpooling: parent

  decoder:
    num_blocks: 1

training:
  forecast_step_hours: 12
  physical_batch_size: 2
  mixed_precision: true
  rollout_schedule: [1]
  chunked_rollout: false
```

---

# Later 2.5° config

```yaml
experiment_name: gnn_graph_unet_2p5deg_raw

model:
  name: GraphWeatherModel
  resolution: 2.5
  graph_path: graphs/graph_2p5deg.pt
  hidden_dim: 128
  num_heads: 4
  k_neighbors: 8

  encoder:
    num_blocks: 1

  processor:
    l0_blocks: 2
    l1_blocks: 2
    l2_blocks: 1
    l1_refine_blocks: 1
    l0_refine_blocks: 1
    pooling: mean_max
    unpooling: parent

  decoder:
    num_blocks: 1

training:
  forecast_step_hours: 12
  physical_batch_size: 2
  mixed_precision: true
  rollout_schedule: [1, 2, 4, 8]
  chunked_rollout: false
```

---

# Later 1.5° initial config

```yaml
experiment_name: gnn_graph_unet_1p5deg_initial

model:
  name: GraphWeatherModel
  resolution: 1.5
  graph_path: graphs/graph_1p5deg.pt
  hidden_dim: 128
  num_heads: 4
  k_neighbors: 8

  encoder:
    num_blocks: 1

  processor:
    l0_blocks: 2
    l1_blocks: 2
    l2_blocks: 1
    l1_refine_blocks: 1
    l0_refine_blocks: 1
    pooling: mean_max
    unpooling: parent

  decoder:
    num_blocks: 1

training:
  forecast_step_hours: 12
  physical_batch_size: 2
  mixed_precision: true
  rollout_schedule: [1, 2, 4]
  chunked_rollout: false
```

---

# Notes

For the first raw prototype, keep:

```yaml
chunked_rollout: false
```

Do not add advanced modules until the 5° model works.
