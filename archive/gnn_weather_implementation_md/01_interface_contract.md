# 01 — Interface Contract With Existing Non-GNN Project

## Main rule

Do **not** create a separate dataset pipeline for the GNN model.

The existing non-GNN project should remain the source of truth for:

- data loading,
- variable selection,
- pressure-level ordering,
- normalization/statistics,
- train/validation/test splits,
- time indexing,
- static features,
- target construction,
- evaluation format.

The GNN implementation should be added as a new model/backend that consumes the **same batch object** produced by the existing dataset.

## Why this matters

We want to compare the CNN/baseline model and the GNN model fairly. If the GNN uses a different dataset class, different variable ordering, different normalization, or different target construction, results will be hard to compare.

The goal is:

```text
same dataset → baseline model
same dataset → GNN model
```

Only the model adapter should be different.

---

# Expected batch principle

The exact keys may differ in your existing project. Keep the names from the existing project. Do not force new names if the baseline already uses another convention.

Typical baseline batch might look like one of these:

```python
batch = {
    "x":      Tensor[B, T_in, C, H, W],
    "y":      Tensor[B, C_out, H, W],
    "static": Tensor[B or 1, C_static, H, W],
    "time":   Tensor[B, C_time],
    "lead_time": ...,
}
```

or:

```python
batch = {
    "input":  Tensor[B, C_in, H, W],
    "target": Tensor[B, C_out, H, W],
    "metadata": {...},
}
```

The GNN code should not assume new dataset logic. Instead, write a **batch adapter** that converts the existing batch into node features.

---

# Required adapter

Create a module like:

```text
models/gnn/batch_adapter.py
```

Responsibilities:

1. Read the baseline batch format.
2. Extract:
   - previous state `x_{t-12h}`,
   - current state `x_t`,
   - static fields,
   - time features.
3. Flatten the spatial grid into nodes.
4. Return node tensor:

```python
node_x: Tensor[B, N, C_in]
current_state_nodes: Tensor[B, N, C_out]
target_nodes: Tensor[B, N, C_out]
metadata needed to unflatten back
```

The adapter should also convert model output back to baseline format:

```python
pred_nodes: Tensor[B, N, C_out]
    ↓
pred_grid: Tensor[B, C_out, H, W]
```

This lets the existing trainer/evaluator consume the prediction.

---

# Node ordering must match dataset flattening

Use a fixed flattening convention, preferably the same as the baseline tensor layout:

```text
node_id = lat_index * num_lon + lon_index
```

For a tensor `x[B, C, H, W]`, flatten as:

```python
x_nodes = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
```

Unflatten as:

```python
x_grid = x_nodes.reshape(B, H, W, C).permute(0, 3, 1, 2)
```

This is critical. If graph node ordering and data flattening do not match, the model will learn nonsense.

---

# Model forward interface

Try to match the baseline model as closely as possible. If the baseline expects:

```python
pred = model(batch)
```

then implement:

```python
class GraphWeatherModel(nn.Module):
    def forward(self, batch):
        node_batch = self.adapter.to_nodes(batch)
        delta_nodes = self.backbone(node_batch)
        pred_nodes = node_batch.current_state + delta_nodes
        pred_grid = self.adapter.to_grid(pred_nodes, batch)
        return pred_grid
```

If the baseline returns a dictionary, return the same structure:

```python
return {
    "prediction": pred_grid,
    "delta": delta_grid,
}
```

The training loop should need minimal changes.

---

# Target format

The model predicts tendency:

```text
Δx = x_{t+12h} - x_t
forecast = x_t + Δx
```

But to keep compatibility, the final output should be the **forecast state**, not just `Δx`, if the baseline trainer/evaluator expects forecast states.

Inside the model:

```python
delta = backbone(...)
prediction = current_state + delta
```

Loss can be computed against the existing target state:

```python
loss(prediction, target)
```

Optionally, add a separate tendency loss later.

---

# Static/time features

Use the same static/time features as the baseline project. If baseline uses:

- orography,
- land-sea mask,
- latitude/longitude channels,
- day-of-year/time-of-day encodings,

then the GNN should use the same.

If static features are stored in grid format, flatten them to nodes and concatenate with weather states:

```text
node_feature_i = [x_{t-12h, i}, x_{t, i}, static_i, time_features]
```

Time features may be global per sample. Broadcast them to all nodes:

```python
time_nodes = time_features[:, None, :].expand(B, N, C_time)
```

---

# Keep one dataset usable by both models

Recommended project layout:

```text
repo_root/
    baseline_project/
        datasets/
        trainer/
        configs/
        ...

    models/
        cnn_baseline_wrapper.py
        gnn_weather/
            graph_weather_model.py
            batch_adapter.py
            graph_builder.py
            layers.py
            processor.py
```

The GNN should import/reuse the baseline dataset classes rather than replacing them.

---

# First rule for debugging

Before training, run this test:

1. Load one baseline batch.
2. Convert grid → nodes.
3. Convert nodes → grid.
4. Check exact reconstruction.

```python
x_grid_reconstructed = adapter.to_grid(adapter.to_nodes(batch).x_current)
assert max_abs_error(x_grid_reconstructed, x_current) < 1e-6
```

If this fails, graph training should not start.
