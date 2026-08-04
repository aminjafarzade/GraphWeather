# 09 — Pseudocode

This is implementation-oriented pseudocode. Adjust it to the existing project style.

---

# Batch adapter

```python
class GNNBatchAdapter:
    def __init__(self, grid_shape, variable_count):
        self.H, self.W = grid_shape
        self.C_out = variable_count

    def grid_to_nodes(self, x_grid):
        # x_grid: [B, C, H, W]
        return x_grid.permute(0, 2, 3, 1).reshape(x_grid.shape[0], self.H * self.W, x_grid.shape[1])

    def nodes_to_grid(self, x_nodes):
        # x_nodes: [B, N, C]
        B, N, C = x_nodes.shape
        return x_nodes.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)

    def to_node_batch(self, batch):
        # This function must adapt to the existing baseline batch keys.
        # Example only:
        x_prev_grid = batch["x_prev"]      # [B, C, H, W]
        x_cur_grid  = batch["x_cur"]       # [B, C, H, W]
        y_grid      = batch["target"]      # [B, C, H, W]
        static_grid = batch.get("static")  # [B or 1, C_static, H, W]
        time_feat   = batch.get("time")    # [B, C_time]

        x_prev = self.grid_to_nodes(x_prev_grid)
        x_cur  = self.grid_to_nodes(x_cur_grid)
        y      = self.grid_to_nodes(y_grid)

        features = [x_prev, x_cur]

        if static_grid is not None:
            if static_grid.shape[0] == 1:
                static_grid = static_grid.expand(x_cur_grid.shape[0], -1, -1, -1)
            static_nodes = self.grid_to_nodes(static_grid)
            features.append(static_nodes)

        if time_feat is not None:
            B, N, _ = x_cur.shape
            time_nodes = time_feat[:, None, :].expand(B, N, time_feat.shape[-1])
            features.append(time_nodes)

        node_x = torch.cat(features, dim=-1)

        return {
            "node_x": node_x,
            "x_current_nodes": x_cur,
            "target_nodes": y,
            "original_batch": batch,
        }
```

---

# Local graph attention block

```python
class LocalGraphAttentionBlock(nn.Module):
    def __init__(self, d, edge_dim, heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = LocalGraphAttention(d, edge_dim, heads)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_ratio * d),
            nn.GELU(),
            nn.Linear(mlp_ratio * d, d),
        )

    def forward(self, h, edge_index, edge_attr):
        h = h + self.attn(self.norm1(h), edge_index, edge_attr)
        h = h + self.mlp(self.norm2(h))
        return h
```

The attention implementation should only attend over graph edges, not the full graph.

---

# Pooling

```python
class MeanMaxPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(2 * d, d)

    def forward(self, h_fine, pool_map, num_coarse):
        # h_fine: [B, N_fine, d]
        # pool_map: [N_fine], maps fine node → coarse node
        mean = scatter_mean(h_fine, pool_map, dim=1, dim_size=num_coarse)
        maxv = scatter_max(h_fine, pool_map, dim=1, dim_size=num_coarse)
        return self.proj(torch.cat([mean, maxv], dim=-1))
```

Use the actual scatter API from your chosen library.

---

# Parent unpool + skip fusion

```python
class ParentUnpoolFuse(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(self, h_coarse, parent_map, h_skip):
        # parent_map: [N_fine], maps fine node → coarse parent
        h_up = h_coarse[:, parent_map, :]
        return self.fuse(torch.cat([h_skip, h_up], dim=-1))
```

---

# Graph U-Net processor

```python
class GraphUNetProcessor(nn.Module):
    def __init__(self, d, edge_dim, graph_bundle, heads=4):
        super().__init__()
        self.graph = graph_bundle

        self.l0_blocks = nn.ModuleList([
            LocalGraphAttentionBlock(d, edge_dim, heads) for _ in range(2)
        ])
        self.pool01 = MeanMaxPool(d)

        self.l1_blocks = nn.ModuleList([
            LocalGraphAttentionBlock(d, edge_dim, heads) for _ in range(2)
        ])
        self.pool12 = MeanMaxPool(d)

        self.l2_blocks = nn.ModuleList([
            LocalGraphAttentionBlock(d, edge_dim, heads) for _ in range(1)
        ])

        self.unpool21 = ParentUnpoolFuse(d)
        self.l1_refine = LocalGraphAttentionBlock(d, edge_dim, heads)

        self.unpool10 = ParentUnpoolFuse(d)
        self.l0_refine = LocalGraphAttentionBlock(d, edge_dim, heads)

    def forward(self, h0):
        g = self.graph

        for block in self.l0_blocks:
            h0 = block(h0, g.L0.edge_index, g.L0.edge_attr)
        skip0 = h0

        h1 = self.pool01(h0, g.pool.L0_to_L1, g.L1.num_nodes)
        for block in self.l1_blocks:
            h1 = block(h1, g.L1.edge_index, g.L1.edge_attr)
        skip1 = h1

        h2 = self.pool12(h1, g.pool.L1_to_L2, g.L2.num_nodes)
        for block in self.l2_blocks:
            h2 = block(h2, g.L2.edge_index, g.L2.edge_attr)

        h1 = self.unpool21(h2, g.pool.L1_to_L2, skip1)
        h1 = self.l1_refine(h1, g.L1.edge_index, g.L1.edge_attr)

        h0 = self.unpool10(h1, g.pool.L0_to_L1, skip0)
        h0 = self.l0_refine(h0, g.L0.edge_index, g.L0.edge_attr)

        return h0
```

Note: the `parent_map` for `L2→L1` is the same mapping used for `L1→L2`: each L1 node knows its L2 parent.

---

# Full model

```python
class GraphWeatherModel(nn.Module):
    def __init__(self, adapter, graph_bundle, c_in, c_out, d=128, edge_dim=6, heads=4):
        super().__init__()
        self.adapter = adapter
        self.graph = graph_bundle

        self.embed = nn.Linear(c_in, d)

        self.encoder = nn.ModuleList([
            LocalGraphAttentionBlock(d, edge_dim, heads)
        ])

        self.processor = GraphUNetProcessor(d, edge_dim, graph_bundle, heads)

        self.decoder = LocalGraphAttentionBlock(d, edge_dim, heads)
        self.head = nn.Linear(d, c_out)

    def forward(self, batch):
        nb = self.adapter.to_node_batch(batch)
        h = self.embed(nb["node_x"])

        for block in self.encoder:
            h = block(h, self.graph.L0.edge_index, self.graph.L0.edge_attr)

        h = self.processor(h)
        h = self.decoder(h, self.graph.L0.edge_index, self.graph.L0.edge_attr)

        delta = self.head(h)
        pred_nodes = nb["x_current_nodes"] + delta
        pred_grid = self.adapter.nodes_to_grid(pred_nodes)
        return pred_grid
```

---

# Rollout training without chunks

```python
def train_rollout_step(model, batch, rollout_steps, loss_fn, optimizer):
    optimizer.zero_grad()

    # You need helper functions from the existing dataset/project to obtain
    # initial states and rollout targets from the batch.
    x_prev, x_cur = get_initial_states(batch)
    static = get_static(batch)

    total_loss = 0.0

    for s in range(1, rollout_steps + 1):
        step_batch = make_step_batch(
            original_batch=batch,
            x_prev=x_prev,
            x_cur=x_cur,
            static=static,
            time_index=s,
        )

        pred = model(step_batch)
        gt = get_rollout_target(batch, step=s)

        total_loss = total_loss + loss_fn(pred, gt)

        # autoregressive update: use prediction, not GT
        x_prev = x_cur
        x_cur = pred

    total_loss = total_loss / rollout_steps
    total_loss.backward()
    optimizer.step()

    return total_loss.item()
```

This keeps full backpropagation through the rollout length. That is intended for the first raw 5° prototype.
