# 02 — Graph Creation Details

## Overview

The graph is fixed by Earth geometry. We do **not** learn the graph structure in the first prototype.

For each resolution, precompute three graph levels:

```text
L0: native graph
L1: coarse graph
L2: coarsest graph
```

Each graph has:

```text
node coordinates
edge_index
edge_features
```

We also precompute:

```text
pool_map_L0_to_L1
pool_map_L1_to_L2
unpool_map_L2_to_L1
unpool_map_L1_to_L0
```

The graph topology is created once and saved to disk. During training only the node features change.

---

# Resolution node counts

Assuming inclusive latitude grid from `-90°` to `90°` and longitude from `0°` to `360° - step`:

| Resolution | Lat count | Lon count | L0 native nodes |
|---:|---:|---:|---:|
| 5° | 37 | 72 | 2,664 |
| 2.5° | 73 | 144 | 10,512 |
| 1.5° | 121 | 240 | 29,040 |

The first prototype should use **5°**.

---

# Coarse graph sizes with 2×2 coarsening

Use simple 2×2 coarsening for the first implementation.

Because latitude counts are odd, use ceiling when grouping rows.

| Resolution | L0 nodes | L1 nodes | L2 nodes |
|---:|---:|---:|---:|
| 5° | 2,664 | 684 | 180 |
| 2.5° | 10,512 | 2,664 | 684 |
| 1.5° | 29,040 | 7,320 | 1,860 |

How these are computed:

```text
L1_lat = ceil(L0_lat / 2)
L1_lon = L0_lon / 2
L2_lat = ceil(L1_lat / 2)
L2_lon = L1_lon / 2
```

For example, at 5°:

```text
L0: 37 × 72 = 2,664
L1: 19 × 36 =   684
L2: 10 × 18 =   180
```

---

# Edge counts

Use spherical kNN with `k=8` directed neighbors.

Approximate directed edge counts:

| Resolution | Level | Nodes | Edges with k=8 |
|---:|---|---:|---:|
| 5° | L0 | 2,664 | 21,312 |
| 5° | L1 | 684 | 5,472 |
| 5° | L2 | 180 | 1,440 |
| 2.5° | L0 | 10,512 | 84,096 |
| 2.5° | L1 | 2,664 | 21,312 |
| 2.5° | L2 | 684 | 5,472 |
| 1.5° | L0 | 29,040 | 232,320 |
| 1.5° | L1 | 7,320 | 58,560 |
| 1.5° | L2 | 1,860 | 14,880 |

If you symmetrize edges manually, edge count may be closer to `2kN`. Start with directed kNN edges.

---

# Step 1 — create L0 nodes

Each grid point becomes one node.

Use this node indexing:

```text
node_id = lat_index * num_lon + lon_index
```

This must match the dataset flattening order.

---

# Step 2 — convert latitude/longitude to 3D sphere coordinates

For each node:

```text
lat, lon in radians
x = cos(lat) cos(lon)
y = cos(lat) sin(lon)
z = sin(lat)
```

Store:

```python
coords_3d: Tensor[N, 3]
lat_lon: Tensor[N, 2]
```

Using 3D sphere coordinates handles longitude wrap-around naturally.

---

# Step 3 — create spherical kNN edges

For each node, find its `k=8` nearest neighbors on the sphere.

Recommended implementation:

1. Build `coords_3d`.
2. Use a nearest-neighbor search over 3D coordinates.
3. Exclude self from neighbors.
4. Save directed edges.

Edge convention:

```text
edge j → i means node i receives message from node j
```

For PyTorch Geometric style:

```python
edge_index[0] = source_nodes
edge_index[1] = target_nodes
```

---

# Step 4 — compute edge features

For every edge `j → i`, compute features describing the relation from target `i` to source `j`.

Recommended edge feature vector:

```text
edge_feature_ij = [
    normalized_great_circle_distance,
    sin(bearing_i_to_j),
    cos(bearing_i_to_j),
    delta_lat,
    sin(delta_lon),
    cos(delta_lon)
]
```

## Great-circle distance

```text
d = R * arccos(
    sin(lat_i) sin(lat_j)
  + cos(lat_i) cos(lat_j) cos(lon_j - lon_i)
)
```

Usually normalize by Earth radius:

```text
d_norm = d / R
```

## Bearing / compass direction

Bearing tells the model in which direction neighbor `j` lies from node `i`:

```text
0°   = north
90°  = east
180° = south
270° = west
```

Store as:

```text
sin(bearing), cos(bearing)
```

because angles wrap around.

---

# Step 5 — create L1 and L2 nodes

Use 2×2 coarsening.

Each coarse node corresponds to a local group of fine nodes.

Example:

```text
fine nodes:
a b
c d

coarse node:
A
```

The coarse coordinate can be the normalized mean of the 3D coordinates:

```python
coord_A = normalize(mean([coord_a, coord_b, coord_c, coord_d]))
```

Repeat to create L2 from L1.

---

# Step 6 — create L1/L2 edges

After L1 and L2 node coordinates are created, run the same spherical kNN process on each level.

```text
L0: kNN over L0 coords
L1: kNN over L1 coords
L2: kNN over L2 coords
```

Each level has its own `edge_index` and `edge_features`.

---

# Step 7 — create pooling maps

For L0 → L1:

```text
pool_map_0_to_1[fine_node_id] = coarse_node_id
```

For L1 → L2:

```text
pool_map_1_to_2[l1_node_id] = l2_node_id
```

Pooling during forward:

```text
h_coarse = Linear([mean_pool(h_fine), max_pool(h_fine)])
```

Use scatter operations:

```python
scatter_mean(h_fine, pool_map)
scatter_max(h_fine, pool_map)
```

---

# Step 8 — create unpooling maps

Simple first implementation:

```text
unpool from parent coarse node
```

For each fine node:

```text
parent = pool_map[fine_node]
h_unpooled[fine_node] = h_coarse[parent]
```

Better later:

```text
fine node attends to nearest 3–4 coarse nodes
```

For the raw prototype, parent unpooling is acceptable and simple.

---

# Saved graph files

Save one graph bundle per resolution:

```text
graphs/
    graph_5deg.pt
    graph_2p5deg.pt
    graph_1p5deg.pt
```

Each `.pt` file should contain:

```python
{
    "resolution": 5.0,
    "levels": {
        "L0": {
            "lat_lon": ...,
            "coords_3d": ...,
            "edge_index": ...,
            "edge_attr": ...,
            "shape": (37, 72),
        },
        "L1": {...},
        "L2": {...},
    },
    "pool": {
        "L0_to_L1": ...,
        "L1_to_L2": ...,
    },
    "unpool": {
        "L2_to_L1": ...,
        "L1_to_L0": ...,
    }
}
```

---

# Validation tests

Before training:

1. Check node count.
2. Check edge count.
3. Check no NaNs in edge features.
4. Check longitude wrap-around neighbors.
5. Check polar node neighbors.
6. Check pooling maps cover every fine node.
7. Check unpooling returns expected shape.
8. Check flatten/unflatten order against dataset.

Do not train until these pass.
