# 06 — Memory and Compute

## Main message

For this model, parameter count is not the main memory problem.

Memory mainly comes from:

```text
activations
edge messages
attention weights
MLP activations
skip features
rollout computation graph
```

The static graphs themselves are cheap.

---

# Symbols

```text
B = physical batch size
S = rollout length
K = rollout chunk length, not used in first raw prototype
N = number of nodes at a graph level
E = number of edges at a graph level
k = neighbors per node, usually 8
d = hidden dimension
H = attention heads
r = MLP expansion factor, usually 4
```

For local kNN graph:

```text
E ≈ kN
```

---

# Node activation memory

Node tensor shape:

```text
[B, N, d]
```

Memory:

```text
B × N × d × bytes_per_element
```

For fp16/bf16:

```text
bytes_per_element = 2
```

Example 5° L0 with `B=2`, `N=2664`, `d=128`:

```text
2 × 2664 × 128 × 2 bytes ≈ 1.36 MB
```

This is small.

---

# Edge-message activation memory

Edge message tensor shape:

```text
[B, E, d]
```

Memory:

```text
B × E × d × bytes
```

For 5° L0 with `B=2`, `E=21312`, `d=128`:

```text
2 × 21312 × 128 × 2 bytes ≈ 10.9 MB
```

For 2.5° L0 with `B=2`, `E=84096`, `d=160`:

```text
2 × 84096 × 160 × 2 bytes ≈ 53.8 MB
```

For 1.5° L0 with `B=2`, `E=232320`, `d=160`:

```text
2 × 232320 × 160 × 2 bytes ≈ 148.7 MB
```

This is why graph models can use more activation memory than CNNs.

---

# Local graph attention compute

One local graph-attention block:

```text
O(N d² + E d + E H)
```

Dominant terms are usually:

```text
N d² for projections/MLP
E d for edge messages
```

Since `E = kN`, this is roughly linear in `N` for fixed `k` and `d`.

---

# Full attention warning

Full attention would cost:

```text
O(N² d)
```

Do not use full attention.

At 5°:

```text
N = 2664
N² ≈ 7.1M pairs
```

At 2.5°:

```text
N = 10512
N² ≈ 110M pairs
```

At 1.5°:

```text
N = 29040
N² ≈ 843M pairs
```

This is too expensive for rollout training.

---

# Multiscale processor cost

For Graph U-Net processor:

```text
Total cost ≈ sum over levels l:
    num_blocks_l × O(N_l d² + E_l d + E_l H)
```

Raw prototype block counts:

```text
L0 blocks: 2
L1 blocks: 2
L2 blocks: 1
L1 refinement: 1
L0 refinement: 1
decoder L0 refinement: 1
encoder L0 block: 1
```

Most cost is still on L0.

---

# 5° prototype memory estimate

Assumptions:

```text
resolution = 5°
B = 2
d = 128
k = 8
raw processor = small 2-2-1 Graph U-Net
mixed precision
no chunking
```

Expected rough training memory:

| Rollout S | Expected memory |
|---:|---:|
| 1 | low, likely < 8GB |
| 2 | likely < 12GB |
| 4 | likely 10–18GB |
| 8 | likely 16–28GB |
| 12 | maybe 22–35GB |
| 20 | may fit 40GB if implementation is efficient; reduce d or B if not |

These are rough because PyTorch/PyG overhead can vary a lot.

---

# 2.5° later estimate

Assumptions:

```text
resolution = 2.5°
B = 2
d = 128–160
k = 8
```

Expected:

```text
S=1 to 4 should be manageable.
S=8+ may require checkpointing or smaller hidden dimension.
S=20 without chunks may be difficult.
```

For later 2.5° final training, chunked rollout may become necessary, but it is intentionally excluded from the first raw implementation.

---

# 1.5° later estimate

Assumptions:

```text
resolution = 1.5°
B = 2
d = 128–160
k = 8
```

Expected:

```text
S=1 likely manageable.
S=2–4 may fit depending on implementation.
S=8+ likely needs checkpointing and/or chunked rollout.
S=20 full backprop without chunks is unlikely on one 40GB GPU.
```

---

# Practical knobs

If out of memory, reduce in this order:

1. rollout length `S`,
2. physical batch size `B`,
3. hidden dimension `d`,
4. processor block counts,
5. number of neighbors `k`,
6. attention heads `H`,
7. enable activation checkpointing.

For first raw prototype, do not remove edge features or geometry information.

---

# Why 5° first

5° gives a very small graph:

```text
L0 = 2,664 nodes
```

This lets you debug:

- graph construction,
- batch adapter,
- model shapes,
- rollout loop,
- loss functions,
- output reshaping,
- validation metrics,

before spending memory on 2.5° or 1.5°.
