# 07 — Implementation Checklist

## Phase 0 — connect to existing project

- [ ] Put the existing non-GNN project at the repo root or import path.
- [ ] Confirm how the existing dataset returns batches.
- [ ] Write down exact batch keys and tensor shapes.
- [ ] Confirm variable order and normalization.
- [ ] Confirm target lead time and time indexing.
- [ ] Confirm how static/time features are represented.

---

# Phase 1 — batch adapter

- [ ] Implement `GNNBatchAdapter`.
- [ ] Convert baseline grid tensors to node tensors.
- [ ] Convert node tensors back to baseline grid format.
- [ ] Test grid → node → grid reconstruction.
- [ ] Verify node ordering matches graph node order.
- [ ] Broadcast time features to nodes if needed.

---

# Phase 2 — graph builder

- [ ] Implement graph creation for 5°.
- [ ] Build L0/L1/L2 nodes.
- [ ] Build spherical kNN edges for each level.
- [ ] Compute edge features.
- [ ] Build pool maps.
- [ ] Build unpool maps.
- [ ] Save `graph_5deg.pt`.
- [ ] Add tests for node count and edge count.
- [ ] Add tests for longitude wrap-around.
- [ ] Add tests for polar nodes.

---

# Phase 3 — layers

- [ ] Implement linear embedding.
- [ ] Implement local graph attention block.
- [ ] Implement edge feature embedding.
- [ ] Implement MLP/feed-forward layer.
- [ ] Implement residual connections.
- [ ] Implement LayerNorm.
- [ ] Add shape tests on random graph.

---

# Phase 4 — pooling/unpooling

- [ ] Implement mean pooling.
- [ ] Implement max pooling.
- [ ] Concatenate mean and max.
- [ ] Project pooled feature back to hidden dim.
- [ ] Implement parent unpooling.
- [ ] Implement skip fusion MLP.
- [ ] Test L0 → L1 → L0 shapes.
- [ ] Test L1 → L2 → L1 shapes.

---

# Phase 5 — processor

- [ ] Implement small Graph U-Net processor:
  - [ ] L0 blocks ×2
  - [ ] pool L0→L1
  - [ ] L1 blocks ×2
  - [ ] pool L1→L2
  - [ ] L2 block ×1
  - [ ] unpool L2→L1 + skip fusion
  - [ ] L1 refinement ×1
  - [ ] unpool L1→L0 + skip fusion
  - [ ] L0 refinement ×1
- [ ] Test with random node tensors.

---

# Phase 6 — full model

- [ ] Implement `GraphWeatherModel`.
- [ ] Use existing batch interface.
- [ ] Predict `Δx`.
- [ ] Add `x_current + Δx` inside forward.
- [ ] Return prediction in baseline format.
- [ ] Run one forward pass on one dataset batch.
- [ ] Check output shape equals baseline target shape.

---

# Phase 7 — loss/training

- [ ] Reuse baseline loss first if possible.
- [ ] Add latitude-weighted MSE if not already present.
- [ ] Implement one-step training.
- [ ] Implement rollout training loop without chunks.
- [ ] Verify no future GT is fed as input.
- [ ] Add validation rollout inference.

---

# Phase 8 — rollout curriculum

Train stages:

- [ ] S=1
- [ ] S=2
- [ ] S=4
- [ ] S=8
- [ ] S=12
- [ ] S=20

Do not increase `S` until previous stage is stable.

---

# Phase 9 — scale later

After 5° works:

- [ ] Build `graph_2p5deg.pt`.
- [ ] Train 2.5° with `S=1`.
- [ ] Increase rollout gradually.
- [ ] Build `graph_1p5deg.pt`.
- [ ] Train 1.5° with smaller hidden dim if needed.
- [ ] Only later add chunked rollout/checkpointing if memory requires.

---

# Do not implement yet

For first raw prototype, do not implement:

- [ ] shared latent mesh,
- [ ] O96 graph,
- [ ] adaptive graph edges,
- [ ] learned long-range edges,
- [ ] temporal memory attention,
- [ ] diffusion/probabilistic training,
- [ ] chunked rollout training.

Keep the first model simple.
