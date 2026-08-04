# 04 — Training Plan

## First raw prototype training rule

For the first raw model:

- use **5° resolution** first,
- use **physical batch size 2**,
- do **not** implement chunked rollout/backpropagation yet,
- train with full backpropagation through the current rollout length,
- gradually increase rollout length.

Chunked rollout can be added later only if full rollout becomes too memory-heavy.

---

# Forecast setup

Prediction step:

```text
12 hours
```

Input:

```text
[x_{t-12h}, x_t, static/time features]
```

Output:

```text
Δx = x_{t+12h} - x_t
```

Forecast:

```text
x_pred = x_t + Δx
```

Rollout:

```text
Step 1: [x_{t-12h}, x_t]          → x̂_{t+12h}
Step 2: [x_t, x̂_{t+12h}]         → x̂_{t+24h}
Step 3: [x̂_{t+12h}, x̂_{t+24h}]  → x̂_{t+36h}
...
Step 20:                          → x̂_{t+240h}
```

20 rollout steps means:

```text
20 × 12h = 240h = 10 days
```

Ground truth future states are used only for loss, not as inputs.

---

# Rollout curriculum

Do not begin with `S=20`. Start simple.

Recommended stages:

| Stage | Rollout steps S | Forecast horizon | Goal |
|---:|---:|---:|---|
| 1 | 1 | 12h | learn one-step dynamics |
| 2 | 2 | 24h | first self-rollout stability |
| 3 | 4 | 48h | short-range autoregressive stability |
| 4 | 8 | 96h | medium-range stability |
| 5 | 12 | 144h | longer-range stability |
| 6 | 20 | 240h | final 10-day rollout |

For the first 5° prototype, run all stages if memory allows.

For 2.5° and 1.5° later, you may need to reduce hidden size or batch size for higher stages.

---

# Batch size

Use:

```text
physical batch size = 2
```

This means two forecast initializations are loaded on the GPU at once.

Do not start with physical batch 8 for rollout training. Rollout memory scales with:

```text
physical_batch × rollout_steps × activations
```

At 5°, batch 2 and `S=20` may be feasible. At 2.5° or 1.5°, it may become heavy.

---

# Gradient accumulation

For the first raw prototype, gradient accumulation is optional.

If you want effective batch size 8 later:

```text
physical batch = 2
gradient accumulation = 4
effective batch = 8
```

But for the first 5° implementation, keep the loop simple:

```text
physical batch = 2
optimizer step after each batch
```

Add gradient accumulation only after the model is correct.

---

# One-step training loop

Stage 1:

```python
x_prev = x_{t-12h}
x_cur = x_t

delta = model(x_prev, x_cur, static, time)
x_pred = x_cur + delta

loss = loss_fn(x_pred, x_gt_{t+12h})
loss.backward()
optimizer.step()
```

---

# Rollout training loop without chunks

For stage with rollout length `S`:

```python
x_prev = x_{t-12h}
x_cur = x_t
loss_total = 0

for s in range(1, S + 1):
    time_features = make_time_features(t + 12 * (s - 1))

    delta = model(x_prev, x_cur, static, time_features)
    x_next = x_cur + delta

    gt = get_ground_truth(t + 12 * s)
    loss_total += loss_fn(x_next, gt)

    # autoregressive update
    x_prev = x_cur
    x_cur = x_next

loss_total = loss_total / S
loss_total.backward()
optimizer.step()
```

No future GT is fed into the model.

---

# Important memory note

This raw training loop keeps the computation graph for all `S` rollout steps.

That is okay for first 5° experiments, but may be too heavy for large resolutions and large `S`.

If memory fails:

1. reduce `S`,
2. reduce physical batch size,
3. reduce hidden dimension,
4. reduce processor layers,
5. use activation checkpointing,
6. only later add chunked rollout training.

But chunked training should not be part of the first raw prototype.

---

# Recommended 5° training settings

Start:

```text
resolution: 5°
hidden dim: 96 or 128
batch: 2
rollout: S=1
optimizer: AdamW
mixed precision: yes if available
```

Then:

```text
S=1 until validation one-step RMSE decreases reliably
S=2 until stable
S=4
S=8
S=12
S=20
```

Do not increase rollout length if the previous stage is unstable.

---

# Validation

For validation, always run pure autoregressive inference:

```text
start from [x_{t-12h}, x_t]
roll out 20 steps without GT input
compute metrics at each lead time
```

Even if training is currently at `S=4`, validation can include longer rollouts to check stability.

Recommended validation plots:

```text
RMSE vs lead time
ACC vs lead time
forecast activity/anomaly std vs lead time
spatial error maps
sample forecast maps
```
