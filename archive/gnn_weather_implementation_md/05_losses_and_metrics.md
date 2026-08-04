# 05 — Losses and Metrics

## Main loss: latitude-weighted MSE

Normal MSE treats every grid point equally:

```text
MSE = mean((prediction - target)^2)
```

But on a latitude-longitude grid, polar regions have many grid points packed into a small physical area. To avoid over-weighting poles, use latitude weighting:

```text
weight(lat) = cos(lat)
```

Loss:

```text
L = sum_i w_i * (pred_i - gt_i)^2 / sum_i w_i
```

Use this as the main training loss.

---

# Variable weighting

Different variables and pressure levels have different scales and importance.

Use the same normalization and variable ordering as the existing baseline project.

If baseline already normalizes each variable/level, keep that.

Optional variable weights:

```text
L = sum_variables alpha_v * latitude_weighted_MSE_v
```

Start simple:

```text
all normalized variables weighted equally
```

Only add custom variable weights after baseline works.

---

# Tendency loss vs state loss

The model outputs:

```text
Δx
```

Forecast:

```text
x_pred = x_current + Δx
```

The simplest loss compares forecast state to target state:

```text
L_state = loss(x_pred, x_gt)
```

Optional later:

```text
L_delta = loss(Δx_pred, x_gt - x_current)
```

For first prototype:

```text
use L_state only
```

---

# Rollout loss

For rollout length `S`:

```text
L_rollout = (1/S) * sum_{s=1}^{S} L(x_pred_{t+12s}, x_gt_{t+12s})
```

This teaches the model to survive its own predictions.

If `S=1`, rollout loss is just one-step loss.

---

# Graph-gradient loss

MSE can encourage blurry predictions. A graph-gradient loss preserves local spatial changes.

For every graph edge `(i, j)`:

```text
pred_edge_diff = pred_i - pred_j
gt_edge_diff   = gt_i - gt_j
```

Loss:

```text
L_grad = mean_edges || pred_edge_diff - gt_edge_diff ||^2
```

This helps preserve:

- fronts,
- wind gradients,
- pressure gradients,
- sharper local structures.

Start without it, then add after the main model trains:

```text
L_total = L_mse + lambda_grad * L_grad
```

Suggested first value:

```text
lambda_grad = 0.01 to 0.05
```

---

# Spectral loss

Spectral loss compares the spatial frequency content of predicted and ground-truth fields.

Purpose:

```text
prevent MSE-induced smoothing/blurring
```

For regular lat-lon grids, a simple approximate version uses FFT:

```text
L_spectral = MSE(|FFT(pred)|, |FFT(gt)|)
```

However, for the graph model, start with graph-gradient loss first. It is easier and more graph-native.

Add spectral loss only later as an ablation.

---

# Activity / anomaly-standard-deviation loss

Blurry forecasts often have too little variability.

Activity loss compares anomaly standard deviation:

```text
L_activity = || std(pred_anomaly) - std(gt_anomaly) ||^2
```

This can help prevent the forecast from becoming too flat at long lead times.

Not needed for first raw implementation.

---

# Metrics

At minimum:

```text
latitude-weighted RMSE
ACC
lead-time RMSE curves
lead-time ACC curves
```

Also recommended:

```text
spatial error maps
forecast activity / anomaly std
sample forecast maps
graph-gradient error
```

For rollout validation:

```text
compute metrics for all lead times:
12h, 24h, 36h, ..., 240h
```

---

# Fair comparison with baseline model

Use exactly the same:

- dataset split,
- normalization,
- variables,
- pressure levels,
- target lead times,
- evaluation code,
- metric definitions.

The only difference should be the model architecture.
