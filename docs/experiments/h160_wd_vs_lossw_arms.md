# hidden-160 dense-L3K24: weight-decay vs loss-channel-weighting arms

Two single-variable arms against the existing hidden-160 dense-L3K24 control.
Each arm trains **from scratch**: S1 for 100 epochs, then an S2..S10 curriculum
at 3 epochs per horizon warm-started from **that arm's own S1 checkpoint**.

## The three runs

| Run | S1 config / run name | Curriculum config / run name | Differs from control by |
|---|---|---|---|
| control | `config_2p5_l3_hidden160_base_S1_100epoch_dense_l3k24.yaml`<br>`2p5_l3_h160_densel3k24_s1x100` | `config_2p5_l3_hidden160_dense_l3k24_curriculum_S2toS10_3ep_initckpt.yaml`<br>`2p5_l3_h160_densel3k24_currS2toS10x3_initckpt` | — (reference) |
| arm A | `..._dense_l3k24_wd1e2.yaml`<br>`2p5_l3_h160_densel3k24_s1x100_wd1e2` | `..._initckpt_wd1e2.yaml`<br>`2p5_l3_h160_densel3k24_currS2toS10x3_initckpt_wd1e2` | `weight_decay: 1.0e-2` (control `1.0e-4`) |
| arm B | `..._dense_l3k24_lossw_floor02.yaml`<br>`2p5_l3_h160_densel3k24_s1x100_lossw_floor02` | `..._initckpt_lossw_floor02.yaml`<br>`2p5_l3_h160_densel3k24_currS2toS10x3_initckpt_lossw_floor02` | `loss_channel_weighting` enabled with `min_level_weight: 0.2`, `reference_level: 1000` |

Within each arm S1 and the curriculum carry **identical** `weight_decay` and
`loss_channel_weighting`, so both phases optimise the same objective.

## Notes on the wiring

- **`init_from_checkpoint` is config-driven, not derived from `--s1-name`.**
  `run_pipeline.sh`'s `train_one()` passes only
  `--config/--config_name/--resolution_mode/--exp_dir/--experiment_name`; it never
  passes `--init_from_checkpoint`. Each arm's curriculum config therefore names its
  own S1 run explicitly:
  `init_from_checkpoint: runs/2p5_l3_h160_densel3k24_s1x100_<arm>/best_ckpt.tar`.
- **`--s1-name` / `--curr-name` must equal the config's top-level section key**, because
  the script passes the same string to both `--config_name` (selects the YAML section)
  and `--experiment_name` (names `runs/<name>/`). They match for all four configs.
- **`clim` is omitted from `--stages`.** The 2.5° climatology
  (`data/stats/2p5_train_dayofyear_climatology.nc`) already exists, so the control's
  stored metrics stay comparable. `stage_clim()` would skip it anyway, but leaving the
  stage out makes that explicit.
- **Call `run_pipeline.sh` directly, not `run_full_pipeline.sh`.** The latter does
  `export RUN_DASHMAPS="${RUN_DASHMAPS:-0}"` — an overridable default rather than a
  hard override, and `--stages` beats the `RUN_*` toggles anyway, so these specific
  commands would still get `dashmaps`. Going direct just removes the shim and the
  chance of an inherited toggle mattering.
- `--compare` points at the **real** control run directory. `runs/` carries
  back-compat symlinks from the old naming scheme
  (`dense_l3k24_hidden160_curriculum_S2toS10_3ep_initckpt` → the path below); the
  canonical directory is used here.

## Arm A — weight decay 1e-2

```bash
bash scripts/run_pipeline.sh \
  --resolution 2p5 \
  --s1-config   configs/experiments/config_2p5_l3_hidden160_base_S1_100epoch_dense_l3k24_wd1e2.yaml \
  --s1-name     2p5_l3_h160_densel3k24_s1x100_wd1e2 \
  --curr-config configs/experiments/config_2p5_l3_hidden160_dense_l3k24_curriculum_S2toS10_3ep_initckpt_wd1e2.yaml \
  --curr-name   2p5_l3_h160_densel3k24_currS2toS10x3_initckpt_wd1e2 \
  --primary-label h160_wd1e2 \
  --stages train,eval,plot,qual,diag,dashmaps \
  --compare runs/2p5_l3_h160_densel3k24_currS2toS10x3_initckpt \
  --compare-labels "control_h160" \
  --gpu 0
```

## Arm B — loss-channel weighting, floor 0.2 at reference level 1000

```bash
bash scripts/run_pipeline.sh \
  --resolution 2p5 \
  --s1-config   configs/experiments/config_2p5_l3_hidden160_base_S1_100epoch_dense_l3k24_lossw_floor02.yaml \
  --s1-name     2p5_l3_h160_densel3k24_s1x100_lossw_floor02 \
  --curr-config configs/experiments/config_2p5_l3_hidden160_dense_l3k24_curriculum_S2toS10_3ep_initckpt_lossw_floor02.yaml \
  --curr-name   2p5_l3_h160_densel3k24_currS2toS10x3_initckpt_lossw_floor02 \
  --primary-label h160_lossw_floor02 \
  --stages train,eval,plot,qual,diag,dashmaps \
  --compare runs/2p5_l3_h160_densel3k24_currS2toS10x3_initckpt \
  --compare-labels "control_h160" \
  --gpu 1
```

Append `--dry-run` to either command to print every stage command without executing.

## Arm B: what the weights become

`loss_channel_weighting.reference_level: 1000` pins the scale to an absolute level
instead of normalizing by the mean of the levels present, and `min_level_weight: 0.2`
floors the result: `weight = max(0.2, level / 1000)`.

**Resolved weights for the actual ERA5-67 2.5° channel set** (levels 1000, 925, 850,
800, 700, 600, 500, 400, 300, 200, 100, 50 — five variables `u`,`v`,`t`,`q`,`z` per
level, 60 pressure channels):

| level | 50 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 850 | 925 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weight | 0.20 | 0.20 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.85 | 0.93 | 1.00 |

Surface channels (`t2m`, `msl`, `sp`, `tcwv`, `skt`, `tisr`, `orog`) keep weight 1.0.
`orog`/`tisr` weights are inert because the loss `channel_mask` already excludes them.

The formula is also unit-tested against a reference table that includes 250 hPa (a
standard ERA5 level that this dataset does not carry) and omits 800 — because with
`reference_level: 1000` the weight depends only on the level, so the contract holds
for any level set. See `tests/test_loss_channel_weighting.py`.

### Caveat: this shifts the surface/upper-air balance, not just the inter-level tilt

The control has **no** `loss_channel_weighting` block, so its loss is fully unweighted
(`channel_weights=None`, every channel at 1.0). Arm B leaves surface channels at 1.0
but puts every pressure channel in `[0.2, 1.0]` — mean **0.5563** across the 12 levels
above. So relative to the control, arm B does not merely re-tilt the pressure levels
against each other: it also raises the surface block's effective share of the loss by
about **1.8x** (`1 / 0.5563`) versus the pressure block. Read arm B as "upper levels
down-weighted, floor 0.2, surface held at 1.0", not as a level-tilt at constant
surface/upper-air balance. A mean-normalized variant (`reference_level: 0` with a
floor) would keep the pressure block's total nearer the control's.

The resolved vector is logged per channel at startup — grep the run log:

```bash
grep 'LOSS_CHANNEL_WEIGHT' runs/2p5_l3_h160_densel3k24_s1x100_lossw_floor02/out.log
```

Both resolved keys are also written into `runs/<name>/config_resolved.yaml`.
