# GraphWeather5p625

Self-contained direct-grid spherical Graph U-Net prototype for weather forecasting on the `KAI_5/kai_data_5p625` dataset.

This package does not import from `GNN/KAI-Atmos`. It keeps a similar user interface: YAML configs, `scripts/train.py`, KAI-style NetCDF data loading, experiment folders, logs, and checkpoints.

## What This Version Implements

This version implements a 5.625-degree graph weather model and fixes the training/validation reporting issues from the earlier prototype.

- Uses the actual KAI_5 5.625-degree grid: `32 x 64 = 2048` native nodes.
- Builds a three-level graph pyramid:
  - `L0: 32 x 64 = 2048`
  - `L1: 16 x 32 = 512`
  - `L2: 8 x 16 = 128`
- Uses directed spherical kNN edges with `k=8`.
- Predicts a tendency `delta`, then returns `x_current + delta`.
- Supports unchunked autoregressive rollout training.
- Uses daily rollout defaults capped at `S=10`, because the current KAI_5 files are daily.
- Fixes misleading validation by validating at the current training rollout when configured.
- Adds multi-horizon validation logs such as `valid_S1`, `valid_S2`, `valid_S4`, `valid_S8`, `valid_S10`.
- Adds rollout-stage-aware LR scheduling so LR can decrease as rollout length increases.
- Saves checkpoint metadata with train/valid loss, rollout lengths, LR, and multi-horizon validation metrics.

The architecture itself is not tied to a specific experiment folder. The included `experiments/*/out.log` files are only logs from previous smoke/full runs. Checkpoint `.tar` files are intentionally not included in the zip archive.

## Repository Layout

```text
GraphWeather5p625/
  configs/gnn_5p625.yaml       # training configs
  graphs/graph_5p625.pt        # prebuilt 5.625-degree graph bundle
  scripts/build_graph.py       # graph builder CLI
  scripts/train.py             # training CLI
  src/
    batch_adapter.py           # grid <-> node conversion
    config.py                  # YAML loader and logging
    data.py                    # KAI-style NetCDF dataset
    graph_builder.py           # spherical graph construction
    graph_bundle.py            # graph bundle loading
    layers.py                  # local graph attention blocks
    losses.py                  # latitude-weighted MSE and graph-gradient helper
    models.py                  # GraphWeatherModel
    pooling.py                 # mean/max pool and parent unpool
    processor.py               # Graph U-Net processor
    trainer.py                 # training, validation, LR schedule, checkpoints
  requirements.txt
  README.md
```

## Environment Setup

Use Python 3.10 or newer.

### Option A: conda

```bash
cd /lustre/home/ziya/GNN/GraphWeather5p625

conda create -n graphweather5p625 python=3.10 -y
conda activate graphweather5p625

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option B: venv

```bash
cd /lustre/home/ziya/GNN/GraphWeather5p625

python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA driver if the default wheel is not suitable. The rest of the dependencies are listed in `requirements.txt`.

## Data Expected

The default config points to:

```text
/lustre/home/ziya/KAI_5/kai_data_5p625/train
/lustre/home/ziya/KAI_5/kai_data_5p625/valid
/lustre/home/ziya/KAI_5/kai_data_5p625/stats/global_mean.npy
/lustre/home/ziya/KAI_5/kai_data_5p625/stats/global_std.npy
```

Each NetCDF file is expected to contain:

```text
fields[time, channel, latitude, longitude]
```

The included KAI_5 files are daily. With these files, `dt: 1` means one forecast day, and `S=10` means a 10-day rollout.

If your data lives elsewhere, edit `train_data_path`, `valid_data_path`, `global_means_path`, and `global_stds_path` in `configs/gnn_5p625.yaml`.

## Build Or Rebuild The Graph

The archive includes `graphs/graph_5p625.pt`. Rebuild it only if you change the grid or want to regenerate it:

```bash
cd /lustre/home/ziya/GNN/GraphWeather5p625

python scripts/build_graph.py \
  --output graphs/graph_5p625.pt \
  --resolution 5.625 \
  --lat-count 32 \
  --lon-count 64 \
  --lat-start -87.1875 \
  --lon-start -180.0 \
  --k 8
```

## Smoke Tests

Basic 2-epoch smoke run:

```bash
cd /lustre/home/ziya/GNN/GraphWeather5p625

python scripts/train.py \
  --yaml_config configs/gnn_5p625.yaml \
  --config smoke_5p625 \
  --run_num smoke01
```

Rollout/LR validation smoke run:

```bash
python scripts/train.py \
  --yaml_config configs/gnn_5p625.yaml \
  --config smoke_rollout_lr_5p625 \
  --run_num smoke_rollout_lr01
```

Expected behavior for `smoke_rollout_lr_5p625`:

```text
Epoch 1 ... train ... S=1 | valid ... S=1 | valid_S1 ... valid_S2 ... | lr 1.00e-04
Epoch 2 ... train ... S=2 | valid ... S=2 | valid_S1 ... valid_S2 ... | lr 7.00e-05
```

Partial fine-tune smoke run, training only decoder and output head:

```bash
python scripts/train.py \
  --yaml_config configs/gnn_5p625.yaml \
  --config smoke_finetune_head_5p625 \
  --run_num smoke_head01
```

## Full Training

The `raw_5p625` config is set for 150 epochs:

```bash
cd /lustre/home/ziya/GNN/GraphWeather5p625

python scripts/train.py \
  --yaml_config configs/gnn_5p625.yaml \
  --config raw_5p625 \
  --run_num full150
```

Default daily rollout schedule:

```yaml
max_rollout_steps: 10
rollout_schedule: [1, 2, 4, 8, 10]
rollout_stage_epochs: [5, 5, 10, 10, 10]
```

After the listed stages finish, training remains at `S=10`.

Default rollout-stage LR schedule:

```yaml
lr_schedule_type: rollout_stage
lr_by_rollout:
  1: 1.0e-4
  2: 1.0e-4
  4: 7.0e-5
  8: 5.0e-5
  10: 3.0e-5
warmup_epochs: 1
```

## Validation Behavior

Main validation loss is controlled by:

```yaml
validate_with_train_rollout: true
valid_rollout_steps: 1
eval_rollout_steps: [1, 2, 4, 8, 10]
```

When `validate_with_train_rollout: true`, the main validation loss uses the same rollout length as training for that epoch. Multi-horizon validation still logs every horizon listed in `eval_rollout_steps`.

When `validate_with_train_rollout: false`, the main validation loss uses `valid_rollout_steps`, while multi-horizon validation remains optional.

## Outputs

Training writes to:

```text
experiments/<config>_<run_num>/
  out.log
  ckpt.tar
  best_ckpt.tar
```

The zip archive excludes `ckpt.tar` and `best_ckpt.tar` checkpoint files but keeps `out.log` files.

