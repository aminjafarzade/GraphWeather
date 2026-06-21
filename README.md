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
- Adds multi-horizon validation logs such as `valid_S1`, `valid_S2`, `valid_S4`, `valid_S6`, `valid_S8`, `valid_S10`.
- Adds rollout-stage-aware LR scheduling so LR can decrease as rollout length increases.
- Saves global and rollout-stage-specific best checkpoints with metadata.
- Fixes stage-comparison plots so `trained S=4` means a checkpoint trained during the S=4 rollout stage, evaluated with the same fixed rollout horizon as the other checkpoints.

The architecture itself is not tied to a specific experiment folder. The included `experiments/*/out.log` files are only logs from previous smoke/full runs. Checkpoint `.tar` files are intentionally not included in the zip archive.

## Repository Layout

```text
GraphWeather5p625/
  configs/gnn_5p625.yaml       # training configs
  graphs/graph_5p625.pt        # prebuilt 5.625-degree graph bundle
  scripts/build_graph.py       # graph builder CLI
  scripts/evaluate.py          # RMSE/ACC evaluation CLI
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
    evaluator.py               # per-variable RMSE/ACC rollout evaluation
    trainer.py                 # training, validation, LR schedule, checkpoints
  requirements.txt
  README.md
```

## Environment Setup

Use Python 3.10 or newer.

Clone the repository and run commands from the repository root:

```bash
git clone https://github.com/aminjafarzade/GraphWeather.git GraphWeather5p625
cd GraphWeather5p625
```

### Option A: conda

```bash
conda create -n graphweather5p625 python=3.10 -y
conda activate graphweather5p625

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option B: venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA driver if the default wheel is not suitable. The rest of the dependencies are listed in `requirements.txt`.

The rollout map visualization requires Cartopy. If `pip install -r requirements.txt` cannot build Cartopy cleanly on your system, install it from conda-forge with `conda install -c conda-forge cartopy`.

## Data Expected

The default config uses repo-relative template paths:

```text
data/kai_data_5p625/train
data/kai_data_5p625/valid
data/kai_data_5p625/test
data/kai_data_5p625/stats/global_mean.npy
data/kai_data_5p625/stats/global_std.npy
```

Each NetCDF file is expected to contain:

```text
fields[time, channel, latitude, longitude]
```

The included KAI_5 files are daily. With these files, `dt: 1` means one forecast day, and `S=10` means a 10-day rollout.

Put your dataset at that location, symlink `data/kai_data_5p625` to your dataset root, or edit `train_data_path`, `valid_data_path`, `test_dataset_path`, `global_means_path`, and `global_stds_path` in `configs/gnn_5p625.yaml`.

## Build Or Rebuild The Graph

The archive includes `graphs/graph_5p625.pt`. Rebuild it only if you change the grid or want to regenerate it:

```bash
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
python scripts/train.py \
  --yaml_config configs/gnn_5p625.yaml \
  --config raw_5p625 \
  --run_num full150
```

Default daily rollout schedule:

```yaml
max_rollout_steps: 10
rollout_schedule: [1, 2, 4, 6, 8, 10]
rollout_stage_epochs: [5, 5, 10, 10, 10, 10]
```

After the listed stages finish, training remains at `S=10`.

Default rollout-stage LR schedule:

```yaml
lr_schedule_type: rollout_stage
lr_by_rollout:
  1: 1.0e-4
  2: 1.0e-4
  4: 7.0e-5
  6: 6.0e-5
  8: 5.0e-5
  10: 3.0e-5
warmup_epochs: 1
```

The training loop writes:

```text
best_ckpt.tar       # global best using checkpoint_metric, usually valid_S10
best_ckpt_S1.tar    # best checkpoint during the S=1 training stage
best_ckpt_S2.tar
best_ckpt_S4.tar
best_ckpt_S6.tar
best_ckpt_S8.tar
best_ckpt_S10.tar
last_ckpt.tar
ckpt.tar
```

## Validation Behavior

Main validation loss is controlled by:

```yaml
validate_with_train_rollout: true
valid_rollout_steps: 1
eval_rollout_steps: [1, 2, 4, 6, 8, 10]
checkpoint_metric: valid_S10
stage_checkpoint_metric_mode: stage_horizon
```

When `validate_with_train_rollout: true`, the main validation loss uses the same rollout length as training for that epoch. Multi-horizon validation still logs every horizon listed in `eval_rollout_steps`.

When `validate_with_train_rollout: false`, the main validation loss uses `valid_rollout_steps`, while multi-horizon validation remains optional.

`valid_S10` means the average validation loss over a 10-step autoregressive rollout. `valid_S10_final` means the validation loss at only the final 10th lead step. `best_ckpt.tar` is selected using `checkpoint_metric`, normally `valid_S10`. `best_ckpt_S4.tar` means the best checkpoint saved during epochs where the training rollout stage was S=4.

For stage-specific checkpoints, `stage_checkpoint_metric_mode: stage_horizon` selects `best_ckpt_S4.tar` using `valid_S4`. `stage_checkpoint_metric_mode: final_horizon` selects every stage checkpoint using the final horizon metric, normally `valid_S10`.

## Per-Variable RMSE And ACC Evaluation

After training, run the standalone evaluator with the same config that was used for the checkpoint. Evaluation uses a fixed autoregressive rollout horizon. For final reports, evaluate one checkpoint for 10 days:

```bash
python scripts/evaluate.py \
  --yaml_config configs/gnn_5p625.yaml \
  --config raw_5p625 \
  --checkpoint experiments/raw_5p625_full150/best_ckpt.tar \
  --output_dir experiments/raw_5p625_full150/evaluation \
  --eval_fixed_rollout_steps 10 \
  --n_initial_conditions 5 \
  --start_timestep 1 \
  --ic_stride 1 \
  --device cuda \
  --plot_variables t2m,z500,msl
```

This produces one model curve per requested variable, plus the persistence baseline if `plot_persistence: true`:

```text
evaluation/
  S10/
    evaluation_metrics.csv
    rollout_rmse.csv
    rollout_acc.csv
  fixed10_global_best_metrics.json
  plots/
    fixed10_global_best_t2m_rmse_acc.png
    fixed10_global_best_z500_rmse_acc.png
    fixed10_global_best_msl_rmse_acc.png
```

Stage comparison is separate. It loads each stage-specific checkpoint and evaluates every checkpoint with the same fixed horizon:

```bash
python scripts/evaluate.py \
  --yaml_config configs/gnn_5p625.yaml \
  --config raw_5p625 \
  --experiment_dir experiments/raw_5p625_full150 \
  --compare_stage_checkpoints \
  --eval_fixed_rollout_steps 10 \
  --plot_variables t2m,z500,msl \
  --device cuda
```

The plot labels mean training stage, not evaluation horizon:

```text
trained S=1
trained S=2
trained S=4
trained S=6
trained S=8
trained S=10
global best
persistence
```

The stage comparison writes:

```text
evaluation/
  fixed10_stage_comparison_metrics.json
  trained_S1/
    evaluation_metrics.csv
    rollout_rmse.csv
    rollout_acc.csv
  trained_S10/
    evaluation_metrics.csv
    rollout_rmse.csv
    rollout_acc.csv
  plots/
    fixed10_stage_comparison_t2m_rmse_acc.png
```

Stage comparison is for debugging and ablation. It should not compare S=1 evaluated for one day against S=10 evaluated for ten days. All stage checkpoints are evaluated with the same `--eval_fixed_rollout_steps` value.

`evaluation_metrics.csv` has one row per lead time and variable:

```text
rollout_steps,lead_time,variable_idx,original_channel_idx,variable_name,rmse,acc
```

`rollout_rmse.csv` and `rollout_acc.csv` have one column per variable and one row per lead time. To make rollout plots for a specific variable, pass `--plot_variables`. The variable can be given as:

- variable name, such as `t2m` or `z500`
- local output `variable_idx`, such as `60`
- original channel index, if it is not ambiguous
- `all`, to plot every output variable

Normal one-checkpoint evaluation for one variable:

```bash
python scripts/evaluate.py \
  --yaml_config configs/gnn_5p625.yaml \
  --config raw_5p625 \
  --checkpoint experiments/raw_5p625_full150/best_ckpt.tar \
  --output_dir experiments/raw_5p625_full150/evaluation \
  --eval_fixed_rollout_steps 10 \
  --plot_variables t2m
```

Normal one-checkpoint evaluation for multiple variables:

```bash
python scripts/evaluate.py \
  --yaml_config configs/gnn_5p625.yaml \
  --config raw_5p625 \
  --checkpoint experiments/raw_5p625_full150/best_ckpt.tar \
  --output_dir experiments/raw_5p625_full150/evaluation \
  --eval_fixed_rollout_steps 10 \
  --plot_variables z500,t2m,msl
```

Plots are written to:

```text
<output_dir>/plots/
  fixed10_global_best_z500_rmse_acc.png
  fixed10_global_best_t2m_rmse_acc.png
```

Each plot has two panels: RMSE vs lead time and ACC vs lead time. In normal mode the model appears as one curve. In stage-comparison mode each curve is a different checkpoint trained at a different rollout stage, evaluated with the same fixed horizon.

ACC requires a daily climatology. By default, if `climatology_path` is not set and `compute_climatology: true`, the evaluator computes climatology from `train_data_path` and caches it as:

```text
<output_dir>/daily_climatology.npz
```

To reuse an existing climatology file:

```bash
python scripts/evaluate.py \
  --yaml_config configs/gnn_5p625.yaml \
  --config raw_5p625 \
  --checkpoint experiments/raw_5p625_full150/best_ckpt.tar \
  --output_dir experiments/raw_5p625_full150/evaluation \
  --eval_fixed_rollout_steps 10 \
  --climatology_path experiments/raw_5p625_full150/evaluation/daily_climatology.npz \
  --no_compute_climatology
```

If evaluating a smoke checkpoint trained with the small debug architecture, use a matching smoke/eval config such as `eval_smoke_5p625`; checkpoint architecture settings must match the config.

## Qualitative Rollout Map Visualization

Use `scripts/visualize_rollout_maps.py` to inspect ground truth, prediction, and bias maps for selected variables and lead times. The script uses Cartopy, so coastlines and country borders are drawn on every panel.

```bash
python scripts/visualize_rollout_maps.py \
  --checkpoint experiments/raw_5p625_full150/best_ckpt.tar \
  --config configs/gnn_5p625.yaml \
  --config_name raw_5p625 \
  --split valid \
  --aggregate_mode sample \
  --sample_index 0 \
  --rollout_steps 10 \
  --lead_times 1 3 5 10 \
  --variables z500 t2m msl t850 \
  --output_dir experiments/raw_5p625_full150/visualizations
```

For a stage-specific checkpoint, point `--checkpoint` at that file:

```bash
python scripts/visualize_rollout_maps.py \
  --checkpoint experiments/raw_5p625_full150/best_ckpt_S6.tar \
  --config configs/gnn_5p625.yaml \
  --config_name raw_5p625 \
  --split valid \
  --aggregate_mode sample \
  --sample_index 0 \
  --rollout_steps 10 \
  --lead_times 1 3 5 10 \
  --variables msl t2m z500 u500 \
  --output_dir experiments/raw_5p625_full150/visualizations_S6
```

Print the variable mapping and exit:

```bash
python scripts/visualize_rollout_maps.py \
  --config configs/gnn_5p625.yaml \
  --config_name raw_5p625 \
  --split valid \
  --rollout_steps 10 \
  --variables z500 t2m t850 msl u500 \
  --print_variable_mapping
```

Make whole-year mean maps instead of a single sample:

```bash
python scripts/visualize_rollout_maps.py \
  --checkpoint experiments/raw_5p625_full150/best_ckpt.tar \
  --config configs/gnn_5p625.yaml \
  --config_name raw_5p625 \
  --split valid \
  --aggregate_mode year_mean \
  --rollout_steps 10 \
  --lead_times 1 3 5 10 \
  --variables z500 t2m t850 msl \
  --output_dir experiments/raw_5p625_full150/visualizations_year_mean
```

For a quick year-mean smoke test, add `--max_batches 2`. The script uses batch size 1 for this visualization path.

Each variable is saved as one figure with rows for lead times and columns:

```text
GT | Prediction | Bias = Prediction - Ground Truth
```

Internally each row uses a fixed five-column GridSpec layout:

```text
GT map | Prediction map | shared GT/Pred colorbar | Bias map | Bias colorbar
```

GT and prediction share the same color scale. Bias is always `prediction - ground_truth` and uses a symmetric diverging colorbar centered on zero. Use `--robust_percentile 99` to control robust color limits and `--same_scale_across_leads` to keep one GT/pred scale across all selected leads for each variable.

Example output:

```text
visualizations/
  sample000_z500_rollout_maps.png
  sample000_t2m_rollout_maps.png
  sample000_msl_rollout_maps.png
  sample000_t850_rollout_maps.png
  variable_channel_mapping.json
  visualization_metadata.json
  visualize_rollout_maps.log
```

For `year_mean`, filenames use the `yearmean_` prefix, such as `yearmean_z500_rollout_maps.png`.

Known variable names include `z500`, `t2m`, `msl`, `t850`, and `u500`; aliases such as `2m_temperature`, `temperature_850`, and `mean_sea_level_pressure` are also supported. The resolver uses metadata from config/NetCDF/checkpoint-side JSON first. Hardcoded fallback channel indices are only used if you pass `--allow_hardcoded_variable_fallback`. By default, maps are denormalized back to physical units using the configured global mean/std files. Use `--no_denormalize` to plot normalized values, `--convert_z_to_height` to convert geopotential z variables from `m^2 s^-2` to meters, and `--save_arrays` to save per-lead `gt`, `pred`, and `bias` arrays as `.npy`.

## Outputs

Training writes to:

```text
experiments/<config>_<run_num>/
  out.log
  ckpt.tar
  best_ckpt.tar
  best_ckpt_S1.tar
  best_ckpt_S2.tar
  best_ckpt_S4.tar
  best_ckpt_S6.tar
  best_ckpt_S8.tar
  best_ckpt_S10.tar
  last_ckpt.tar
```

The zip archive excludes `*.tar` checkpoint files but keeps `out.log` files.
