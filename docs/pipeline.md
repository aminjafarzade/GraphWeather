# Pipeline (train → eval → plot → maps → diagnostics)

## Environment

Python 3.10+. Install the package (editable) with dev tools:

```bash
pip install -e '.[dev]'
```

For GPU training, install the torch build that matches your CUDA driver first.
On the sm_120 box, use the pinned `graphweather-cu128` interpreter — **do not
assume bare `python`** (it silently misbehaves under the wrong env;
`CONTRIBUTING.md §6.2`). The dashboard has separate deps
(`dashboard/requirements.txt`).

## The golden path: `run_full_pipeline.sh`

`scripts/run_full_pipeline.sh` runs an end-to-end pipeline for one model,
overlaid against comparison models. It is env-var driven with per-stage toggles:

```
1. TRAIN        scripts/train.py                    -> runs/<name>/best_ckpt.tar
2. EVAL         scripts/evaluate.py                 -> RMSE/ACC, 10-day rollout on test
3. PLOT         scripts/plot_experiment_rmse_acc.py -> overlay vs comparison models (+ KAI)
4. QUALITATIVE  scripts/visualize_rollout_maps.py   -> pred/gt/bias maps
5. DIAGNOSTICS  scripts/run_diagnostics.py          -> oversmoothing/attention/spectra
```

Minimal invocation:

```bash
CONFIG=configs/experiments/<id>.yaml \
CONFIG_NAME=<id> \
bash scripts/run_full_pipeline.sh
```

Common controls:

| Env var | Meaning |
|---|---|
| `RUN_TRAIN` / `RUN_EVAL` / `RUN_PLOT` / `RUN_QUALITATIVE` / `RUN_DIAGNOSTICS` | per-stage on/off (default 1) |
| `RESOLUTION_MODE` | `2p5` (default) / `1p5` / `5p625` |
| `COMPARISON_EXPERIMENTS` / `COMPARISON_LABELS` | overlay runs + labels |
| `RUNS_DIR` | results root (default `runs/`; `experiments` reproduces the legacy layout) |
| `CUDA_VISIBLE_DEVICES` | training GPU |
| `DRY_RUN=1` | print every command, execute nothing |
| `KAI_CSV` | external KAI baseline CSV for the plot stage |

New runs land in `runs/<CONFIG_NAME>/`. The pipeline writes a `run.json`
manifest (status / resolution / config / seed / horizon / git SHA / timestamps)
at the end of a successful run — the machine-readable definition-of-done
(`CONTRIBUTING.md §8`). Two-phase warm-start ("initckpt") runs train an S1 base,
then a curriculum phase initialized from that checkpoint.

## Training details

Daily rollout curriculum (defaults):

```yaml
max_rollout_steps: 10
rollout_schedule: [1, 2, 4, 6, 8, 10]
rollout_stage_epochs: [5, 5, 10, 10, 10, 10]
```

After the listed stages finish, training stays at `S=10`.

LR schedules (`src/lr_schedulers.py`):

- `rollout_stage` — LR steps down as the rollout stage grows (`lr_by_rollout`).
- `warmup_cosine` — global warmup then cosine decay.
- `rollout_stage_warmup_cosine` — restarts warmup+cosine **inside each rollout
  curriculum stage** (the current recipe; per project findings, two-phase LR
  restarts beat a single shared cosine).

Checkpoints written by the loop: `best_ckpt.tar` (global best on
`checkpoint_metric`, usually `valid_S10`), per-stage `best_ckpt_S{1,2,4,6,8,10}.tar`,
`last_ckpt.tar`, `ckpt.tar`, plus `lr_schedule.csv`/`.png`.

Seeds: `--seed` (default 777) seeds torch/cuda/numpy/random/PYTHONHASHSEED;
evaluation uses `bootstrap_seed 42`.

## Building graphs

Graphs auto-build at runtime, or explicitly:

```bash
python scripts/build_graph.py --config <cfg> --resolution_mode 2p5 --force_rebuild
```

Rebuild only when grid, coordinates, or graph settings change.
