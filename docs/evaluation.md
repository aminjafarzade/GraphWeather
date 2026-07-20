# Evaluation

Evaluation is a standalone step (`scripts/evaluate.py`) run with the **same
config as the checkpoint**. It performs a true autoregressive rollout with ground
truth at every lead, computing latitude-weighted RMSE and ACC per variable, plus
a persistence baseline and bootstrap confidence intervals.

> The pipeline's canonical eval output dir is `evaluation_test_weekly52` — the
> name matters: the plotter auto-discovers by it, and the dashboard treats any
> `evaluation*` dir as first-class (`CONTRIBUTING.md §4.3`).

## Final RMSE/ACC (weekly-52 on the test split)

```bash
EXP=runs/<id>
CFG=configs/experiments/<id>.yaml
CLIM=data/stats/2p5_train_dayofyear_climatology.nc

python scripts/evaluate.py \
  --config "$CFG" --config_name <id> --resolution_mode 2p5 \
  --checkpoint "$EXP/best_ckpt.tar" \
  --split test --fixed_rollout_steps 10 \
  --selection stride --stride 7 \
  --variables z500 t2m t850 msl q700 u850 \
  --include_persistence \
  --climatology_path "$CLIM" \
  --bootstrap_samples 1000 --confidence_level 0.95 --bootstrap_seed 42 \
  --output_dir "$EXP/evaluation_test_weekly52"
```

Output layout:

```
evaluation_test_weekly52/
  S10/{evaluation_metrics.csv, rollout_rmse.csv, rollout_acc.csv}
  fixed10_global_best_metrics.json      # read verbatim by the dashboard
  plots/fixed10_global_best_<var>_rmse_acc.png
```

`evaluation_metrics.csv` has one row per lead-time × variable
(`rollout_steps,lead_time,variable_idx,original_channel_idx,variable_name,rmse,acc`).
`rollout_rmse.csv`/`rollout_acc.csv` have one column per variable, one row per lead.

## WeatherBench2-compatible backend (KAI comparison)

Use `--rmse_backend weatherbench2` to compute RMSE as
`sqrt(mean(latitude-weighted spatial MSE over initial conditions))`, matching KAI,
and attach an external baseline:

```bash
python scripts/evaluate.py ... \
  --rmse_backend weatherbench2 \
  --external_baseline_csv data/baselines/kai_2p5.csv \
  --external_baseline_label "Kai 7M" \
  --output_dir "$EXP/evaluation_test_weekly52_weatherbench2"
```

`--rmse_backend both` writes the repo's current backend and the WB2 backend from
the same rollout for a side-by-side.

## Stage comparison

Loads each stage-specific checkpoint and evaluates **all** with the same fixed
horizon (labels mean the training stage, not the eval horizon):

```bash
python scripts/evaluate.py --config "$CFG" --config_name <id> \
  --experiment_dir "$EXP" --compare_stage_checkpoints \
  --eval_fixed_rollout_steps 10 --variables t2m z500 msl --device cuda
```

This is for debugging/ablation only — never compare S=1 at one day against S=10
at ten days; all stage checkpoints use the same `--eval_fixed_rollout_steps`.

## Qualitative rollout maps

`scripts/visualize_rollout_maps.py` draws GT / prediction / bias panels (bias =
`prediction − ground_truth`, symmetric diverging scale) with Cartopy coastlines,
for selected variables and lead times, either for a single sample
(`--aggregate_mode sample`) or a whole-year mean (`--aggregate_mode year_mean`).
Useful flags: `--convert_z_to_height`, `--same_scale_across_leads`,
`--robust_percentile`, `--save_arrays`, `--print_variable_mapping`.

## Latitude-zone RMSE dominance

`scripts/evaluate_zone_rmse_dominance.py` reports which 15° latitude band
contributes most to global weighted squared error (dominance % from
latitude-weighted squared error, not RMSE). Covered by
`tests/test_zone_rmse_dominance.py` (fast lane).
