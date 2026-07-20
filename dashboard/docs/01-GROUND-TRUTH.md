# 01 — Ground truth about the artifacts

Everything here was verified by opening real files in the repo. **Trust it as a
starting point, but re-verify against the live repo before you depend on any
single fact** — the repo is active and files change. If the repo contradicts
this document, the repo wins: stop and report the contradiction.

Repo root: `/lustre/home/ziya/GNN/GraphWeather5p625`
Working env: `graphweather-cu128` (torch 2.11+cu128). The dashboard imports
nothing from `src/` and never needs torch.

## What counts as a run

A run is a **directory under `runs/` that contains `config_resolved.yaml`**.
The scanner must ignore, silently:

- files at the `runs/` root (e.g. a stray `runs/config_resolved.yaml`,
  `runs/model_summary.txt`),
- `wandb/` trees,
- the empty nested `runs/<name>/<name>/` directory some launchers create.

## Artifact inventory: guaranteed vs conditional

**Guaranteed once training has started** (`train.py:206-213`,
`trainer.py:3164-3187`, `:3884`, `:4044-4114`, `:1331-1373`):
`config_resolved.yaml`, `model_summary.txt` (contains
`trainable_parameters: N`), `out.log`, `epoch_logs.csv`,
`training_metrics.csv`, `s1_overfit_curve.csv`, `logs/valid_*.csv`,
`ckpt.tar`, `last_ckpt.tar`; plus `best_ckpt.tar` after the first improvement
and `best_ckpt_S{stage}.tar` per curriculum stage reached
(`trainer.py:2865-2867`).

**Guaranteed if and only if the eval stage ran** (writers in §"Where metrics
live"): the full `evaluation_test_weekly52/` set — the metrics JSON, the
`S{N}/` CSVs, the summary CSV/TXT, and the per-variable RMSE/ACC plot PNGs.

**Conditional even when a stage ran** (guarded writers — older runs lack these,
their absence is NOT an error):

- persistence block in the JSON — only with `--include_persistence`
  (`evaluator.py:3785-3792`; the pipeline always passes it).
- diagnostics extras: `{stem}_power_spectrum.csv/.npz`,
  `{stem}_variance_ratio.csv`, `spectral.npz`, `graph_structure_metrics.csv`
  (`collector.py:1154-1161`).
- the whole `visualizations_test_biasmaps/` dir — only if the qualitative stage
  ran (roughly 7 of ~11 evaluated runs have it).

## Where the metrics live (authoritative)

The one authoritative source of numbers is
`runs/<run>/<evaldir>/fixed{N}_global_best_metrics.json`
(writer `src/evaluator.py:3479-3513`, filename `:3798`). **Read metrics only
from this JSON.**

- `N` = `eval_fixed_rollout_steps` (`evaluator.py:494-496`, `:3728`;
  CLI `--fixed_rollout_steps`). The `S{N}/` dir and `fixed{N}_` prefix are
  f-strings on `N` — a 7-day eval writes `S7/` and `fixed7_*`.
- `lead_times` in the JSON = `list(range(1, N+1))` (`:3489`). **This is the
  lead axis. Never assume 10.**
- **Trap:** the summary CSV has hardcoded column names `rmse_day10` /
  `rmse_avg_1_10` even when values are horizon-correct
  (`evaluator.py:3537-3560`). **Never parse those headers.** Values come from
  the JSON only.
- `S{N}/rollout_rmse.csv` and `S{N}/rollout_acc.csv` are **model-only**
  (`_save_metrics`, `:2431-2479`); they do not contain persistence. Treat them
  as secondary artifacts.

### JSON layout (current code)

Top-level keys: `eval_fixed_rollout_steps`, `lead_times`, `selection`
(`split`, `n_initial_conditions`, `stride`, first/last start time),
`climatology`, `bootstrap` (`samples`, `confidence_level`, `seed`),
`checkpoints`, `external_baselines`, and several others
(`features`, `evaluation_target_override`, ...). **Older files miss some of
these top-level keys** — tolerate this and emit a `schema_drift` *warning*, do
not fail.

`checkpoints.global_best` holds `label`, `checkpoint_path`, `epoch`,
`train_rollout_steps`, `params_m`, `aggregate_loss`, and `metrics`.
`checkpoints.persistence` (optional) mirrors that shape.

`checkpoints.<label>.metrics` is keyed by variable name:

```
metrics: {
  <var>: {
    name, variable_idx, channel,
    rmse: { mean: [f × H], ci_lower: [f × H] | null, ci_upper: [f × H] | null },
    acc:  { mean: [f × H], ci_lower: [f × H] | null, ci_upper: [f × H] | null }
  }
}
```

CI arrays exist only when `bootstrap.samples > 0`. When present, every CI array
must be the same length as `mean`.

## How the metrics are defined (do not reimplement — for context only)

- **RMSE (default "current" backend):** latitude-weighted
  (`cos(lat)` clipped at 0), errors de-normalized to physical units, per-lead
  MSE summed over the grid, then `sqrt(nanmean(...))` across initial conditions
  (`evaluator.py:2290-2366`, `:2012-2029`).
- **RMSE ("weatherbench2" backend):** mean-1-normalized weights, spatial
  `nanmean` (masks NaNs) (`weatherbench2_metrics.py:8-61`). No run on disk
  currently uses it; reserve a schema slot, flag as untested.
- **ACC:** latitude-weighted anomaly correlation against a day-of-year
  climatology, in normalized space (`evaluator.py:366-376`, `:2380-2381`).
- **Persistence:** computed inside the evaluator by holding the initial
  condition constant across leads (`:2152-2186`, `:2390-2423`).
- **Bootstrap CIs:** paired resampling over initial conditions, seeded (default
  seed 42), percentile CIs at `confidence_level` (default 0.95)
  (`:1985-2010`).

## Lifecycle signals (no manifest/status file exists)

- **Training finished ⇔** `out.log` contains the exact line `DONE rank 0`
  (written by `scripts/train.py:221` after `trainer.train()` returns — not by
  the trainer). Absence ⇒ in-progress or crashed.
- **in_progress** = no `DONE` and `last_ckpt.tar` mtime is fresh (< ~30 min).
- **crashed** = no `DONE` and stale.
- **Eval finished ⇔** `fixed{N}_global_best_metrics.json` exists and parses
  (written near the end of eval, `:3798`). A run mid-eval shows
  `evaluation.log` growing without the JSON.

## Evaluation directories

Canonical dir is `evaluation_test_weekly52/`. Discovery priority (mirror
`scripts/plot_experiment_rmse_acc.py:36-42`):

```
["evaluation_test_weekly52", "evaluation_test_with_kai",
 "evaluation_test_allstarts", "evaluation_test", "evaluation"]
```

Additional `evaluation*` dirs (e.g. `evaluation_test_weekly52_S5ckpt`) are
ingested as secondary evaluations of the same run. Primary = first
priority-list match present.

## Baselines

- **Persistence** — per-run, inside the metrics JSON at
  `checkpoints.persistence.metrics[var].{rmse,acc}.{mean,ci_lower,ci_upper}`
  (full per-lead arrays). Not present in the `S{N}` CSVs.
- **Climatology** — used only to define ACC anomalies; **no stored
  climatology-forecast RMSE/ACC curve exists.** Do not fabricate one.
- **External reference** — `experiments/kai_2.5.csv`, columns
  `variable,timestep,rmse,acc` (6 vars × ~10 leads, **2.5° only**). Expose as
  external baseline id `kai-2p5` with resolution tag `2p5`; attach a
  `resolution_mismatch` warning when overlaid on a run of another resolution.

## Diagnostics (`diagnostics_full_eval/`, optional)

`{stem}` = `epoch_{NNNN:04d}`. Guaranteed-if-present:
`{stem}_scalars.json`, `{stem}_layer_metrics.csv`,
`{stem}_attention_metrics.csv`,
`{stem}_rollout_curve.csv` (cols `epoch,phase,horizon,step,loss` — use
`phase=valid` rows at max horizon), `final_full_diagnostics.json`, `tables/`,
`logs/`.
Conditional (older runs lack them):
`{stem}_power_spectrum.csv` (cols
`epoch,phase,lead,variable,wavenumber_bin,power_pred,power_truth,power_ratio,backend`),
`{stem}_variance_ratio.csv` (cols `epoch,phase,lead,variable,variance_ratio,backend`),
spectral `.npz`, `graph_structure_metrics.csv`.

## Qualitative (`visualizations_test_biasmaps/`, optional)

Pre-rendered `yearmean_<var>_rollout_maps.png` (leads [1,3,5,10]) plus
`visualization_metadata.json` (`lead_times`, `grid_shape`, lat/lon,
`bias_definition = "prediction_minus_ground_truth"`, per-variable units/channels)
and `variable_channel_mapping.json`. **The model is never run by the dashboard**
— `visualize_rollout_maps.py` loads a checkpoint and rolls out, so the service
must never invoke it. Map view = serve these files + metadata where present.

## Config fields for the architecture panel (`config_resolved.yaml`)

`resolution_mode`, `grid_shape`, `level_shapes`, `node_counts`, `edge_counts`,
`hidden_dim`, `num_heads`, `level_k_neighbors`, `graph_connectivity_strategy`,
`encoder_blocks`, `decoder_blocks`, `l0_blocks…l3_blocks` and the `*_refine`
counts, `rollout_schedule`, `rollout_stage_epochs`, `max_epochs`, `lr`,
`min_lr`, `lr_schedule_type`, `warmup_epochs`,
`target_handling.known_future_variables` (forcings), `init_from_checkpoint`,
`wandb.tags`. Parameter count from `model_summary.txt`
(`trainable_parameters: N`). A working extractor already exists at
`scripts/build_dashboard_data.py:60-99` — **lift its logic** rather than writing
a new one.

## Units are NOT stored in the artifacts

Declare a static lookup table and mark it `"declared"` in `/api/meta` so the UI
never implies it came from the data:

```
{ z500: "m²/s²", t850: "K", t2m: "K", msl: "Pa", q700: "kg/kg", u850: "m/s" }
```

Default variable set emitted by the pipeline: `z500, t2m, t850, msl, q700, u850`
(`evaluate.py:228-232`), but metric arrays exist for all 67 channels — read the
variable list from the JSON, do not hardcode it.

## Filesystem note

The repo lives on **Lustre**, where inotify is unreliable. Use a **polling
scanner** (default 20 s), not a filesystem watcher. See
`docs/03-ARCHITECTURE.md` for fingerprinting and caching.
