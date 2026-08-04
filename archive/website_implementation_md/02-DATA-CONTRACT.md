# 02 — Data contract (`gw-run/1`)

A `RunRecord` is assembled **read-only** from the artifacts. Every field is
either present-from-artifact or `null` accompanied by a `Problem` entry.
**Nothing is invented.** Types use a JSON-schema-ish shorthand; `[f × H]` means
"array of floats of length H (the horizon)".

Bump `schema_version` when the shape changes; older on-disk JSON that predates a
key is tolerated with a `schema_drift` warning, not a failure.

## RunRecord

```
RunRecord {
  schema_version: "gw-run/1"
  run_id: str            # directory basename under runs/ (unique by construction)
  path: str              # absolute run dir
  status: "in_progress" | "trained" | "evaluated" | "invalid" | "removed"
      # in_progress: no "DONE rank 0" in out.log AND last_ckpt.tar mtime fresh
      # trained:     DONE present, no parseable eval
      # evaluated:   >= 1 valid EvalRecord
  discovered_at, last_scanned_at: iso8601
  fingerprint: { config_mtime_ns, out_log_size, eval_json_mtime_ns, ... }  # cache key

  architecture: Architecture
  evaluations: [EvalRecord, ...]   # one per evaluation* dir; all first-class & selectable
  diagnostics: DiagRecord | null
  qualitative: QualRecord | null
  problems:    [Problem, ...]
}
```

## Architecture (from `config_resolved.yaml` + `model_summary.txt`)

```
Architecture {
  resolution_mode: str            # "2p5" | "1p5" | ...
  grid_shape: [int, int]
  level_shapes: [[int, int], ...]
  node_counts: [int, ...] | null
  edge_counts: [int, ...] | null
  hidden_dim: int
  num_heads: int
  level_k_neighbors: [int, ...]
  graph_connectivity_strategy: str
  blocks: { encoder, decoder, l0, l1, l2, l3, l0_refine, l1_refine, l2_refine: int }
  params_millions: float | null   # from "trainable_parameters"
  lr_schedule: { type: str, lr: float, min_lr: float, warmup_epochs: int }
  rollout: { schedule: [int, ...], stage_epochs: [int, ...], max_epochs: int }
  forcings: { known_future_variables: [str, ...], copy_variables: [str, ...] }
  init_from_checkpoint: str | null
  tags: [str, ...]
}
```

## EvalRecord (from `fixed{N}_global_best_metrics.json`)

Every `evaluation*` dir produces its own `EvalRecord` and is independently
selectable (Q1). `is_primary` marks the priority-list match; secondary
evaluations (e.g. `evaluation_test_weekly52_S5ckpt`) are equal first-class
citizens the UI can select and overlay.

```
EvalRecord {
  eval_id: str          # eval dir name, e.g. "evaluation_test_weekly52"
  is_primary: bool      # priority order (see ground-truth doc)
  horizon: int          # eval_fixed_rollout_steps
  lead_times: [int, ...]      # JSON "lead_times"; MUST have len == horizon
  selection: { split, n_initial_conditions, stride, first_start_time, last_start_time }
  climatology_path: str
  bootstrap: { samples: int, confidence_level: float, seed: int }
  checkpoint: { label, epoch, train_rollout_steps, params_m }   # checkpoints.global_best
  variables: [str, ...]       # keys of checkpoints.global_best.metrics
  series: {                   # VERBATIM arrays from the JSON — never rescaled
    <var>: {
      channel: int, variable_idx: int,
      rmse: { mean: [f × H], ci_lower: [f × H] | null, ci_upper: [f × H] | null },
      acc:  { mean: [f × H], ci_lower: [f × H] | null, ci_upper: [f × H] | null }
    }
  }
  baselines: {
    persistence: { <var>: { rmse: {...}, acc: {...} } } | null   # checkpoints.persistence
  }
  units: { <var>: str }       # DECLARED static table, not from artifacts
  artifacts: { summary_csv, summary_txt, plots: {<var>: relpath}, s_dir: "S{N}", log: relpath }
}
```

## DiagRecord (`diagnostics_full_eval/`; every sub-block optional)

```
DiagRecord {
  epoch: int                                   # from epoch_{NNNN}_ stem
  rollout_curve: { steps: [int, ...], losses: [f, ...] } | null    # phase=valid, max horizon
  variance_ratio: { <var>: [f × L] } | null
  power_spectrum: { <var>: { wavenumbers: [int, ...], truth: [f, ...],
                             pred_by_lead: { <lead>: [f, ...] } } } | null
  attention: relpath | null
  scalars: relpath | null
  final_json: relpath | null
}
```

## QualRecord (`visualizations_test_biasmaps/`)

```
QualRecord {
  metadata: { lead_times, grid_shape, aggregate_mode, bias_definition,
              denormalized, ... }             # visualization_metadata.json verbatim
  maps: { <var>: relpath.png }
  channel_mapping: relpath | null
}
```

## Training curves — intentionally not ingested

By decision (Q5 → "awaiting evaluation only"), the dashboard does **not** ingest
or plot training-time curves. A run that has finished training but has no valid
evaluation carries no training record; it is surfaced purely by `status`
(`trained` → shown as "awaiting evaluation", `in_progress` → "training in
progress"), while its `architecture` block still renders. Do not build a
training-curve view.

## Problem (the structured format-error — invariant I3)

```
Problem {
  run_id: str
  eval_id: str | null
  artifact: str        # logical name, e.g. "fixed_metrics_json"
  path: str            # absolute file path
  expected: str        # e.g. "len(series.z500.rmse.mean) == lead_times length (10)"
  found: str           # e.g. "9 values"
  reason: "missing_file" | "parse_error" | "axis_mismatch" | "missing_key" |
          "unexpected_value" | "schema_drift" | "stale_partial"
  severity: "error" | "warning"    # error => run/eval marked invalid; warning => shown, usable
  detected_at: iso8601
  schema_version: "gw-run/1"
}
```

## Validation rules on ingest

Each maps to a `Problem`. **Preferring a Problem over a fabricated value is the
whole point (I3).**

- Metrics JSON exists and parses. *(missing → `missing_file`; unparseable →
  `parse_error`.)*
- `len(lead_times) == eval_fixed_rollout_steps`. *(else `axis_mismatch`.)*
- Every `series.<var>.{rmse,acc}.mean` has length `== horizon`. *(else
  `axis_mismatch`.)*
- When present, each CI array has the same length as its `mean`. *(else
  `axis_mismatch`.)*
- `S{N}` dir name is consistent with `horizon`. *(else `unexpected_value`,
  warning.)*
- `checkpoints.global_best` present and `variables` non-empty. *(else
  `missing_key`, error.)*
- All values finite. **A NaN is a `warning`; the value is passed through as
  `null`, never interpolated.**
- Older-JSON key drift (missing `features`, `evaluation_target_override`, …) ⇒
  `schema_drift` **warning**; ingest continues on the keys that exist.
- The summary CSV's hardcoded `*_day10` / `*_1_10` headers are **ignored by
  design** — values come from the JSON only, never those headers.

An `error`-severity Problem marks the run (or that eval) `invalid`; the run is
still returned and shown, tagged invalid, with the Problem attached. A
`warning` leaves the run usable and the warning visible.

## Cross-run semantics (invariant I4)

- Overlay/ranking responses always carry each run's declared `lead_times`.
- When horizons differ, the backend serves the **intersection** of leads
  (`common = sorted(set.intersection(...))`) and attaches a
  `horizon_truncated` warning listing every run's horizon and the common max.
- A `day-k` criterion with `k` outside a run's axis flags that run
  `not_comparable` — never a guessed value.
