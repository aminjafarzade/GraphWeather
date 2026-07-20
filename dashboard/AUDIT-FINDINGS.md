# gw-dashboard audit findings (docs/05-AUDIT.md)

Audit date: 2026-07-15. Method: source inspection + greps + adversarial
fixtures. Every finding below carries its fix and the test that proves it.
Full battery: `tests/test_dashboard_audit.py` (14 tests) on top of the existing
`tests/test_dashboard_{ingest,api,smoke}.py`; all green after fixes.

## Findings & fixes

### F1 — [B][**major**] `mean_leads` averaged each run's FULL axis, not the common leads
- **Evidence:** `dashboard/ranking.py` `_criterion_value` (pre-fix) computed
  `sum(ms.mean)/len(ms.mean)` over the run's own lead axis; an H=10 run's mean
  (55.0 in the fixture) was ranked against an H=7 run's mean (40.0) — not
  comparable, violating the audit rule and the spirit of I4.
- **Fix:** `rank()` now resolves every run's evaluation first, computes the
  common lead intersection (including externals when `include_external`), and
  `mean_leads` averages stored values **only at those leads**; the criterion
  echo carries `leads_used`.
- **Test:** `test_mean_leads_averages_common_leads_only` — H=10 vs H=7 both
  yield exactly 40.0 (mean over leads 1..7), `leads_used == [1..7]`, plus the
  `horizon_truncated` warning.

### F2 — [C][major] persistence requested but absent → silently omitted
- **Evidence:** docs/03 risk list requires "baselines omitted **+ warning**";
  `overlay()` omitted the baseline with no signal when the eval lacked a
  persistence block.
- **Fix:** `overlay()` appends a `baseline_missing` warning when persistence is
  requested, runs are selected, and no stored persistence exists.
- **Test:** `test_missing_persistence_warns_never_fabricates` — empty
  `baselines`, warning present, nothing fabricated.

### F3 — [D][minor] artifact ETag did not honor conditional requests
- **Evidence:** `GET /api/artifacts/...` always returned 200 with a body;
  `If-None-Match` was ignored (audit D: "honor conditional requests").
- **Fix:** the handler compares `If-None-Match` against the mtime+size ETag and
  returns `304 Not Modified` on match.
- **Test:** `test_artifact_conditional_request_304`.

### F4 — [E][minor] header problem badge summed counts client-side
- **Evidence:** `main.js` used `state.runs.reduce(...)` to total problem counts —
  UI chrome rather than a metric, but the strictest reading of I1 says the
  backend produces every number shown.
- **Fix:** `/api/meta` now returns backend-computed `problem_totals`; the badge
  renders it verbatim.
- **Test:** `test_meta_reports_problem_totals` (totals equal the length of the
  problems feed).

### F5 — [A][minor] coverage gaps (no code defect)
Validation rules that existed but had no adversarial fixture:
`len(lead_times) != eval_fixed_rollout_steps` (error), CI array shorter than
`mean` (warning; CI dropped, mean kept). Added
`test_lead_times_length_mismatch_is_axis_error` and
`test_ci_length_mismatch_is_warning_ci_dropped`. Also added: hand-computed
`delta_vs_best_pct` (+10% fixture), CI-overlap true/false fixtures for
`within_ci_of_runner_up`, `acc_crossing` no-cross sentinel, determinism with
stable ties, empty runs root, and Q1 selection of a specific secondary
`eval_id` through the API.

## Checks that passed as built (evidence)

- **A / I2 verbatim values:** canary `z500.rmse.mean[-1] ≈ 683.826`, horizon 10,
  6 variables, persistence, epoch 27 (`test_initckpt_canary`); the smoke test
  asserts overlay returns byte-identical arrays to the record.
- **A: summary-CSV headers never parsed** — grep source scan is itself a test
  (`test_summary_csv_headers_never_parsed`): zero references to
  `rmse_day10`/`acc_day10`/`avg_1_10` in `dashboard/*.py`.
- **A: every docs/02 rule fires** with the exact reason
  (`missing_file`, `parse_error`, `axis_mismatch`, `missing_key`,
  `schema_drift` warning + continue, NaN → null + warning, never interpolated)
  — `tests/test_dashboard_ingest.py`.
- **A: status logic + Q5** — DONE marker drives `trained`; fresh checkpoints
  without DONE → `in_progress` (+ `stale_partial` warning when stale); no
  training record is ever ingested.
- **B: best-per-variable** argmin/argmax over stored values only; ranking
  deterministic (stable tie-break = criterion order).
- **C: grace window** — mid-write JSON withheld for 2 sweeps as a warning
  before surfacing invalid; recovery resets counters
  (`test_parse_error_grace_window`).
- **C: fingerprint cache** — unchanged run not re-parsed; touched JSON →
  re-parse + `run_updated`; removal → tombstone + `run_removed`.
- **C: concurrency** — records are built fully, then swapped into the store
  under a lock; a reader sees the old or the new record, never a partial one.
  (Accepted minor: `last_scanned_at` on an unchanged record is refreshed
  in-place; single-field string mutation, harmless under the GIL.)
- **D: read-only service** — grep: no write-mode opens of run paths in service
  modules (the only `json.dump` writes to stdout in the CLI); no `src/` or
  torch imports; the service never invokes eval/visualization scripts.
  (`dashboard/generate_maps.py` is the documented OFFLINE exception per
  docs/00 Q6 — it is never imported or called by the service.)
- **D: traversal-proof artifacts** — encoded `..%2F`, `%2e%2e`, and absolute
  escapes rejected (`test_artifact_passthrough_and_traversal`).
- **D: no hardcoded horizon** — grep: no literal-10 lead logic; axes come from
  JSON `lead_times`; an H=7 fixture flows through ingest → overlay → ranking
  (`test_horizon_seven_never_assumes_ten`, intersection tests).
- **D/Q1: specific `eval_id` selectable** in overlay/ranking/best
  (`test_specific_eval_id_is_selectable`).
- **D/Q3: external mismatch warns, never blocks**
  (`test_overlay_external_resolution_mismatch`, `test_ranking_external_row`).
- **E: frontend computes nothing** — grep of `main.js`: remaining arithmetic is
  option-list construction (`Math.max` over API-provided horizons), display
  ordering (`.sort()` of variable/day name lists), and color hex math; ranks,
  deltas, means, best badges, and winner emphasis all arrive from the API.
- **E: Problems verbatim; awaiting-eval state; eval selector; warning banners**
  — implemented in `main.js` (`problemHtml`, awaiting branch in `renderDetail`,
  `detEval`/`cmpEval` selectors, `cmpWarnings` banners).

## Definition of done — status

- Four invariants: backed by tests (canary, verbatim-overlay, Problem battery,
  H≠10 flows). ✅
- Every A–E item: pass or fixed-with-test (F1–F5). ✅
- Full suite: 52 tests green (`test_dashboard_ingest` 20, `_api` 15, `_smoke` 3,
  `_audit` 14). ✅
- No file under `src/`, `scripts/`, `configs/`, `runs/` modified by the audit;
  all changes inside `dashboard/` + `tests/`. ✅
