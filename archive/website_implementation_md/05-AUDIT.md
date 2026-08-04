# 05 — Audit & harden

## Purpose

Independently verify that the implemented **gw-dashboard** is correct and robust
— backend algorithms, server, and frontend — and **fix** what is wrong. This is
a review pass, not a rewrite. Read `CLAUDE.md` and `docs/00`–`04` first; the
four invariants (I1–I4) are the backbone of every check below.

## Ground rules for the audit

- **Verify with evidence, do not trust.** Open the actual source, run the tests,
  and hit endpoints via FastAPI `TestClient`. A passing comment is not proof;
  code behavior against real runs and adversarial fixtures is.
- **Same guardrails as the build.** Do not alter the eval algorithms; do not
  write into `runs/`, `src/`, `scripts/`, or `configs/`; import nothing from
  `src/`. Fixes are **minimal diffs inside `dashboard/`** (or its tests).
- **Fix, then re-verify.** After each fix, re-run the affected tests; a fix that
  breaks another test isn't done.
- **Ask before reversing a decision.** The decided open questions (Q1 first-class
  evaluations, Q3 warn-only, Q5 awaiting-eval only) are settled — do not "fix"
  them back to defaults.

## A. Contract & ingest correctness

- **Values verbatim (I2).** Spot-check the canary:
  `dense_l3k24_curriculum_S2toS10_3ep_initckpt` → `series.z500.rmse.mean[-1] ≈
  683.826`, horizon 10, 6 variables, persistence present, epoch 27. Any
  transformation of stored values is a bug.
- **JSON is the only metric source.** Confirm nothing parses the summary CSV's
  `rmse_day10` / `*_1_10` headers (they lie for `N ≠ 10`). Grep for those
  strings; there should be zero parsing use.
- **Validation actually fires.** For each rule in `docs/02`, drive a fixture and
  confirm the exact `Problem.reason`: missing JSON → `missing_file`; unparseable
  → `parse_error`; `len(lead_times) != N` and short `mean` arrays →
  `axis_mismatch`; CI length ≠ `mean` length → `axis_mismatch`; missing
  `checkpoints.global_best` → `missing_key`; old JSON missing top-level keys →
  `schema_drift` **warning** (ingest continues); NaN → **warning**, value passed
  through as `null`, **never interpolated**.
- **No fabrication (I3).** Confirm there is no code path that fills, guesses, or
  interpolates a missing value; a gap stays a gap and produces a Problem.
- **Status logic.** `DONE rank 0` presence drives `trained`; fresh
  `last_ckpt.tar` + no DONE → `in_progress`; ≥1 valid eval → `evaluated`. A run
  without a valid eval carries **no training record** (Q5) and surfaces as
  "awaiting evaluation".
- **Q1.** A run with two `evaluation*` dirs yields two `EvalRecord`s, exactly one
  `is_primary`, both selectable.

## B. Backend algorithm correctness (the new logic in `ranking.py`)

- **best-per-variable:** argmin for RMSE, argmax for ACC, over **stored** values
  only. Feed a fixture with a known winner and assert it.
- **delta_vs_best_pct:** check the formula and sign against a hand-computed
  fixture (e.g. reference vs best); confirm the reference run shows 0 and worse
  runs show the correct signed percentage.
- **common-lead intersection (I4):** mixed-horizon fixtures (H=10 vs H=7) →
  results only on leads 1..7 + a `horizon_truncated` warning listing per-run
  horizons. `mean_leads` must average **only over the common leads**, not each
  run's full axis.
- **day-k out of range:** a `day-k` with `k` beyond a run's horizon flags that
  run `not_comparable` — never a guessed or clamped value.
- **acc_crossing:** first lead where stored ACC crosses the threshold, computed
  from stored curves, labeled `derived`. Test the no-crossing case (returns a
  sentinel / not_comparable, not a fabricated lead).
- **CI-overlap flag:** `within_ci_of_runner_up` compares stored CI bounds only;
  verify with overlapping and non-overlapping fixtures.
- **Determinism:** identical inputs → identical ordering (stable tie-break).

## C. Robustness & edge cases

Walk every risk in `docs/03` §"Risks / edge cases" and prove each with a test:

- Missing persistence block → baselines omitted + warning, not fabricated.
- Empty `runs/` → empty list, healthy `/api/meta`, UI empty-state.
- Junk entries (root files, empty nested `runs/<name>/<name>/`, `wandb/`) →
  ignored; a dir needs `config_resolved.yaml` to count as a run.
- **Partial / mid-write JSON:** a JSON caught mid-write gets one `parse_error`
  and a **2-sweep grace window** before being surfaced invalid — confirm it is
  not flagged invalid on the first sweep.
- **Scanner fingerprint:** touching a copied metrics JSON changes the
  `(mtime_ns, size)` fingerprint and triggers a re-parse + `run_updated`; an
  unchanged run is **not** re-parsed (cache hit).
- **Lustre:** confirm a **polling** scanner, not inotify/watchfiles; the sweep is
  time-boxed and reports `last_sweep`.
- Re-eval in place → fingerprint catches it, record updates, `run_updated`
  emitted (no duplicate run).
- Concurrency: a scan racing an API read must not serve a half-parsed record.

## D. Server correctness

- **Read-only (I2):** confirm the process never opens a `runs/` path for write
  and never calls the visualization/eval scripts; grep for write modes and any
  `src` import.
- **No torch import:** the service must import nothing from `src/`; starting it
  must not require a GPU.
- **Path-traversal:** `GET /api/artifacts/{id}/{relpath}` rejects `..` and
  absolute escapes; only files resolving **inside** the run dir are served. Add
  an explicit traversal test.
- **ETag / caching:** artifact responses carry an `ETag` from mtime and honor
  conditional requests.
- **`schema_version` on every response.** Grep every endpoint.
- **Horizon never assumed 10 (I4):** grep for literal `10` in axis/lead logic;
  every lead axis comes from JSON `lead_times`.
- **SSE:** `run_added` / `run_updated` / `run_removed` / `problem` fire on the
  right transitions; a client reconnect refetches cleanly.
- **`eval` selection (Q1):** overlay/ranking/best accept a specific `eval_id`,
  not only `primary`.
- **External baseline (Q3):** overlaying `kai-2p5` on a non-2.5° run attaches a
  `resolution_mismatch` warning and still returns the baseline (never blocks).

## E. Frontend correctness

- **Computes nothing (I1):** search the frontend for arithmetic on metric arrays
  (`+ - * /`, `reduce`, `Math.*`, sorting for rank). Ranks, deltas, best badges,
  and means must all come from the API. Any client-side metric math is a bug —
  move it server-side.
- **Options are API-driven:** variables, leads, baselines, and the horizon come
  from `/api/meta` + per-run `lead_times`; grep for hardcoded variable names or
  a hardcoded 10.
- **Problems verbatim:** the Problems panel renders `Problem` objects as-is
  (reason, expected, found, path); nothing is summarized away.
- **Awaiting-eval (Q5):** a run with no valid eval shows the "awaiting
  evaluation" state (architecture only, no curves, no training plots).
- **Eval selector (Q1):** runs with >1 evaluation expose a selector; switching
  re-queries the backend.
- **Warnings as banners:** `horizon_truncated` and `resolution_mismatch` render
  as visible banners, not silent.

## Method & deliverable

1. Produce `dashboard/AUDIT-FINDINGS.md`: one entry per issue with
   `{area (A–E), severity (blocker|major|minor), evidence (file:line or failing
   test), expected vs actual, proposed fix}`.
2. Add any missing **adversarial fixtures** (corrupt JSON, truncated arrays,
   mixed horizons, NaN, old-schema JSON, traversal attempt, mid-write race).
3. Apply minimal fixes inside `dashboard/`; re-run the full `tests/` suite after
   each.
4. Update `AUDIT-FINDINGS.md` with the fix and its verifying test per issue.

## Definition of done

- All four invariants demonstrably hold, backed by tests (not assertions in
  prose).
- Every A–E check either passes or has a fix + a test that proves the fix.
- The canary (`z500.rmse.mean[-1] ≈ 683.826`) and the full `tests/` suite are
  green.
- No file under `src/`, `scripts/`, `configs/`, or `runs/` was modified; new
  code stays in `dashboard/`.
