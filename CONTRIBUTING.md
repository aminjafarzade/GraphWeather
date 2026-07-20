# Contributing to GraphWeather

> The standing rules that keep this repo clean. Rules are stated as imperatives
> with a one-line rationale. Where a rule reflects a hard external constraint (the
> dashboard `gw-run/1` contract), it is marked **[contract]**. New code, configs,
> runs, and docs are expected to follow these; see `docs/` for the reference guides
> they refer to.

---

## 1. Naming

**1.1 Resolution tag — one canonical form in identifiers: `Np5`.**
Use `5p625`, `2p5`, `1p5` in every filename, config key, run dir, variable, and
code string. The dotted form (`2.5`, `1.5`) is **only** allowed in human-facing
plot labels/titles, always with the degree sign (`2.5°`). *Rationale: today
`kai_2.5.csv` and `"2.5°"` break the `Np5` convention used everywhere else.*

**1.2 Identifier tokens are fixed and ordered.** A model/run/config identifier is
built from these tokens, in this order, `_`-joined, lowercase:
```
<res>_<levels>_h<hidden>_<graph>_<phase>[_<tag>...]
res    : 5p625 | 2p5 | 1p5
levels : l3 | l4
hidden : h128 | h160          (always the h-prefixed form)
graph  : densel3k24 | densel4k24 | rowaware   (NO inner underscores)
phase  : s1x100 | s1x200 | currS2toS10x3 | scratch
tag    : initckpt | flatlr5e7 | tisrfix | orogtisr | overfitprobe | spectral | ...
```
*Rationale: today the leading token varies (res vs phase vs graph), and
`hidden128` is sometimes dropped.*

**1.3 One name per artifact.** A config's `config_name` **equals its file stem**
(drop the multi-key YAML + per-tool dispatch tables). *Rationale: today
`config_name ≠ filename` forces a hand-maintained map that has drifted across 7
tools (train 25 branches, evaluate 22, visualize 19, …).*

**1.4 No whitespace, no newlines, no bare integers in directory names.** Eval
output dirs are descriptive (`evaluation_test_weekly52`), never `0/1/2`.

---

## 2. Where each kind of file lives (the directory map)

| Put… | …here | Tracked? |
|---|---|---|
| Library code | `src/` (import as a package) | ✅ |
| CLIs + the launcher + thin wrappers | `scripts/` | ✅ |
| One-off dev/debug/profile tools | `scripts/dev/` | ✅ |
| Source configs | `configs/{base,2p5,1p5,experiments}/` | ✅ |
| Tests | `tests/` | ✅ |
| Dashboard | `dashboard/` (never imports `src/`) | ✅ |
| Prose docs | `docs/` (+ `docs/experiments/` for specs) | ✅ |
| **Baseline inputs** (KAI CSVs) | `data/baselines/kai_<res>.csv` | ✅ **tracked** |
| Normalization stats / climatology | `data/stats/` | ❌ (regenerable) |
| Auto-built graphs | `graphs/` | ❌ |
| **All experiment outputs** | `runs/<run-id>/` | ❌ (except `.gitkeep`) |
| Comparison figures | `comparisons/` | ❌ |
| Run/tool logs | `logs/` | ❌ |
| Legacy tree, release zips, wandb | `archive/` or off-repo | ❌ |

**Rule:** code, configs, docs, tests, and *baseline inputs* are tracked; everything
generated or large is ignored. Nothing generated lives at the repo root.

---

## 3. Config layering (base + override, no more flat copies)

**3.1 Three layers, merged deepest-last:**
```
configs/base/model.yaml         # arch defaults (levels, k, blocks)
configs/base/training.yaml      # optimizer, schedule, curriculum defaults
configs/base/paths.yaml         # data/graph/exp roots via ${GW_DATA_ROOT} etc.
configs/<res>/<res>.yaml        # resolution overrides (grid, graph, clim, kai csv)
configs/experiments/<id>.yaml   # ONLY the deltas for this experiment
```
An experiment config declares `base: [../base/model.yaml, ../base/training.yaml,
../2p5/2p5.yaml]` and then a handful of overridden keys. *Requires a file-merge
step in `src/config.py` — small, golden-covered.* (This layering is planned; today
configs are still flat — see `docs/data.md` and the fix plan.)

**3.2 No absolute paths in any config.** Data/graph/exp roots come from
`configs/base/paths.yaml` using environment interpolation
(`train_data_path: ${GW_DATA_ROOT}/era5_67/train`). A `.env.example` documents the
variables. *Rationale: today 59/59 configs hardcode `/lustre/…`, 10 of them another
user's home.*

**3.3 Source configs contain no runtime metadata.** Never commit `name`,
`experiment_dir`, `checkpoint_path`, or other resolved-run fields into a source
config — those belong only in the generated `runs/<id>/config_resolved.yaml`.

**3.4 One experiment = one config file = one `config_name` (= file stem).** No
multi-`config_name` YAMLs.

---

## 4. Run naming & results storage

**4.1 One results root: `runs/`.** New runs land in `runs/<run-id>/` where
`<run-id>` follows the §1.2 grammar. The legacy `experiments/` tree is frozen under
`archive/` (read-only, ignored).

**4.2 [contract] A run directory is defined by `config_resolved.yaml` at its root.**
The dashboard scanner treats any dir containing `config_resolved.yaml` as a run.
Keep that file at the run-dir root. *Never* place stray `config_resolved.yaml` /
`model_summary.txt` at `runs/` root.

**4.3 [contract] Evaluations are subdirs named `evaluation*`** containing
`fixed{N}_global_best_metrics.json` (N = horizon, per-run — never assume 10). All
`evaluation*` dirs are first-class and selectable.

**4.4 Canonical run contents:** `config_resolved.yaml`, `model_summary.txt`,
`out.log`, `best_ckpt.tar` (+ per-stage `best_ckpt_S{N}.tar`), `evaluation*/`,
`diagnostics_full_eval/`, `visualizations_test_biasmaps/`,
`visualizations_dashboard/`, and a `run.json` (§7). Logs go in the run's
`logs/` or the top-level `logs/` — never loose in `runs/`.

**4.5 No self-nested dirs.** `runs/<id>/<id>/` must never be created (fix the
`exp_dir`+`experiment_name` join).

---

## 5. Git-ignore policy

Keep the current `.gitignore`. Standing policy: **ignore all of** `runs/*`,
`experiments/`, `archive/`, `comparisons/*`, `wandb/`, `data/` (except
`!data/baselines/`), `graphs/`, `logs/`, `runtime_diagnosis/`, `/diagnostics/`
(top-level output only — keep `src/diagnostics/`), `*.zip`, `*.tar`, `*.out`,
`tests/config_resolved_goldens/`, `dashboard/data/`, `__pycache__/`. An un-ignore
keeps tracked baselines: `!data/baselines/` + `!data/baselines/*.csv`. Never commit
checkpoints, `.npy`, `.png`, `.nc`, or release zips.

---

## 6. Dependencies & environment

**6.1 `pyproject.toml` is the source of truth.** All runtime deps (incl.
**`wandb`**) with sensible upper bounds; a `[project.optional-dependencies] dev`
extra for `pytest`. `pip install -e '.[dev]'` makes `import src` work without
`sys.path` hacks. Keep `dashboard/` deps separate (`dashboard/requirements.txt`).

**6.2 One pinned interpreter for GPU work.** Scripts default `PYTHON` to the pinned
`graphweather-cu128` interpreter; `python` bare is never assumed. *Rationale: the
sm_120 box silently misbehaves under the wrong env.*

**6.3 Seeds are always set and logged.** `--seed` (default 777) →
`set_seed(torch/cuda/numpy/random/PYTHONHASHSEED)`; eval `bootstrap_seed 42`. Record
the seed in `run.json`. *(Already correct — keep it.)*

---

## 7. How to add a new experiment (the golden path)

1. **Write one config** `configs/experiments/<id>.yaml` (`<id>` follows §1.2). No
   absolute paths, no runtime metadata.
2. **Launch** with the pipeline launcher (`scripts/run_full_pipeline.sh`):
   ```
   CONFIG=configs/experiments/<id>.yaml CONFIG_NAME=<id> \
     bash scripts/run_full_pipeline.sh
   ```
   (Stages are individually toggleable via `RUN_TRAIN`/`RUN_EVAL`/`RUN_PLOT`/
   `RUN_QUALITATIVE`/`RUN_DIAGNOSTICS`; see `docs/pipeline.md`.)
3. **Result** appears at `runs/<id>/` and is picked up by the dashboard
   automatically (it contains `config_resolved.yaml`).
4. **Compare** via `COMPARISON_EXPERIMENTS`/`COMPARISON_LABELS`; the KAI baseline is
   read from the configured `KAI_CSV`.

---

## 8. Definition of Done for a run

A run is **done** when `runs/<id>/` contains:
- [ ] `config_resolved.yaml` + `model_summary.txt` at root **[contract]**
- [ ] `best_ckpt.tar` and `out.log` ending in `DONE rank 0`
- [ ] at least one `evaluation*/fixed{N}_global_best_metrics.json` **[contract]**
- [ ] `run.json` = `{status, resolution, config_name, seed, horizon, git_sha,
      started_at, finished_at}` (makes status explicit instead of grepping
      `out.log`)
- [ ] no stray files at `runs/` root, no `runs/<id>/<id>/` nesting

A run is **reproducible** when its config uses only `${env}`/relative paths and its
`run.json` records the git SHA and seed.

---

## 9. Tests & CI (definition of "safe to merge")

- **Fast lane (every push, no GPU):** the 6 `test_dashboard_*` + `test_config_*` +
  `test_resolution*` + `test_zone_rmse_dominance` (12 tests, no torch). This is the
  **dashboard-contract smoke** + config/resolution invariants.
- **Full lane (opt-in / self-hosted GPU):** the remaining 13 torch tests.
- A change is safe to merge when the fast lane is green; structural PRs must
  additionally keep the full 24 green locally.
- Golden `config_resolved` baselines regenerate on first run inside CI, then are
  compared within the same run (they are machine-specific — never committed).

---

## 10. Invariants that must never regress (the dashboard contract)

1. A run = a dir with `config_resolved.yaml`.
2. Metrics are read **verbatim** from `evaluation*/fixed{N}_global_best_metrics.json`;
   nothing recomputes them; the dashboard writes nothing into `runs/`.
3. Horizon `N` is per-run data — never hardcode 10.
4. `dashboard/` imports nothing from `src/` (no torch).

Any restructure that would touch these ships the matching `dashboard/` change
(scanner path or `dashboard/docs/02-DATA-CONTRACT.md`) **in the same commit**.
