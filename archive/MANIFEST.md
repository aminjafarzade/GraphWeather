# archive/ — frozen legacy material

Created 2026-07-20 during the repo cleanup (P2/P4/P5). Everything here is retired
but kept for reference. The big data trees are git-ignored; the small retired
code/config is tracked so its history survives.

| Path | What | Tracked? | Why archived |
|---|---|---|---|
| `experiments/` | The legacy 11 GB results tree (pre-`runs/` layout, smoke outputs, comparison plots, legacy runs). | ❌ ignored | Superseded by the single `runs/` results root (P4.1). A compat symlink `../experiments` still resolves for legacy analysis scripts. |
| `zips/` | ~3.7 GB of release/experiment `.zip` bundles that were at the repo root. | ❌ ignored | Build artifacts don't belong at the source root (A1.6). |
| `scripts/` | One-off / superseded shell scripts. | ✅ tracked | Consolidated behind `scripts/run_pipeline.sh` or superseded (P2). |
| `configs/` | Retired / quarantined source configs (if any). | ✅ tracked | Removed from the active config set (P3/P5). |

## Notes
- `runs/` is now the single results root. The legacy tree is reachable via the
  `experiments` → `archive/experiments` symlink or `RUNS_DIR=archive/experiments`.
- Nothing here is imported or executed by the active pipeline, tests, or dashboard.
- To reclaim disk, `archive/experiments/` and `archive/zips/` can be deleted or
  pushed off-repo; they are regenerable / already-released artifacts.
