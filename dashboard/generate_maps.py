"""OFFLINE qualitative-map generator — per (variable, day) triptychs.

This is the documented exception to the read-only rule (docs/00-OVERVIEW.md,
open question Q6): it is a user-triggered, GPU-using generation step that runs
the repo's OWN qualitative stage (scripts/visualize_rollout_maps.py) with
--save_arrays, then renders one clear PNG per (variable, day) from the saved
arrays. The dashboard SERVICE never calls this; it only serves the results.

Outputs into runs/<run>/visualizations_dashboard/:
    yearmean_<var>_day<DD>_{gt,pred,bias}.npy   (written by the repo script)
    map_<var>_day<DD>.png                       (rendered here from the arrays)
    maps_index.json                             (what the dashboard ingests)

Usage:
    python -m dashboard.generate_maps --run <run_id> [--gpu 3] [--force]
    python -m dashboard.generate_maps --all-evaluated [--gpu 3]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from . import ingest
from .contract import DECLARED_UNITS, utc_now_iso
from .ranking import select_eval

REPO_ROOT = ingest.REPO_ROOT
PYTHON = sys.executable
OUT_DIR_NAME = "visualizations_dashboard"
PREFIX = "yearmean"

_CFG_FILE_RE = re.compile(r"Configuration file:\s*(\S+)")
_CFG_NAME_RE = re.compile(r"Configuration name:\s*(\S+)")


def _config_from_out_log(run_dir: Path) -> tuple[str, str]:
    """The training launch recorded its config file + section in out.log."""
    head = (run_dir / "out.log").read_text(errors="replace")[:4000]
    file_m, name_m = _CFG_FILE_RE.search(head), _CFG_NAME_RE.search(head)
    if not file_m or not name_m:
        raise RuntimeError(f"{run_dir.name}: config file/name not found in out.log header")
    cfg = file_m.group(1)
    if not Path(cfg).is_file():
        raise RuntimeError(f"{run_dir.name}: recorded config {cfg} no longer exists")
    return cfg, name_m.group(1)


def _run_visualizer(run_dir: Path, record, gpu: str | None) -> Path:
    ev = select_eval(record, "primary")
    if ev is None:
        raise RuntimeError(f"{run_dir.name}: no valid evaluation — nothing to visualize")
    cfg_file, cfg_name = _config_from_out_log(run_dir)
    out_dir = run_dir / OUT_DIR_NAME
    out_dir.mkdir(exist_ok=True)

    cmd = [
        PYTHON, str(REPO_ROOT / "scripts" / "visualize_rollout_maps.py"),
        "--config", cfg_file, "--config_name", cfg_name,
        "--resolution_mode", str(record.architecture.resolution_mode),
        "--checkpoint", str(run_dir / "best_ckpt.tar"),
        "--split", "test", "--aggregate_mode", "year_mean",
        "--rollout_steps", str(ev.horizon),
        "--lead_times", *[str(l) for l in ev.lead_times],
        "--variables", *ev.variables,
        "--output_dir", str(out_dir),
        "--save_arrays",
    ]
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[{run_dir.name}] rollout + arrays (H={ev.horizon}, "
          f"vars={','.join(ev.variables)}) …", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    return out_dir


def _render_pngs(run_dir: Path, out_dir: Path) -> dict:
    """Render map_<var>_day<DD>.png from the saved arrays. CPU only.

    Panels are drawn on a PlateCarree projection with coastlines (like the
    repo's own qualitative stage) and keep the DATA's aspect ratio — a
    180°x360° grid renders as a 1:2 panel, never stretched.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        import cartopy.crs as ccrs
        from cartopy.util import add_cyclic_point
        HAVE_CARTOPY = True
    except Exception:                    # cartopy or its data unavailable
        HAVE_CARTOPY = False

    triplets: dict = {}
    for p in sorted(out_dir.glob(f"{PREFIX}_*_day*_gt.npy")):
        m = re.match(rf"^{PREFIX}_(.+)_day(\d+)_gt\.npy$", p.name)
        if m:
            days = triplets.setdefault(m.group(1), [])
            if int(m.group(2)) not in days:
                days.append(int(m.group(2)))

    lat = lon = None
    grid_note = ""
    meta_path = out_dir / "visualization_metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(errors="replace"))
        ll = meta.get("lat_lon") or {}
        grid = meta.get("grid_shape")
        if grid:
            grid_note = f" · grid {grid[0]}×{grid[1]}"
        if ll.get("lat_min") is not None and grid:
            # regular grid: reconstruct the coordinate axes from the metadata
            lat = np.linspace(ll["lat_min"], ll["lat_max"], int(grid[0]))
            lon = np.linspace(ll["lon_min"], ll["lon_max"], int(grid[1]))

    files: dict = {}
    for var, days in triplets.items():
        days = sorted(days)
        gts, preds, biases = [], [], []
        for d in days:
            base = out_dir / f"{PREFIX}_{var}_day{d:02d}"
            gts.append(np.load(f"{base}_gt.npy"))
            preds.append(np.load(f"{base}_pred.npy"))
            biases.append(np.load(f"{base}_bias.npy"))
        # one shared color scale per variable across ALL days (comparable);
        # symmetric scale for the bias
        allmain = np.concatenate([a.ravel() for a in gts + preds])
        vmin, vmax = np.nanpercentile(allmain, [1, 99])
        bmax = float(np.nanpercentile(np.abs(np.concatenate([b.ravel() for b in biases])), 99)) or 1.0
        unit = DECLARED_UNITS.get(var, "")

        h, w = gts[0].shape
        v_lat = lat if lat is not None and len(lat) == h else np.linspace(-90, 90, h)
        v_lon = lon if lon is not None and len(lon) == w else np.linspace(0, 360, w, endpoint=False)
        # The saved arrays follow the model's grid ordering, whose first column
        # is the dateline (measured against ERA5: r=0.9999 at a half-width
        # column shift, for every run/variable), while the metadata records the
        # NetCDF axis starting at 0 deg. Shift the axis so values land on the
        # right coastlines: column j sits at (metadata_lon[j] - 180).
        v_lon = np.asarray(v_lon, dtype=float) - 180.0
        # panel keeps the data proportions: width/height = lon span / lat span
        lat_span = float(v_lat[-1] - v_lat[0]) or 180.0
        lon_span = float(v_lon[-1] - v_lon[0]) or 360.0
        panel_w = 6.4
        panel_h = panel_w * (lat_span / lon_span)
        figsize = (3 * panel_w + 2.4, panel_h + 1.3)

        for d, gt, pred, bias in zip(days, gts, preds, biases):
            # standard -180..180 view (Greenwich at center); the DATA stays in
            # its own coordinates — only the projection/view changes
            proj = ccrs.PlateCarree() if HAVE_CARTOPY else None
            fig, axes = plt.subplots(
                1, 3, figsize=figsize, constrained_layout=True,
                subplot_kw={"projection": proj} if proj else {})
            panels = [(gt, "ground truth", "viridis", vmin, vmax),
                      (pred, "prediction", "viridis", vmin, vmax),
                      (bias, "bias (pred − truth)", "RdBu_r", -bmax, bmax)]
            for ax, (arr, title, cmap, lo, hi) in zip(axes, panels):
                if HAVE_CARTOPY:
                    data, lon_c = add_cyclic_point(arr, coord=v_lon)
                    mesh = ax.pcolormesh(lon_c, v_lat, data, cmap=cmap,
                                         vmin=lo, vmax=hi, shading="auto",
                                         transform=ccrs.PlateCarree())
                    ax.coastlines(linewidth=0.55, color="#333333")
                    ax.set_global()
                else:
                    center = 0.0
                    lon_view = ((v_lon - (center - 180.0)) % 360.0) + (center - 180.0)
                    order = np.argsort(lon_view)
                    mesh = ax.pcolormesh(lon_view[order], v_lat, arr[:, order],
                                         cmap=cmap, vmin=lo, vmax=hi, shading="auto")
                    ax.set_aspect("equal")          # degrees: keep 1:2 grid shape
                    ax.set_xlabel("lon"); ax.set_ylabel("lat")
                ax.set_title(title, fontsize=12)
                cb = fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02)
                cb.set_label(unit, fontsize=9)
            fig.suptitle(f"{var} · day {d} · year-mean over test split{grid_note}",
                         fontsize=14, fontweight="bold")
            name = f"map_{var}_day{d:02d}.png"
            fig.savefig(out_dir / name, dpi=150)
            plt.close(fig)
            files.setdefault(var, {})[str(d)] = name
        print(f"[{run_dir.name}] rendered {len(days)} days for {var}"
              f"{' (cartopy)' if HAVE_CARTOPY else ' (no cartopy — plain axes)'}", flush=True)
    return files


def generate_for_run(run_dir: Path, *, gpu: str | None, force: bool) -> bool:
    out_dir = run_dir / OUT_DIR_NAME
    index_path = out_dir / "maps_index.json"
    if index_path.is_file() and not force:
        print(f"[{run_dir.name}] maps_index.json exists — skipping (--force to redo)")
        return False
    record = ingest.build_run_record(run_dir)
    have_arrays = any(out_dir.glob(f"{PREFIX}_*_day*_gt.npy")) if out_dir.is_dir() else False
    if not have_arrays:
        _run_visualizer(run_dir, record, gpu)   # GPU stage only when arrays are absent
    files = _render_pngs(run_dir, out_dir)
    if not files:
        raise RuntimeError(f"{run_dir.name}: no arrays found after the rollout stage")
    days = sorted({int(d) for var in files.values() for d in var})
    index = {
        "generated_at": utc_now_iso(),
        "run_id": run_dir.name,
        "aggregate_mode": "year_mean",
        "split": "test",
        "days": days,
        "variables": sorted(files.keys()),
        "files": files,
        "note": "rendered offline by dashboard/generate_maps.py from arrays "
                "saved by scripts/visualize_rollout_maps.py --save_arrays",
    }
    index_path.write_text(json.dumps(index, indent=1))
    print(f"[{run_dir.name}] wrote {index_path.relative_to(REPO_ROOT)}")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", default=[],
                        help="run id under runs/ (repeatable)")
    parser.add_argument("--all-evaluated", action="store_true",
                        help="every run with a valid evaluation")
    parser.add_argument("--root", default=str(ingest.DEFAULT_RUNS_ROOT))
    parser.add_argument("--gpu", default=None,
                        help="nvidia-smi index to pin (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root)
    targets: list = [root / r for r in args.run]
    if args.all_evaluated:
        for run_dir in ingest.discover_run_dirs(root):
            record = ingest.build_run_record(run_dir)
            if any(e.is_valid for e in record.evaluations):
                targets.append(run_dir)
    if not targets:
        parser.error("nothing to do: pass --run <id> or --all-evaluated")

    failures = 0
    for run_dir in dict.fromkeys(targets):        # dedupe, keep order
        try:
            generate_for_run(run_dir, gpu=args.gpu, force=args.force)
        except Exception as exc:  # noqa: BLE001 — keep going, report at the end
            failures += 1
            print(f"[{run_dir.name}] FAILED: {exc}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
