#!/usr/bin/env python3
"""Regenerate the data block of experiments/experiment_dashboard.html.

Scans runs/*/evaluation_test_weekly52/fixed10_global_best_metrics.json for every
evaluated run (model + persistence per-lead RMSE/ACC), pulls diagnostics
(power spectra, variance ratios, rollout curves) and config summaries, parses
experiments/kai_2.5.csv as an external reference, and rewrites the section
between the BEGIN/END GENERATED markers in the dashboard HTML.

Usage:  python scripts/build_dashboard_data.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = os.path.join(ROOT, "experiments", "experiment_dashboard.html")
KAI_CSV = os.path.join(ROOT, "experiments", "kai_2.5.csv")
BEGIN = "// ==== BEGIN GENERATED DATA (build_dashboard_data.py) ===="
END = "// ==== END GENERATED DATA ===="

VAR_ORDER = ["z500", "t850", "t2m", "msl", "q700", "u850"]
SPECTRA_LEADS = [1, 5, 10]

PRETTY = {
    "s1only_2p5_l3_hidden128_base_100epoch_dense_l3k24": "s1only base (100 ep)",
    "s1only_2p5_l3_hidden128_base_200epoch_overfit_probe": "s1only overfit probe (200 ep)",
    "s1only_2p5_l3_hidden128_base_100epoch_spectral_loss": "s1only spectral loss",
    "dense_l3k24_curriculum_S2toS10_3ep_initckpt": "initckpt (cosine 5e-5)",
    "dense_l3k24_curriculum_S2toS10_3ep_flat_lr5e7": "flat LR 5e-7",
    "dense_l3k24_curriculum_S2toS10_3ep_flat_lr1e7": "flat LR 1e-7",
    "dense_l3k24_curriculum_S2toS10_3ep_lr1e5": "cosine 1e-5 (stopped @S5)",
    "dense_l3k24_curriculum_S2toS10_3ep_initckpt_tisrfix_l2k16": "tisrfix dense-L2 graph",
    "dense_l3k24_curriculum_S2toS10_3ep_initckpt_tisrfix_l0dec2": "tisrfix l0refine×2 dec×2",
    "dense_l3k24_curriculum_S2toS10_3ep_initckpt_w96_tisrfix": "tisrfix width-96",
    "dense_l3k24_orogtisr_lossw_scratch_S1x100_S2toS10x3": "scratch orog+tisr (127 ep)",
}


def sig(x, digits=5):
    try:
        return float(f"{float(x):.{digits}g}")
    except (TypeError, ValueError):
        return None


def read_config(run_dir):
    path = os.path.join(run_dir, "config_resolved.yaml")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def read_params_millions(run_dir):
    path = os.path.join(run_dir, "model_summary.txt")
    if os.path.isfile(path):
        m = re.search(r"trainable_parameters:\s*(\d+)", open(path).read())
        if m:
            return round(int(m.group(1)) / 1e6, 2)
    return None


def config_summary(cfg, run_dir):
    res = {"2p5": "2.5°", "1p5": "1.5°", "5p625": "5.625°"}.get(str(cfg.get("resolution_mode")), str(cfg.get("resolution_mode")))
    sched = list(cfg.get("rollout_schedule") or [])
    epochs = list(cfg.get("rollout_stage_epochs") or [])
    if sched and epochs:
        if len(set(epochs)) == 1:
            stages = f"S{sched[0]}..S{sched[-1]} ×{epochs[0]}ep"
        else:
            stages = f"S{sched[0]}..S{sched[-1]} staged ({'/'.join(str(e) for e in epochs)})"
    else:
        stages = "—"
    th = cfg.get("target_handling") or {}
    forcings = "tisr" in (th.get("known_future_variables") or [])
    init = cfg.get("init_from_checkpoint")
    init_note = f"init from {os.path.basename(os.path.dirname(str(init)))}" if init else "from scratch"
    ks = cfg.get("level_k_neighbors") or []
    return {
        "resolution": res,
        "hidden_dim": cfg.get("hidden_dim"),
        "graph_connectivity": ("k=" + "/".join(str(k) for k in ks)) if ks else "—",
        "params_millions": read_params_millions(run_dir),
        "rollout_stages": stages,
        "lr": f"{cfg.get('lr_schedule_type', '?')} {cfg.get('lr'):g}→{cfg.get('min_lr'):g}" if cfg.get("lr") else "—",
        "forcings_prescribed": forcings,
        "notes": f"{init_note}; {cfg.get('max_epochs')} epochs",
    }


def curves_from_metrics_json(path):
    d = json.load(open(path))
    out = {}
    for label in ("global_best", "persistence"):
        blk = (d.get("checkpoints") or {}).get(label)
        if not blk:
            continue
        m = blk.get("metrics") or {}
        rmse, acc = {}, {}
        for v in VAR_ORDER:
            if v in m:
                rmse[v] = [sig(x) for x in m[v]["rmse"]["mean"]]
                acc[v] = [sig(x, 4) for x in m[v]["acc"]["mean"]]
        out[label] = {"rmse": rmse, "acc": acc}
    out["epoch"] = (d.get("checkpoints") or {}).get("global_best", {}).get("epoch")
    return out


def find_diag_csv(run_dir, pattern):
    hits = sorted(glob.glob(os.path.join(run_dir, "diagnostics_full_eval", pattern)))
    return hits[-1] if hits else None


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def diagnostics_for(run_dir):
    diag = {}
    # variance ratio -> stability_by_var
    vr = find_diag_csv(run_dir, "epoch_*_variance_ratio.csv")
    if vr:
        by = {}
        for row in read_rows(vr):
            if row.get("phase") not in (None, "valid"):
                continue
            by.setdefault(row["variable"], {})[int(row["lead"])] = sig(row["variance_ratio"], 4)
        diag["stability_by_var"] = {v: [d[k] for k in sorted(d)] for v, d in by.items()}
    # power spectrum -> spectra.byVar
    ps = find_diag_csv(run_dir, "tables/epoch_*_power_spectrum.csv") or find_diag_csv(run_dir, "epoch_*_power_spectrum.csv")
    if ps:
        rows = [r for r in read_rows(ps) if r.get("phase") in (None, "valid")]
        by_var = {}
        for r in rows:
            by_var.setdefault(r["variable"], []).append(r)
        spectra = {}
        for v, rws in by_var.items():
            leads_avail = sorted({int(r["lead"]) for r in rws})
            leads = [l for l in SPECTRA_LEADS if l in leads_avail] or leads_avail[:3]
            base = sorted((r for r in rws if int(r["lead"]) == leads[0]), key=lambda r: int(r["wavenumber_bin"]))
            ks, truth = [], []
            for r in base:
                pt = float(r["power_truth"])
                if pt > 0 and int(r["wavenumber_bin"]) > 0:
                    ks.append(int(r["wavenumber_bin"]))
                    truth.append(sig(pt, 4))
            model_by_lead = {}
            for l in leads:
                pred = {int(r["wavenumber_bin"]): float(r["power_pred"]) for r in rws if int(r["lead"]) == l}
                ys = [sig(pred.get(k), 4) if pred.get(k, 0) > 0 else None for k in ks]
                model_by_lead[str(l)] = ys
            if ks:
                spectra[v] = {"wavenumbers": ks, "truth": truth, "model_by_lead": model_by_lead}
        if spectra:
            diag["spectra"] = {"byVar": spectra}
    # rollout curve (valid loss per step at the longest horizon)
    rc = find_diag_csv(run_dir, "epoch_*_rollout_curve.csv") or find_diag_csv(run_dir, "tables/epoch_*_rollout_curve.csv")
    if rc:
        rows = [r for r in read_rows(rc) if r.get("phase") == "valid"]
        if rows:
            hmax = max(int(r["horizon"]) for r in rows)
            pts = sorted(((int(r["step"]), float(r["loss"])) for r in rows if int(r["horizon"]) == hmax))
            diag["rollout_curve"] = {"steps": [p[0] for p in pts], "losses": [sig(p[1], 5) for p in pts]}
    return diag


def collect_runs():
    runs = []
    for met in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "evaluation_test_weekly52", "fixed10_global_best_metrics.json"))):
        run_dir = os.path.dirname(os.path.dirname(met))
        rid = os.path.basename(run_dir)
        curves = curves_from_metrics_json(met)
        gb = curves.get("global_best")
        if not gb or not gb["rmse"]:
            print(f"  skip {rid}: no global_best metrics", file=sys.stderr)
            continue
        cfg = read_config(run_dir)
        summary = config_summary(cfg, run_dir)
        baselines = {}
        if curves.get("persistence"):
            baselines["persistence"] = curves["persistence"]
        tags = [summary["resolution"], f"hidden {summary['hidden_dim']}",
                "tisr prescribed" if summary["forcings_prescribed"] else "tisr predicted"]
        runs.append({
            "id": rid,
            "name": PRETTY.get(rid, rid),
            "tags": tags,
            "checkpoint_epoch": curves.get("epoch"),
            "config": summary,
            "metrics": {"leads": list(range(1, 11)), "rmse": gb["rmse"], "acc": gb["acc"], "baselines": baselines},
            "diagnostics": diagnostics_for(run_dir),
        })
    # best day-10 z500 RMSE first (palette slots go to the strongest runs)
    runs.sort(key=lambda r: (r["metrics"]["rmse"].get("z500") or [float("inf")] * 10)[-1])
    return runs


def collect_kai():
    if not os.path.isfile(KAI_CSV):
        return []
    rmse, acc = {}, {}
    for row in csv.DictReader(open(KAI_CSV)):
        v, t = row["variable"], int(row["timestep"])
        rmse.setdefault(v, {})[t] = sig(row["rmse"])
        acc.setdefault(v, {})[t] = sig(row["acc"], 4)
    to_arr = lambda d: [d.get(t) for t in range(1, 11)]
    return [{
        "id": "kai-2p5",
        "name": "KAI (2.5° reference)",
        "external": True,
        "color": {"light": "#52514e", "dark": "#c3c2b7"},
        "metrics": {"leads": list(range(1, 11)),
                    "rmse": {v: to_arr(d) for v, d in rmse.items()},
                    "acc": {v: to_arr(d) for v, d in acc.items()},
                    "baselines": {}},
    }]


def main():
    runs = collect_runs()
    external = collect_kai()
    print(f"collected {len(runs)} runs, {len(external)} external reference(s)")
    for r in runs:
        d10 = (r["metrics"]["rmse"].get("z500") or [None] * 10)[-1]
        has = ",".join(k for k in ("spectra", "stability_by_var", "rollout_curve") if k in r["diagnostics"])
        print(f"  {r['id']}: z500 d10 RMSE {d10} | diag: {has or 'none'}")
    block = (f"{BEGIN}\n"
             f"const RUNS = {json.dumps(runs, indent=1)};\n\n"
             f"const EXTERNAL = {json.dumps(external, indent=1)};\n"
             f"{END}")
    html = open(HTML).read()
    i, j = html.index(BEGIN), html.index(END) + len(END)
    open(HTML, "w").write(html[:i] + block + html[j:])
    print(f"wrote data block into {os.path.relpath(HTML, ROOT)} "
          f"({len(block)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
