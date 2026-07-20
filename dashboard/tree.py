"""Editable experiment-lineage FOREST for the Summary tab.

Multiple independent trees are supported: any node may have parent_id null
(a root). Deleting a node re-attaches its children to its parent — deleting
a root therefore promotes its children to independent roots.

The tree is USER-AUTHORED metadata (rationale, expectations, verdicts, notes),
not derived from artifacts — so unlike everything else in the dashboard it is
writable. It persists to dashboard/data/tree.json (the service's own data
location; runs/ stays strictly read-only). Writes are atomic (tmp + rename)
and guarded by a lock. Nodes may reference a run_id; the frontend joins that
against /api/runs for live status.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Optional

from .contract import utc_now_iso

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_TREE_PATH = DATA_DIR / "tree.json"

NODE_STATUSES = ("completed", "in-progress", "suggested")
NODE_VERDICTS = ("keep", "kill", "promising", "pending")

EDITABLE_FIELDS = ("title", "change", "rationale", "expected", "actual",
                   "verdict", "status", "winning", "run_id", "notes", "parent_id")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(title: str) -> str:
    return _SLUG_RE.sub("-", title.lower()).strip("-")[:40] or "node"


def seed_nodes() -> list:
    """The real 2.5° lineage as of 2026-07-15 (mirrors the static dashboard's
    tree, plus the in-flight hidden-160 and 1p5 branches)."""
    now = utc_now_iso()

    def n(id, parent, title, change, rationale, expected, actual, verdict,
          status, run_id=None, winning=False):
        return {"id": id, "parent_id": parent, "run_id": run_id, "title": title,
                "change": change, "rationale": rationale, "expected": expected,
                "actual": actual, "verdict": verdict, "status": status,
                "winning": winning, "notes": "", "created_at": now,
                "updated_at": now}

    return [
        n("s1-base", None, "S1 base (100 ep)",
          "Foundation: 2.5°, hidden 128, dense L3 (k=24), single-step training, cosine 1e-4→1e-6.",
          "A strong 1-step model to warm-start all rollout fine-tunes from.",
          "Good day-1 skill; poor long-lead skill without rollout training.",
          "Solid S1 skill (valid_S1_final 0.0234); long-lead degrades fast without curriculum, as expected.",
          "keep", "completed", run_id="2p5_l3_h128_densel3k24_s1x100", winning=True),
        n("probe-overfit", "s1-base", "200-ep overfit probe",
          "Same S1 recipe, 200 epochs, tracking the train/valid gap.",
          "Check whether longer S1 pretraining overfits before scaling it.",
          "Gap growth if memorization sets in.",
          "Gap stays ~0.002 — no meaningful single-step overfitting.",
          "keep", "completed", run_id="2p5_l3_h128_rowaware_s1x200_overfitprobe"),
        n("probe-spectral", "s1-base", "S1 spectral-loss probe",
          "S1 base recipe + spectral loss term enabled.",
          "Pre-test the spectral penalty (targets blurring) in the cheap S1 setting.",
          "Similar RMSE with better high-wavenumber energy retention.",
          "Completed; compare its curves and spectra in the dashboard before promoting.",
          "promising", "completed", run_id="2p5_l3_h128_rowaware_s1x100_spectral"),
        n("initckpt", "s1-base", "curriculum fine-tune (cosine 5e-5)",
          "Warm start from the S1 base; S2→S10 ×3 epochs; warmup-cosine 5e-5→1e-6.",
          "Teach stable multi-step rollouts without losing S1 skill.",
          "Large long-lead gains vs the S1 base.",
          "Best run to date: day-10 z500 RMSE 683.8 / ACC 0.493; z500 ACC≥0.6 through day 8. Reference model.",
          "keep", "completed", run_id="2p5_l3_h128_densel3k24_currS2toS10x3_initckpt", winning=True),
        n("flat5e7", "initckpt", "flat LR 5e-7",
          "Same curriculum, constant LR 5e-7 instead of cosine.",
          "LR ablation: is the cosine schedule doing the work?",
          "Slightly worse than cosine if the early high-LR phase matters.",
          "Close second (day-10 z500 681.0) but loses on nearly every other variable.",
          "kill", "completed", run_id="2p5_l3_h128_densel3k24_currS2toS10x3_flatlr5e7"),
        n("flat1e7", "initckpt", "flat LR 1e-7",
          "Same curriculum, constant LR 1e-7.",
          "Lower bound of the flat-LR ablation.",
          "Under-trained if too small.",
          "Consistently below flat 5e-7 everywhere — under-powered.",
          "kill", "completed", run_id="2p5_l3_h128_densel3k24_currS2toS10x3_flatlr1e7"),
        n("lr1e5", "initckpt", "cosine 1e-5 (stopped @S5)",
          "Same curriculum, cosine peak 1e-5.",
          "Does a lower cosine peak help long-horizon stability?",
          "Similar to 5e-5 if the peak matters little.",
          "INVALID as an LR test: training stopped at epoch 10 (S5 stage); numbers match initckpt's S5 checkpoint.",
          "kill", "completed", run_id="2p5_l3_h128_densel3k24_currS2toS10x3_lr1e5"),
        n("hidden160", "initckpt", "hidden 160 (capacity)",
          "hidden_dim 128 → 160 (5 heads), same recipe end-to-end: S1×100 base then S2→S10 ×3.",
          "The 128-dim family lands within CI noise across recipes — test whether capacity separates.",
          "If capacity-limited: uniform gains over initckpt beyond CI noise.",
          "Training now (full pipeline incl. eval, comparison, diagnostics, maps).",
          "promising", "in-progress", run_id="2p5_l3_h160_densel3k24_currS2toS10x3_initckpt", winning=True),
        n("orogtisr", "s1-base", "scratch + orog/tisr forcing",
          "From scratch S1×100 → S2→S10×3 (127 ep), orog+tisr prescribed, latitude-only loss.",
          "Do prescribed forcings + more epochs beat the fine-tune recipe?",
          "Competitive with initckpt if forcings matter.",
          "Second best (day-10 z500 690.2) — within CI of initckpt. Forcing benefit not isolated.",
          "promising", "completed", run_id="2p5_l3_h128_densel3k24_scratch_orogtisr_lossw"),
        n("l2k16", "s1-base", "tisrfix + dense L2 graph",
          "From-scratch staged curriculum (80/1..4 ep), tisr prescribed, denser L2 mesh (k=16).",
          "Larger mid-level receptive field against long-lead blurring.",
          "Better day 5–10 mass fields.",
          "Mid-pack (689.2); confounded — 100 total epochs vs 127 for others. Graph question unresolved.",
          "promising", "completed", run_id="2p5_l3_h128_densel3k24_currS2toS10x3_initckpt_tisrfix_l2k16"),
        n("l0dec2", "l2k16", "2× l0-refine + 2× decoder",
          "Same tisrfix recipe, sparse graph, doubled decode-path depth (+16% params).",
          "More capacity where the finest scales are decoded.",
          "Sharper output if decode depth is the bottleneck.",
          "Tie on valid; worse day-10 t2m on test (3.04 vs 2.56 K) at +74% S10 time. Retired.",
          "kill", "completed", run_id="2p5_l3_h128_densel3k24_currS2toS10x3_initckpt_tisrfix_l0dec2"),
        n("w96", "s1-base", "tisrfix @ width 96",
          "tisrfix recipe at hidden 96 / 3 heads (1.4M params).",
          "How small can the model go with prescribed forcings?",
          "Graceful degradation if forcings offset lost capacity.",
          "Weakest evaluated run — capacity cut dominates; says nothing about the tisr fix.",
          "kill", "completed", run_id="2p5_l3_h96_densel3k24_currS2toS10x3_initckpt_tisrfix"),
        n("res1p5", "initckpt", "1p5 resolution transfer",
          "Same recipe & architecture at 1.5° (121×240 incl. poles) on the kai_1p5 dataset.",
          "Resolution is the largest untested lever.",
          "Better short-lead skill; long-lead parity or better.",
          "Training now (S1 stage; curriculum + weekly eval on 2020 run automatically after).",
          "promising", "in-progress", winning=True),
        n("sug-forcings", "res1p5", "prescribe forcings (isolated)",
          "known_future_variables=[tisr] on the winning recipe — the ONLY change.",
          "t2m is the weakest field everywhere and is radiation-driven; existing tisr runs are confounded.",
          "Largest gains on t2m/q700 ACC at day 5–10; zero parameter cost.",
          "not yet run", "pending", "suggested", winning=True),
        n("sug-ema", "res1p5", "EMA weights",
          "Evaluate an exponential moving average of weights (decay ~0.999).",
          "msl variance ratio falls to ~0.78–0.82 by day 10; EMA is a near-free stabilizer.",
          "+0.005–0.015 ACC at day 7–10, no training cost.",
          "not yet run", "pending", "suggested"),
        n("sug-stagecos", "initckpt", "S10 cosine restart",
          "Extend the final S10 stage +6 epochs with a fresh small cosine (5e-6→5e-7).",
          "Best checkpoint lands on the final epoch; S10 only ever saw LR ≈1e-6 — LR-starved.",
          "Small consistent day 7–10 gain, or a cheap kill of the hypothesis (~8 GPU-h).",
          "not yet run", "pending", "suggested"),
        n("sug-spectral", "initckpt", "spectral loss @ rollout",
          "Enable the spectral-loss hook (mid-band k≈8–24) during the S2→S10 fine-tune.",
          "Every run keeps only ~60% of synoptic-band z500 power at day 10 — a loss-shape failure.",
          "Sharper day 5–10 systems; possible small RMSE penalty (RMSE rewards blur).",
          "not yet run", "pending", "suggested"),
    ]


class TreeStore:
    """Thread-safe JSON-backed store for the lineage tree."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_TREE_PATH
        self._lock = threading.Lock()
        self._nodes: dict = {}
        self._load()

    # -- persistence ----------------------------------------------------------

    def _load(self) -> None:
        if self.path.is_file():
            try:
                payload = json.loads(self.path.read_text(errors="replace"))
                self._nodes = {n["id"]: n for n in payload.get("nodes", [])}
                if self._nodes:
                    return
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # fall through to seed; the broken file is overwritten
        self._nodes = {n["id"]: n for n in seed_nodes()}
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"nodes": list(self._nodes.values())}, indent=1))
        os.replace(tmp, self.path)

    # -- queries ----------------------------------------------------------------

    def nodes(self) -> list:
        with self._lock:
            return [dict(n) for n in self._nodes.values()]

    # -- mutations ----------------------------------------------------------------

    def create(self, fields: dict) -> dict:
        with self._lock:
            parent_id = fields.get("parent_id")
            if parent_id is not None and parent_id not in self._nodes:
                raise ValueError(f"parent {parent_id!r} does not exist")
            # parent_id None is allowed: it starts a new independent tree
            title = str(fields.get("title") or "untitled").strip() or "untitled"
            node_id = fields.get("id") or _slug(title)
            base, k = node_id, 2
            while node_id in self._nodes:
                node_id = f"{base}-{k}"
                k += 1
            status = fields.get("status") or "suggested"
            verdict = fields.get("verdict") or "pending"
            if status not in NODE_STATUSES:
                raise ValueError(f"status must be one of {NODE_STATUSES}")
            if verdict not in NODE_VERDICTS:
                raise ValueError(f"verdict must be one of {NODE_VERDICTS}")
            now = utc_now_iso()
            node = {
                "id": node_id, "parent_id": parent_id,
                "run_id": fields.get("run_id") or None,
                "title": title,
                "change": str(fields.get("change") or ""),
                "rationale": str(fields.get("rationale") or ""),
                "expected": str(fields.get("expected") or ""),
                "actual": str(fields.get("actual") or "not yet run"),
                "verdict": verdict, "status": status,
                "winning": bool(fields.get("winning")),
                "notes": str(fields.get("notes") or ""),
                "created_at": now, "updated_at": now,
            }
            self._nodes[node_id] = node
            self._save()
            return dict(node)

    def update(self, node_id: str, fields: dict) -> dict:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                raise KeyError(node_id)
            for key, value in fields.items():
                if key not in EDITABLE_FIELDS:
                    continue
                if key == "status" and value not in NODE_STATUSES:
                    raise ValueError(f"status must be one of {NODE_STATUSES}")
                if key == "verdict" and value not in NODE_VERDICTS:
                    raise ValueError(f"verdict must be one of {NODE_VERDICTS}")
                if key == "parent_id":
                    if value == node_id or (value is not None and value not in self._nodes):
                        raise ValueError("invalid parent")
                    # refuse cycles: the new parent must not be a descendant
                    cursor = value
                    while cursor is not None:
                        if cursor == node_id:
                            raise ValueError("re-parenting would create a cycle")
                        cursor = (self._nodes.get(cursor) or {}).get("parent_id")
                if key == "winning":
                    value = bool(value)
                node[key] = value
            node["updated_at"] = utc_now_iso()
            self._save()
            return dict(node)

    def delete(self, node_id: str) -> dict:
        """Remove a node; children re-attach to its parent (deleting a root
        promotes its children to independent roots)."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                raise KeyError(node_id)
            for other in self._nodes.values():
                if other["parent_id"] == node_id:
                    other["parent_id"] = node["parent_id"]
                    other["updated_at"] = utc_now_iso()
            removed = self._nodes.pop(node_id)
            self._save()
            return dict(removed)
