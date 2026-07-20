"""Polling run store: fingerprint sweeps, cache, grace window, SSE fan-out.

The repo lives on Lustre, where inotify is unreliable — the store polls
(default 20 s) and re-parses only runs whose fingerprint changed
(docs/03-ARCHITECTURE.md). Run directories are opened strictly read-only.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional

from . import ingest
from .contract import Problem, RunRecord, utc_now_iso

# A metrics JSON that fails to parse may be mid-write (the evaluator writes it
# in one pass near the end of the eval). It gets this many sweeps of grace —
# surfaced as a warning, the eval withheld — before the invalid record is
# published (docs/03, "partial / still-writing runs").
GRACE_SWEEPS = 2


class RunStore:
    """Thread-safe snapshot of RunRecords, updated by polling sweeps."""

    def __init__(self, runs_root=None, *, grace_sweeps: int = GRACE_SWEEPS,
                 fresh_seconds: int = ingest.FRESH_SECONDS) -> None:
        self.runs_root = runs_root or ingest.DEFAULT_RUNS_ROOT
        self.grace_sweeps = grace_sweeps
        self.fresh_seconds = fresh_seconds
        self._lock = threading.Lock()
        self._runs: dict = {}                 # run_id -> RunRecord
        self._parse_failures: dict = {}       # json path -> consecutive count
        self._grace_runs: set = set()          # run_ids to re-parse despite fingerprint
        self._subscribers: list = []          # asyncio.Queue
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.last_sweep: Optional[str] = None
        self.sweep_count = 0

    # -- public snapshot ------------------------------------------------------

    def snapshot(self) -> list:
        with self._lock:
            return list(self._runs.values())

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            return self._runs.get(run_id)

    def problems(self) -> list:
        out = []
        with self._lock:
            for r in self._runs.values():
                out.extend(r.problems)
        out.sort(key=lambda p: p.detected_at, reverse=True)
        return out

    # -- sweeping -------------------------------------------------------------

    def sweep_once(self, *, now: Optional[float] = None) -> list:
        """One polling sweep. Returns the emitted events (also fanned out)."""
        now = time.time() if now is None else now
        events: list = []
        seen: set = set()

        for run_dir in ingest.discover_run_dirs(self.runs_root):
            run_id = run_dir.name
            seen.add(run_id)
            fingerprint = ingest.build_fingerprint(run_dir)
            with self._lock:
                prev = self._runs.get(run_id)
            if prev is not None and prev.status != "removed" \
                    and prev.fingerprint == fingerprint \
                    and run_id not in self._grace_runs:
                prev.last_scanned_at = utc_now_iso()
                continue

            record = ingest.build_run_record(run_dir, now=now,
                                             fresh_seconds=self.fresh_seconds)
            record = self._apply_parse_grace(record, run_dir, now=now)
            if prev is not None:
                record.discovered_at = prev.discovered_at
            with self._lock:
                self._runs[run_id] = record
            events.append({"type": "run_added" if prev is None else "run_updated",
                           "run_id": run_id, "status": record.status,
                           "at": record.last_scanned_at})

        # removals -> tombstone (Q8: kept for this store's lifetime)
        with self._lock:
            missing = [rid for rid, r in self._runs.items()
                       if rid not in seen and r.status != "removed"]
        for run_id in missing:
            with self._lock:
                old = self._runs[run_id]
                tombstone = RunRecord(
                    run_id=run_id, path=old.path, status="removed",
                    discovered_at=old.discovered_at,
                    last_scanned_at=utc_now_iso(),
                    fingerprint={}, architecture=old.architecture,
                    evaluations=[], diagnostics=None, qualitative=None,
                    problems=[Problem(
                        run_id=run_id, eval_id=None, artifact="run_dir",
                        path=old.path, expected="run directory present on disk",
                        found="removed while the dashboard was watching",
                        reason="missing_file", severity="warning",
                    )],
                )
                self._runs[run_id] = tombstone
            events.append({"type": "run_removed", "run_id": run_id,
                           "at": utc_now_iso()})

        self.sweep_count += 1
        self.last_sweep = utc_now_iso()
        for event in events:
            self._publish(event)
        return events

    def _apply_parse_grace(self, record: RunRecord, run_dir, *, now: float) -> RunRecord:
        """Withhold parse-error evals for grace_sweeps sweeps (mid-write race)."""
        failing_paths = {p.path for p in record.problems
                         if p.reason == "parse_error"
                         and p.artifact == "fixed_metrics_json"
                         and p.severity == "error"}
        # reset counters for anything that recovered
        for path in list(self._parse_failures):
            if path not in failing_paths:
                del self._parse_failures[path]
        if not failing_paths:
            self._grace_runs.discard(record.run_id)
            return record

        in_grace: set = set()
        for path in failing_paths:
            count = self._parse_failures.get(path, 0) + 1
            self._parse_failures[path] = count
            if count <= self.grace_sweeps:
                in_grace.add(path)
        if not in_grace:
            self._grace_runs.discard(record.run_id)   # grace exhausted: publish invalid
            return record
        self._grace_runs.add(record.run_id)

        # Rebuild the presentation: drop the stub eval(s), soften the error to
        # a stale_partial warning, and re-derive the status.
        problems = []
        for p in record.problems:
            if p.path in in_grace and p.reason == "parse_error":
                problems.append(Problem(
                    run_id=p.run_id, eval_id=p.eval_id, artifact=p.artifact,
                    path=p.path,
                    expected="parseable metrics JSON",
                    found=f"unreadable — possibly mid-write; grace "
                          f"{self._parse_failures[p.path]}/{self.grace_sweeps} "
                          f"before it is surfaced as invalid",
                    reason="stale_partial", severity="warning",
                ))
            else:
                problems.append(p)
        graced_eval_ids = {p.eval_id for p in record.problems if p.path in in_grace}
        evaluations = [e for e in record.evaluations
                       if not (not e.is_valid and e.eval_id in graced_eval_ids)]
        status = ingest.compute_status(run_dir, record.run_id, evaluations,
                                       problems, now=now,
                                       fresh_seconds=self.fresh_seconds)
        record.problems = problems
        record.evaluations = evaluations
        record.status = status
        return record

    # -- events / SSE ----------------------------------------------------------

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _publish(self, event: dict) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        loop = self._loop
        for q in subscribers:
            if loop is not None:
                loop.call_soon_threadsafe(q.put_nowait, event)
            else:
                q.put_nowait(event)

    # -- background loop -------------------------------------------------------

    async def run_forever(self, interval_s: float = 20.0) -> None:
        self.attach_loop(asyncio.get_running_loop())
        while True:
            try:
                await asyncio.to_thread(self.sweep_once)
            except Exception:  # noqa: BLE001 — a bad sweep must not kill the loop
                import logging
                logging.getLogger("gw-dashboard").exception("sweep failed")
            await asyncio.sleep(interval_s)
