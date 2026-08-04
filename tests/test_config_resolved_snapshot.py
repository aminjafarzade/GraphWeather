"""Phase-0 safety net: golden snapshot of every resolved config.

This is the regression net that MUST exist before any config refactor (typed
schema, YParams shim, YAML `_extends` dedup) lands. It captures the *resolved*
``YParams.params`` for every config section under ``configs/`` (and
``configs/experiments/``) across each resolution mode, using the CURRENT code,
and stores a type-strict golden. Any later change that alters a resolved value
-- even bool->int or float->int -- fails this test with a readable path.

Run:  python -m unittest tests.test_config_resolved_snapshot
      (first run generates the golden and skips; re-run compares)

This module adds NO production code and changes no behavior; it only observes.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import YParams  # noqa: E402

import yaml  # noqa: E402

ROOT_STR = str(PROJECT_ROOT)
GOLDEN_DIR = PROJECT_ROOT / "tests" / "config_resolved_goldens"
GOLDEN_FILE = GOLDEN_DIR / "resolved_config_goldens.json"
MODES = [None, "2p5", "5p625"]


# --------------------------------------------------------------------------- #
# Type-strict canonicalization + comparator (reused later for old-vs-new gate) #
# --------------------------------------------------------------------------- #
def canon(obj) -> str:
    """Deterministic, TYPE-TAGGED serialization.

    bool is tagged before int so ``True`` != ``1``; float is tagged separately
    so ``1.0`` != ``1``. Dict keys are sorted; the repo-root prefix in string
    leaves is replaced with ``<ROOT>`` so goldens do not hard-code the checkout
    path.
    """
    if isinstance(obj, bool):
        return "bool:%s" % obj
    if isinstance(obj, int):
        return "int:%d" % obj
    if isinstance(obj, float):
        return "float:%r" % obj
    if obj is None:
        return "none"
    if isinstance(obj, str):
        return "str:%r" % obj.replace(ROOT_STR, "<ROOT>")
    if isinstance(obj, dict):
        return "{" + ",".join("%r:%s" % (k, canon(obj[k])) for k in sorted(obj, key=str)) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(canon(v) for v in obj) + "]"
    return "%s:%r" % (type(obj).__name__, obj)


def type_strict_diff(a, b, path="$"):
    """Return a list of human-readable diff strings; empty means identical.

    Strictly type-aware: ``type(True) is bool`` differs from ``type(1) is int``,
    and ``float`` differs from ``int``. Same key-SET required (order-insensitive).
    """
    diffs = []
    if type(a) is not type(b):
        diffs.append("%s: type %s != %s (%r vs %r)" % (path, type(a).__name__, type(b).__name__, a, b))
        return diffs
    if isinstance(a, dict):
        ka, kb = set(a), set(b)
        for k in sorted(ka - kb, key=str):
            diffs.append("%s.%s: only in current" % (path, k))
        for k in sorted(kb - ka, key=str):
            diffs.append("%s.%s: only in golden" % (path, k))
        for k in sorted(ka & kb, key=str):
            diffs.extend(type_strict_diff(a[k], b[k], "%s.%s" % (path, k)))
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append("%s: list len %d != %d" % (path, len(a), len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs.extend(type_strict_diff(x, y, "%s[%d]" % (path, i)))
    else:
        if a != b:
            diffs.append("%s: %r != %r" % (path, a, b))
    return diffs


def _jsonable(obj):
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return {"__canon__": canon(obj)}


def _sections_of(path: Path):
    root = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        return []
    return [k for k, v in root.items() if isinstance(v, dict)]


def _iter_config_files():
    yield from sorted((PROJECT_ROOT / "configs").glob("*.yaml"))
    yield from sorted((PROJECT_ROOT / "configs" / "experiments").glob("*.yaml"))


def _sanitize(text: str) -> str:
    return text.replace(ROOT_STR, "<ROOT>")


def _source_section_is_opt_in_mesh(snapshot_key: str) -> bool:
    """Detect mesh configs even when an intentionally unsupported mode errors."""

    try:
        relative_path, section, _ = snapshot_key.split("::", 2)
        root = yaml.safe_load(
            (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        )
        mesh_encoder = root[section].get("mesh_encoder")
        return isinstance(mesh_encoder, dict) and bool(
            mesh_encoder.get("enabled", False)
        )
    except (KeyError, TypeError, ValueError, OSError, yaml.YAMLError):
        return False


def build_entries() -> dict:
    """Resolve every (file, section, mode) through the CURRENT code path.

    Success -> {"canon", "resolved", "error": None}; failure -> the error type
    and message are captured so that a change in *error* behavior is also caught.
    """
    entries: dict = {}
    for path in _iter_config_files():
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        try:
            sections = _sections_of(path)
        except Exception as exc:  # pragma: no cover - malformed YAML would be a real change
            entries["%s::<loadfail>::-" % rel] = {
                "canon": "LOADFAIL::%s::%s" % (type(exc).__name__, _sanitize(str(exc))),
                "resolved": None,
                "error": _sanitize(str(exc)),
            }
            continue
        for section in sections:
            for mode in MODES:
                key = "%s::%s::%s" % (rel, section, mode or "default")
                try:
                    yp = YParams(str(path), section, resolution_mode=mode)
                    entries[key] = {"canon": canon(yp.params), "resolved": _jsonable(yp.params), "error": None}
                except Exception as exc:
                    msg = _sanitize("%s: %s" % (type(exc).__name__, exc))
                    entries[key] = {"canon": "ERROR::%s" % msg, "resolved": None, "error": msg}
    return entries


class TypeStrictComparatorTest(unittest.TestCase):
    """The comparator must catch exactly the silent-divergence classes we fear."""

    def test_identical_structures_have_no_diff(self):
        a = {"x": 1, "y": [1, 2, {"z": True}], "p": None, "f": 1.5}
        self.assertEqual(type_strict_diff(a, json.loads(json.dumps(a))), [])

    def test_bool_vs_int_is_flagged(self):
        self.assertTrue(type_strict_diff({"a": True}, {"a": 1}))

    def test_float_vs_int_is_flagged(self):
        self.assertTrue(type_strict_diff({"a": 1.0}, {"a": 1}))

    def test_missing_key_and_length_flagged(self):
        self.assertTrue(type_strict_diff({"a": 1}, {"a": 1, "b": 2}))
        self.assertTrue(type_strict_diff([1, 2], [1, 2, 3]))

    def test_canon_distinguishes_bool_int_float(self):
        self.assertNotEqual(canon(True), canon(1))
        self.assertNotEqual(canon(1.0), canon(1))


class ConfigResolvedSnapshotTest(unittest.TestCase):
    """Golden snapshot of resolved configs. Baseline-generates on first run."""

    def test_resolved_configs_match_golden(self):
        current = build_entries()
        self.assertGreater(len(current), 0, "no configs discovered under configs/")

        if not GOLDEN_FILE.exists():
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            GOLDEN_FILE.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
            self.skipTest(
                "Baseline golden generated with %d entries at %s. Re-run to compare."
                % (len(current), GOLDEN_FILE.relative_to(PROJECT_ROOT))
            )

        golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
        cur_keys, gold_keys = set(current), set(golden)
        problems = []
        for k in sorted(gold_keys - cur_keys):
            problems.append("[missing now] %s" % k)
        for k in sorted(cur_keys - gold_keys):
            # New opt-in mesh experiment configs have their own targeted config
            # tests. Do not force the legacy grid-mode golden to change merely
            # because a new checkpoint family was added.
            resolved = current[k].get("resolved") or {}
            mesh_encoder = resolved.get("mesh_encoder") if isinstance(resolved, dict) else None
            if (
                isinstance(mesh_encoder, dict)
                and bool(mesh_encoder.get("enabled", False))
            ) or _source_section_is_opt_in_mesh(k):
                continue
            problems.append("[new now]     %s" % k)
        for k in sorted(cur_keys & gold_keys):
            if current[k]["canon"] != golden[k]["canon"]:
                cur_res, gold_res = current[k].get("resolved"), golden[k].get("resolved")
                if cur_res is not None and gold_res is not None:
                    detail = "; ".join(type_strict_diff(cur_res, gold_res)[:8])
                else:
                    detail = "canon changed (current=%s.. golden=%s..)" % (
                        current[k]["canon"][:80],
                        golden[k]["canon"][:80],
                    )
                problems.append("[changed] %s | %s" % (k, detail))

        if problems:
            self.fail(
                "%d resolved-config diffs vs golden (behaviour changed!):\n%s"
                % (len(problems), "\n".join(problems[:40]))
            )


if __name__ == "__main__":
    unittest.main()
