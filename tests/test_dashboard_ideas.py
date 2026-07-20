"""Contract tests for dashboard/ideas.json.

The ideas tab is authored content, not scanned artifacts, so nothing validates it
on ingest. But dashboard/static/main.js renders it with a fixed, silent contract:
only three block types exist, every table row is zipped positionally against the
column list, and every string is HTML-escaped before display. A malformed entry
does not error -- it renders wrong (ragged tables, literal markdown, empty
sections). These tests pin that contract.

unittest-style, runnable by pytest. No GPU, no network, no writes.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

IDEAS_PATH = PROJECT_ROOT / "dashboard" / "ideas.json"

# main.js ideaBlockHtml() returns "" for anything else, so an unknown type is an
# invisible block rather than an error.
RENDERABLE_BLOCK_TYPES = {"p", "list", "table"}

# Substrings that betray markdown/HTML authored into a field that main.js escapes.
LEAKED_MARKUP = ("`", "**", "<b>", "</b>", "<i>", "</i>", "<br", "<code>", "&nbsp;", "&amp;")


def _load() -> list:
    with IDEAS_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    ideas = payload.get("ideas")
    assert isinstance(ideas, list), "ideas.json must have a top-level 'ideas' list"
    return ideas


def _iter_text(idea: dict):
    """Yield (location, string) for every field main.js renders as escaped text."""
    for key in ("title", "verdict", "status", "date"):
        if idea.get(key):
            yield f"{idea.get('id')}.{key}", str(idea[key])
    for s_i, section in enumerate(idea.get("sections", [])):
        where = f"{idea.get('id')}.sections[{s_i}]"
        yield f"{where}.heading", str(section.get("heading", ""))
        for b_i, block in enumerate(section.get("blocks", [])):
            b_where = f"{where}.blocks[{b_i}]"
            if block.get("type") == "p":
                yield f"{b_where}.text", str(block.get("text", ""))
            elif block.get("type") == "list":
                for i_i, item in enumerate(block.get("items", [])):
                    yield f"{b_where}.items[{i_i}]", str(item)
            elif block.get("type") == "table":
                for c_i, col in enumerate(block.get("columns", [])):
                    yield f"{b_where}.columns[{c_i}]", str(col)
                for r_i, row in enumerate(block.get("rows", [])):
                    for c_i, cell in enumerate(row):
                        yield f"{b_where}.rows[{r_i}][{c_i}]", str(cell)


class TestIdeasFile(unittest.TestCase):
    def setUp(self) -> None:
        self.ideas = _load()

    def test_file_is_non_empty(self) -> None:
        self.assertGreater(len(self.ideas), 0, "ideas.json has no entries")

    def test_required_fields_present(self) -> None:
        for idea in self.ideas:
            for key in ("id", "title", "date", "status", "verdict", "sections"):
                self.assertIn(key, idea, f"idea {idea.get('id')!r} is missing {key!r}")
                self.assertTrue(str(idea[key]).strip(), f"idea {idea.get('id')!r} has an empty {key!r}")

    def test_ids_are_unique(self) -> None:
        ids = [idea.get("id") for idea in self.ideas]
        self.assertEqual(len(ids), len(set(ids)), f"duplicate idea ids: {ids}")

    def test_dates_are_iso(self) -> None:
        import datetime

        for idea in self.ideas:
            datetime.date.fromisoformat(str(idea["date"]))  # raises on a bad date

    def test_every_block_type_renders(self) -> None:
        for idea in self.ideas:
            for s_i, section in enumerate(idea.get("sections", [])):
                self.assertTrue(str(section.get("heading", "")).strip(),
                                f"{idea['id']}.sections[{s_i}] has no heading")
                blocks = section.get("blocks", [])
                self.assertGreater(len(blocks), 0,
                                   f"{idea['id']}.sections[{s_i}] has no blocks")
                for b_i, block in enumerate(blocks):
                    self.assertIn(
                        block.get("type"), RENDERABLE_BLOCK_TYPES,
                        f"{idea['id']}.sections[{s_i}].blocks[{b_i}] type "
                        f"{block.get('type')!r} renders as nothing",
                    )

    def test_table_rows_match_columns(self) -> None:
        """main.js zips cells positionally; a ragged row silently shifts the table."""
        for idea in self.ideas:
            for s_i, section in enumerate(idea.get("sections", [])):
                for b_i, block in enumerate(section.get("blocks", [])):
                    if block.get("type") != "table":
                        continue
                    where = f"{idea['id']}.sections[{s_i}].blocks[{b_i}]"
                    columns = block.get("columns", [])
                    self.assertGreater(len(columns), 0, f"{where} has no columns")
                    for r_i, row in enumerate(block.get("rows", [])):
                        self.assertEqual(
                            len(row), len(columns),
                            f"{where}.rows[{r_i}] has {len(row)} cells for "
                            f"{len(columns)} columns",
                        )

    def test_block_payload_matches_its_type(self) -> None:
        for idea in self.ideas:
            for s_i, section in enumerate(idea.get("sections", [])):
                for b_i, block in enumerate(section.get("blocks", [])):
                    where = f"{idea['id']}.sections[{s_i}].blocks[{b_i}]"
                    kind = block.get("type")
                    if kind == "p":
                        self.assertTrue(str(block.get("text", "")).strip(), f"{where} is an empty paragraph")
                    elif kind == "list":
                        self.assertGreater(len(block.get("items", [])), 0, f"{where} is an empty list")
                    elif kind == "table":
                        self.assertGreater(len(block.get("rows", [])), 0, f"{where} is an empty table")

    def test_no_markup_leaks_into_escaped_text(self) -> None:
        """Every rendered string is HTML-escaped, so markup would show up literally."""
        offenders = []
        for idea in self.ideas:
            for where, text in _iter_text(idea):
                for token in LEAKED_MARKUP:
                    if token in text:
                        offenders.append(f"{where}: contains {token!r}")
        self.assertEqual(offenders, [], "markup leaked into escaped text:\n" + "\n".join(offenders))

    def test_references_are_label_url_pairs(self) -> None:
        for idea in self.ideas:
            for r_i, ref in enumerate(idea.get("references", [])):
                where = f"{idea['id']}.references[{r_i}]"
                self.assertIn("label", ref, f"{where} has no label")
                self.assertIn("url", ref, f"{where} has no url")
                self.assertTrue(
                    str(ref["url"]).startswith(("http://", "https://")),
                    f"{where} url is not absolute: {ref['url']!r}",
                )


class TestIdeasEndpoint(unittest.TestCase):
    """The API must hand the frontend exactly what the file contains."""

    def test_endpoint_returns_every_idea(self) -> None:
        import tempfile

        from fastapi.testclient import TestClient

        from dashboard.app import create_app

        # /api/ideas reads dashboard/ideas.json directly and ignores runs_root, so
        # point the scanner at an empty dir to keep the test off the real runs/.
        with tempfile.TemporaryDirectory() as empty_runs:
            app = create_app(Path(empty_runs), auto_scan=False, externals=[])
            client = TestClient(app)
            response = client.get("/api/ideas")
            self.assertEqual(response.status_code, 200)
            served = response.json().get("ideas", [])
        self.assertEqual(
            [i["id"] for i in served], [i["id"] for i in _load()],
            "the endpoint dropped or reordered entries relative to ideas.json",
        )


if __name__ == "__main__":
    unittest.main()
