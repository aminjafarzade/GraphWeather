"""Tests for the editable experiment-lineage tree (Summary tab backend)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from dashboard.app import create_app  # noqa: E402
from dashboard.tree import TreeStore, seed_nodes  # noqa: E402


class TreeStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "tree.json"
        self.store = TreeStore(self.path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_seed_is_a_valid_forest(self) -> None:
        nodes = self.store.nodes()
        ids = {n["id"] for n in nodes}
        roots = [n for n in nodes if n["parent_id"] is None]
        self.assertGreaterEqual(len(roots), 1)
        for n in nodes:
            if n["parent_id"] is not None:
                self.assertIn(n["parent_id"], ids)

    def test_independent_trees(self) -> None:
        root2 = self.store.create({"title": "side project", "parent_id": None})
        self.assertIsNone(root2["parent_id"])
        child = self.store.create({"title": "branch", "parent_id": root2["id"]})
        # deleting the new root promotes its child to an independent root
        self.store.delete(root2["id"])
        nodes = {n["id"]: n for n in self.store.nodes()}
        self.assertIsNone(nodes[child["id"]]["parent_id"])

    def test_create_update_delete_and_persistence(self) -> None:
        created = self.store.create({"title": "My Experiment!", "parent_id": "initckpt",
                                     "notes": "first idea"})
        self.assertEqual(created["parent_id"], "initckpt")
        self.assertEqual(created["status"], "suggested")
        node_id = created["id"]

        self.store.update(node_id, {"notes": "updated note", "verdict": "promising"})
        # children re-attach to grandparent on delete
        child = self.store.create({"title": "grandchild", "parent_id": node_id})
        self.store.delete(node_id)
        nodes = {n["id"]: n for n in TreeStore(self.path).nodes()}   # reload from disk
        self.assertNotIn(node_id, nodes)
        self.assertEqual(nodes[child["id"]]["parent_id"], "initckpt")

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.store.create({"title": "x", "parent_id": "no-such-node"})
        with self.assertRaises(ValueError):
            self.store.update("initckpt", {"status": "bogus"})
        # detaching a subtree into its own tree is allowed
        moved = self.store.update("l2k16", {"parent_id": None})
        self.assertIsNone(moved["parent_id"])

    def test_reparent_cycle_rejected(self) -> None:
        # initckpt is an ancestor of flat5e7; making initckpt a child of flat5e7 must fail
        with self.assertRaises(ValueError):
            self.store.update("initckpt", {"parent_id": "flat5e7"})

    def test_duplicate_title_gets_unique_id(self) -> None:
        a = self.store.create({"title": "same name", "parent_id": "initckpt"})
        b = self.store.create({"title": "same name", "parent_id": "initckpt"})
        self.assertNotEqual(a["id"], b["id"])


class TreeApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "runsroot").mkdir()
        self.app = create_app(root / "runsroot", auto_scan=False, externals=[],
                              tree_path=root / "tree.json")
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        self._tmp.cleanup()

    def test_full_crud_over_http(self) -> None:
        tree = self.client.get("/api/tree").json()
        self.assertEqual(len(tree["nodes"]), len(seed_nodes()))

        created = self.client.post("/api/tree/nodes", json={
            "title": "http child", "parent_id": "initckpt", "notes": "hi"}).json()
        node_id = created["node"]["id"]

        patched = self.client.patch(f"/api/tree/nodes/{node_id}",
                                    json={"notes": "edited", "status": "in-progress"})
        self.assertEqual(patched.status_code, 200)
        self.assertEqual(patched.json()["node"]["notes"], "edited")

        self.assertEqual(self.client.patch("/api/tree/nodes/nope",
                                           json={"notes": "x"}).status_code, 404)
        self.assertEqual(self.client.post("/api/tree/nodes", json={
            "title": "x", "parent_id": "nope"}).status_code, 422)

        deleted = self.client.delete(f"/api/tree/nodes/{node_id}")
        self.assertEqual(deleted.status_code, 200)
        ids = {n["id"] for n in self.client.get("/api/tree").json()["nodes"]}
        self.assertNotIn(node_id, ids)

        # forest: creating an independent root over HTTP works
        new_root = self.client.post("/api/tree/nodes",
                                    json={"title": "another tree"}).json()
        self.assertIsNone(new_root["node"]["parent_id"])


if __name__ == "__main__":
    unittest.main()
