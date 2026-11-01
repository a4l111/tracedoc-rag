"""Metadata filter contract tests (doc_ids / category / page range)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.retrieval import query as query_module

SID = "20260101T000000Z-00000001"


class MatchesFiltersTests(unittest.TestCase):
    def test_doc_id_match_and_mismatch(self):
        chunk = {"doc_id": "docA", "page_start": 1, "page_end": 2}
        self.assertTrue(query_module._matches_filters(chunk, ["docA"], None, None))
        self.assertFalse(query_module._matches_filters(chunk, ["docB"], None, None))

    def test_no_filters_matches_everything(self):
        chunk = {"doc_id": "docA"}
        self.assertTrue(query_module._matches_filters(chunk, None, None, None))
        self.assertTrue(query_module._matches_filters(chunk, [], None, None))

    def test_page_range_overlap(self):
        chunk = {"page_start": 2, "page_end": 3}
        self.assertTrue(query_module._matches_filters(chunk, None, 1, 4))
        self.assertTrue(query_module._matches_filters(chunk, None, 3, 4))  # partial overlap
        self.assertFalse(query_module._matches_filters(chunk, None, 4, 6))

    def test_open_ended_page_range(self):
        chunk = {"page_start": 5, "page_end": 6}
        self.assertTrue(query_module._matches_filters(chunk, None, 3, None))  # only lower bound
        self.assertFalse(query_module._matches_filters(chunk, None, 7, None))
        self.assertTrue(query_module._matches_filters(chunk, None, None, 6))  # only upper bound
        self.assertFalse(query_module._matches_filters(chunk, None, None, 4))

    def test_page_filter_rejects_page_less_chunks(self):
        chunk = {"doc_id": "docA", "page_start": None, "page_end": None}
        self.assertFalse(query_module._matches_filters(chunk, None, 1, 3))
        self.assertTrue(query_module._matches_filters(chunk, None, None, None))


class QueryFilterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.indexes = Path(self.temp.name) / "indexes" / SID
        self.indexes.mkdir(parents=True)
        (self.indexes / "build_report.json").write_text(json.dumps({
            "snapshot_id": SID, "complete": True,
            "routes": {"dense": {"model": "Qwen3-Embedding-8B"}}}), encoding="utf-8")
        (self.indexes / "snapshot.json").write_text(json.dumps({
            "model_config": {"reranker": {"provider": "moark", "model": "Qwen3-Reranker-8B"}}}), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def _chunks(self):
        return [
            {"chunk_id": "c1", "doc_id": "docA", "page_start": 1, "page_end": 2},
            {"chunk_id": "c2", "doc_id": "docB", "page_start": 5, "page_end": 6},
            {"chunk_id": "c3", "doc_id": "docA", "page_start": 10, "page_end": 11},
        ]

    def test_invalid_doc_ids_rejected(self):
        with self.assertRaises(ValueError):
            query_module.query(self.indexes.parent, SID, "问题", doc_ids="docA")
        with self.assertRaises(ValueError):
            query_module.query(self.indexes.parent, SID, "问题", doc_ids=[1])

    def test_invalid_page_range_rejected(self):
        with self.assertRaises(ValueError):
            query_module.query(self.indexes.parent, SID, "问题", page_start=0)
        with self.assertRaises(ValueError):
            query_module.query(self.indexes.parent, SID, "问题", page_start=True)
        with self.assertRaises(ValueError):
            query_module.query(self.indexes.parent, SID, "问题", page_start=5, page_end=3)

    def test_filter_applied_before_fusion(self):
        captured = {}

        def fuse(question, rankings, on_step=None):
            captured.update(rankings)
            return [{"chunk_id": "c1", "doc_id": "docA", "page_start": 1, "page_end": 2}]

        with patch.object(query_module, "search", return_value=self._chunks()) as search, \
             patch.object(query_module, "fuse_and_rerank", side_effect=fuse):
            result = query_module.query(self.indexes.parent, SID, "问题",
                                        doc_ids=["docA"], page_start=1, page_end=3, expand=False)

        for route in query_module.ROUTES:
            self.assertEqual([c["chunk_id"] for c in captured[route]], ["c1"])
        self.assertEqual(search.call_args.kwargs["top_k"], 50)  # retrieve_k raised when filtered
        self.assertEqual([c["chunk_id"] for c in result], ["c1"])

    def test_no_filter_uses_default_retrieve_k(self):
        with patch.object(query_module, "search", return_value=self._chunks()) as search, \
             patch.object(query_module, "fuse_and_rerank", return_value=[]):
            query_module.query(self.indexes.parent, SID, "问题", expand=False)
        self.assertEqual(search.call_args.kwargs["top_k"], 20)


if __name__ == "__main__":
    unittest.main()
