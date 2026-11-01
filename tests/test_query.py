"""Conservative query expansion contract tests without model calls."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.retrieval import query as query_module

SID = "20260101T000000Z-00000001"


class QueryExpansionTests(unittest.TestCase):
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

    def _call(self, question, fuse_results, expand=True):
        with patch.object(query_module, "search", return_value=[]) as search, \
             patch.object(query_module, "fuse_and_rerank", side_effect=fuse_results) as fuse:
            result = query_module.query(self.indexes.parent, SID, question, expand=expand)
        return result, search, fuse

    def test_no_expansion_when_two_hits(self):
        result, search, _ = self._call("什么是 RAG？", [[{"chunk_id": "a"}, {"chunk_id": "b"}]])
        self.assertEqual([chunk["chunk_id"] for chunk in result], ["a", "b"])
        self.assertEqual(search.call_count, 3)

    def test_expansion_merges_new_chunk_when_single_hit(self):
        result, search, fuse = self._call(
            "什么是 RAG？", [[{"chunk_id": "a"}], [{"chunk_id": "a"}, {"chunk_id": "b"}]])
        self.assertEqual([chunk["chunk_id"] for chunk in result], ["a", "b"])
        self.assertEqual(search.call_count, 6)
        self.assertEqual(fuse.call_args_list[1].args[0], "RAG")

    def test_no_expansion_without_technical_terms(self):
        result, search, _ = self._call("结论是什么", [[{"chunk_id": "a"}]])
        self.assertEqual([chunk["chunk_id"] for chunk in result], ["a"])
        self.assertEqual(search.call_count, 3)

    def test_expansion_can_be_disabled(self):
        result, search, _ = self._call("什么是 RAG？", [[{"chunk_id": "a"}]], expand=False)
        self.assertEqual([chunk["chunk_id"] for chunk in result], ["a"])
        self.assertEqual(search.call_count, 3)


if __name__ == "__main__":
    unittest.main()
