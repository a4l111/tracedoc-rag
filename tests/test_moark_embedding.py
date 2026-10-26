"""MoArk embedding contract checks without external requests."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import faiss
import numpy as np

from src.index.build import _build_vector_index, build
from src.retrieval.query import search
from src.utils import moark


class MoArkEmbeddingTests(unittest.TestCase):
    def test_missing_key(self) -> None:
        with patch.dict("os.environ", {"MOARK_API_KEY": "your_key_here"}):
            with self.assertRaisesRegex(ValueError, "MOARK_API_KEY"):
                moark.client()

    def test_rerank_scores_follow_original_candidate_order(self) -> None:
        api = SimpleNamespace(post=lambda *args, **kwargs: {"results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.2},
        ]})
        scores = moark.rerank(api, "问题", ["甲", "乙"], "/rerank")
        self.assertEqual(scores.tolist(), [np.float32(0.2), np.float32(0.9)])
        bad = SimpleNamespace(post=lambda *args, **kwargs: {"results": [
            {"index": 1, "relevance_score": 0.9},
        ]})
        with self.assertRaisesRegex(ValueError, "omitted candidates"):
            moark.rerank(bad, "问题", ["甲", "乙"], "/rerank")

    def test_rerank_batches_keep_candidate_positions(self) -> None:
        calls = []
        def post(endpoint, cast_to, body):
            calls.append(body["documents"])
            return {"results": [{"index": i, "relevance_score": float(i)}
                                for i in range(len(body["documents"]))]}
        api = SimpleNamespace(post=post)
        scores = moark.rerank(api, "问题", [str(i) for i in range(45)], "/rerank")
        self.assertEqual([len(batch) for batch in calls], [20, 20, 5])
        self.assertEqual(scores[20], 0.0)
        self.assertEqual(scores[44], 4.0)

    def test_embedding_dimension_and_normalization(self) -> None:
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[3.0] + [0.0] * (moark.DIMENSION - 1))])
        api = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        vector = moark.embed(api, "测试")
        self.assertEqual(vector.shape, (4096,))
        self.assertEqual(calls[0]["dimensions"], 4096)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0)
        self.assertEqual(float(vector[0]), 1.0)

    def test_missing_key_does_not_create_snapshot(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            parsed = root / "parsed" / "doc"
            parsed.mkdir(parents=True)
            (parsed / "manifest.json").write_text("{}", encoding="utf-8")
            indexes = root / "indexes"
            with patch.object(moark, "client", side_effect=ValueError("MOARK_API_KEY is missing")):
                with self.assertRaisesRegex(ValueError, "MOARK_API_KEY"):
                    build(parsed.parent, indexes)
            self.assertFalse(indexes.exists())

    def test_build_and_query_use_moark_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            snapshot_id = "20261024T000000Z-12345678"
            folder = root / snapshot_id
            folder.mkdir()
            chunk = {"chunk_id": "c1", "doc_id": "d1", "text": "测试内容"}
            vector = np.zeros(moark.DIMENSION, dtype="float32")
            vector[0] = 1.0
            with patch.object(moark, "client", return_value=object()), patch.object(moark, "embed", return_value=vector):
                errors = []
                report = _build_vector_index(folder, "dense", [chunk], snapshot_id, "checksum", errors, False)
                self.assertEqual(report["indexed_count"], 1)
                self.assertFalse(errors)
                meta = json.loads((folder / "dense_meta.json").read_text(encoding="utf-8"))
                self.assertEqual(meta["model_source"], "moark")
                self.assertEqual(meta["base_url"], moark.BASE_URL)
                self.assertEqual(faiss.read_index(str(folder / "dense.faiss")).d, 4096)

            line = (json.dumps(chunk) + "\n").encode("utf-8")
            (folder / "chunks.jsonl").write_bytes(line)
            import hashlib
            checksum = hashlib.sha256(line).hexdigest()
            meta["chunks_sha256"] = checksum
            (folder / "dense_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (folder / "snapshot.json").write_text(json.dumps({"snapshot_id": snapshot_id,
                "chunks_sha256": checksum}), encoding="utf-8")
            with patch.object(moark, "client", return_value=object()), patch.object(moark, "embed", return_value=vector):
                results = search(root, snapshot_id, "dense", "问题")
            self.assertEqual(results[0]["chunk_id"], "c1")


if __name__ == "__main__":
    unittest.main()
