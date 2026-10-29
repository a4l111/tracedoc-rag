"""Metric checks with synthetic labels and mocked retrieval rankings."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation.metrics import compare_chunk_snapshots, evaluate_snapshot, render_report, _relevant_ids


class EvaluationTests(unittest.TestCase):
    def test_chunk_comparison_rejects_different_parsed_inputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            for suffix, markdown_hash, mode in (("11111111", "a", "fixed"), ("22222222", "b", "structured")):
                snapshot_id = "20261021T043050Z-" + suffix
                folder = root / snapshot_id
                folder.mkdir()
                content = (json.dumps({"chunk_id": suffix, "doc_id": "doc", "text": "text",
                                       "page_start": None, "heading_path": []}) + "\n").encode("utf-8")
                (folder / "chunks.jsonl").write_bytes(content)
                (folder / "snapshot.json").write_text(json.dumps({"snapshot_id": snapshot_id,
                    "chunk_mode": mode, "chunks_sha256": hashlib.sha256(content).hexdigest(),
                    "sources": [{"doc_id": "doc", "source_file": "note.txt", "sha256": "same",
                                 "markdown_sha256": markdown_hash}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different parsed Markdown"):
                compare_chunk_snapshots(root, ["20261021T043050Z-11111111", "20261021T043050Z-22222222"])

    def test_recall_and_mrr(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            snapshot_id = "20261021T043050Z-496ab369"
            folder = root / snapshot_id
            folder.mkdir()
            chunks = [{"chunk_id": "a", "doc_id": "doc", "page_start": 1, "page_end": 1},
                      {"chunk_id": "b", "doc_id": "doc", "page_start": 2, "page_end": 2}]
            content = "".join(json.dumps(chunk) + "\n" for chunk in chunks).encode("utf-8")
            (folder / "chunks.jsonl").write_bytes(content)
            (folder / "snapshot.json").write_text(json.dumps({"snapshot_id": snapshot_id,
                "chunk_mode": "fixed", "chunks_sha256": hashlib.sha256(content).hexdigest()}), encoding="utf-8")
            (folder / "build_report.json").write_text(json.dumps({"snapshot_id": snapshot_id,
                "complete": True}), encoding="utf-8")
            gold = root / "gold.jsonl"
            gold.write_text(json.dumps({"question": "Where?", "relevant": [{"doc_id": "doc", "page": 2}]}) + "\n",
                            encoding="utf-8")

            def search_stub(indexes_dir, selected_id, route, question, top_k):
                return [{"chunk_id": "a"}, {"chunk_id": "b"}]

            with patch("src.evaluation.metrics.search", side_effect=search_stub), \
                 patch("src.evaluation.metrics.fuse_and_rerank", return_value=[{"chunk_id": "b"}, {"chunk_id": "a"}]):
                result = evaluate_snapshot(root, snapshot_id, gold)
            self.assertEqual(result["metrics"]["bm25"], {"recall_at_5": 1.0, "mrr": 0.5})
            self.assertEqual(result["metrics"]["hybrid_reranked"], {"recall_at_5": 1.0, "mrr": 1.0})
            with patch("src.evaluation.metrics.search", side_effect=search_stub), \
                 patch("src.evaluation.metrics.fuse_and_rerank", return_value=[{"chunk_id": "b"}, {"chunk_id": "a"}]):
                top_one = evaluate_snapshot(root, snapshot_id, gold, k=1)
            self.assertEqual(top_one["metrics"]["bm25"], {"recall_at_1": 0.0, "mrr": 0.0})
            self.assertEqual(top_one["metrics"]["hybrid_reranked"], {"recall_at_1": 1.0, "mrr": 1.0})
            self.assertEqual(top_one["questions"][0]["routes"]["bm25"]["missed_chunk_ids"], ["b"])
            self.assertIn("### Missed labels", render_report([top_one], gold))
            self.assertEqual(len(top_one["gold_sha256"]), 64)

    def test_slide_gold_label(self) -> None:
        chunks = [{"chunk_id": "slide-1", "doc_id": "deck", "page_start": None,
                   "slide_start": 1, "slide_end": 1},
                  {"chunk_id": "slide-2", "doc_id": "deck", "page_start": None,
                   "slide_start": 2, "slide_end": 2}]
        self.assertEqual(_relevant_ids(chunks, [{"doc_id": "deck", "slide": 2}]), {"slide-2"})

    def test_checkpoint_resume_does_not_repeat_retrieval(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            snapshot_id = "20261021T043050Z-496ab369"
            folder = root / snapshot_id
            folder.mkdir()
            chunks = [{"chunk_id": "a", "doc_id": "doc", "page_start": 1, "page_end": 1}]
            content = (json.dumps(chunks[0]) + "\n").encode("utf-8")
            (folder / "chunks.jsonl").write_bytes(content)
            (folder / "snapshot.json").write_text(json.dumps({"snapshot_id": snapshot_id,
                "chunk_mode": "fixed", "chunks_sha256": hashlib.sha256(content).hexdigest()}), encoding="utf-8")
            (folder / "build_report.json").write_text(json.dumps({"snapshot_id": snapshot_id,
                "complete": True}), encoding="utf-8")
            gold = root / "gold.jsonl"
            gold.write_text(json.dumps({"question": "Where?", "relevant": [{"doc_id": "doc", "page": 1}]}) + "\n",
                            encoding="utf-8")
            checkpoint = root / "checkpoint.json"

            with patch("src.evaluation.metrics.search", return_value=[{"chunk_id": "a"}]), \
                 patch("src.evaluation.metrics.fuse_and_rerank", return_value=[{"chunk_id": "a"}]):
                first = evaluate_snapshot(root, snapshot_id, gold, checkpoint_path=checkpoint)
            with patch("src.evaluation.metrics.search", side_effect=AssertionError("retrieval repeated")), \
                 patch("src.evaluation.metrics.fuse_and_rerank", side_effect=AssertionError("rerank repeated")):
                resumed = evaluate_snapshot(root, snapshot_id, gold, checkpoint_path=checkpoint)

            self.assertEqual(resumed, first)

    def test_one_page_with_multiple_chunks_counts_once(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            snapshot_id = "20261021T043050Z-496ab369"
            folder = root / snapshot_id
            folder.mkdir()
            chunks = [{"chunk_id": "a", "doc_id": "doc", "page_start": 1, "page_end": 1},
                      {"chunk_id": "b", "doc_id": "doc", "page_start": 1, "page_end": 1}]
            content = "".join(json.dumps(chunk) + "\n" for chunk in chunks).encode()
            (folder / "chunks.jsonl").write_bytes(content)
            (folder / "snapshot.json").write_text(json.dumps({"snapshot_id": snapshot_id,
                "chunk_mode": "fixed", "chunks_sha256": hashlib.sha256(content).hexdigest()}), encoding="utf-8")
            (folder / "build_report.json").write_text(json.dumps({"snapshot_id": snapshot_id,
                "complete": True}), encoding="utf-8")
            gold = root / "gold.jsonl"
            gold.write_text(json.dumps({"question": "Where?", "relevant": [{"doc_id": "doc", "page": 1}]}) + "\n",
                            encoding="utf-8")
            with patch("src.evaluation.metrics.search", return_value=[{"chunk_id": "a"}]), \
                 patch("src.evaluation.metrics.fuse_and_rerank", return_value=[{"chunk_id": "a"}]):
                result = evaluate_snapshot(root, snapshot_id, gold, k=1)
            self.assertEqual(result["metrics"]["bm25"]["recall_at_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
