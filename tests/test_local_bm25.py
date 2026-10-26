"""Offline BM25 benchmark checks."""

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.local_bm25 import evaluate_mode, run


class LocalBM25Tests(unittest.TestCase):
    def test_evaluates_both_chunk_modes_without_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parsed = root / "parsed" / "doc"
            parsed.mkdir(parents=True)
            manifest = {"doc_id": "doc", "source_file": "doc.pdf"}
            (parsed / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (parsed / "document.md").write_text(
                "<!-- PAGE:1 -->\n# Fruit\n苹果是一种水果。\n\n<!-- PAGE:2 -->\n# Transport\n火车在轨道上行驶。",
                encoding="utf-8")
            gold = root / "gold.jsonl"
            gold.write_text(json.dumps({"question": "什么在轨道上行驶？",
                                        "relevant": [{"doc_id": "doc", "page": 2}]},
                                       ensure_ascii=False) + "\n", encoding="utf-8")
            fixed = evaluate_mode(root / "parsed", gold, "fixed")
            self.assertEqual(fixed["route"], "bm25_offline")
            self.assertEqual(fixed["question_count"], 1)
            output = run(root / "parsed", gold, root / "reports")
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual({item["chunk_mode"] for item in report["results"]}, {"fixed", "structured"})
            self.assertIn("no embedding", (output / "report.md").read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
