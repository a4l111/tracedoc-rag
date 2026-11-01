"""Local benchmark import checks."""

import json
import tempfile
import unittest
from pathlib import Path

import fitz

from src.evaluation.mmlongbench import prepare


class MMLongBenchTests(unittest.TestCase):
    def test_one_based_page_and_unanswerable_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            documents = root / "dataset" / "data" / "documents"
            documents.mkdir(parents=True)
            pdf = fitz.open()
            pdf.new_page().insert_text((40, 40), "A relevant fact")
            pdf.save(documents / "a.pdf")
            (documents.parent / "samples.json").write_text(json.dumps([
                {"doc_id": "a.pdf", "question": "What fact?", "answer": "A relevant fact",
                 "evidence_pages": "[1]", "evidence_sources": "['Pure-text']"},
                {"doc_id": "a.pdf", "question": "What is missing?", "answer": "Not answerable",
                 "evidence_pages": "[]", "evidence_sources": "[]"},
                {"doc_id": "a.pdf", "question": "Invalid page?", "answer": "No",
                 "evidence_pages": "[0]", "evidence_sources": "['Pure-text']"},
            ]), encoding="utf-8")
            result = prepare(root / "dataset", root / "output", 1)
            self.assertEqual(result["answerable_questions"], 1)
            self.assertEqual(result["unanswerable_questions"], 1)
            self.assertEqual(len(result["invalid_labels"]), 1)
            self.assertIn("cannot be mapped", result["invalid_labels"][0]["reason"])
            gold = json.loads((root / "output" / "gold.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(gold["relevant"][0]["page"], 1)
            self.assertEqual(gold["evidence_pages_raw"], "[1]")
            self.assertIn("retained unchanged", result["page_rule"])
            self.assertEqual(result["parsed_documents"], 1)


if __name__ == "__main__":
    unittest.main()
