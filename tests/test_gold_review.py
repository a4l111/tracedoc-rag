"""Auditable gold review workflow tests."""

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.gold_review import _validate_reviews, prepare


class GoldReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gold = [{"benchmark_row": 10, "question": "Q?", "source_doc": "doc.pdf", "answer": "A",
                      "relevant": [{"doc_id": "doc", "page": 3}]}]

    def test_confirmed_replace_is_validated_by_benchmark_row(self) -> None:
        review = {"benchmark_row": 10, "question_number": 1, "question": "Q?", "source_doc": "doc.pdf", "answer": "A",
                  "original_pages": [3], "reviewed_pages": [2], "decision": "replace",
                  "status": "confirmed", "reviewer": "r", "evidence": "checked",
                  "visual_description_reviewed": False}
        self.assertEqual(_validate_reviews(self.gold, [review])[10]["reviewed_pages"], [2])

    def test_keep_cannot_change_pages(self) -> None:
        review = {"benchmark_row": 10, "question_number": 1, "question": "Q?", "source_doc": "doc.pdf", "answer": "A",
                  "original_pages": [3], "reviewed_pages": [2], "decision": "keep",
                  "status": "confirmed", "reviewer": "r", "evidence": "checked",
                  "visual_description_reviewed": False}
        with self.assertRaisesRegex(ValueError, "keep must preserve"):
            _validate_reviews(self.gold, [review])

    def test_prepare_writes_separate_uncertain_template(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            gold_path = root / "gold.jsonl"
            gold_path.write_text(json.dumps(self.gold[0]) + "\n", encoding="utf-8")
            output = prepare(gold_path, [1], root / "review.jsonl")
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["decision"], "uncertain")
            self.assertEqual(record["original_pages"], [3])

    def test_prepare_refuses_to_overwrite_review(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            gold_path = root / "gold.jsonl"
            gold_path.write_text(json.dumps(self.gold[0]) + "\n", encoding="utf-8")
            output = root / "review.jsonl"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                prepare(gold_path, [1], output)


if __name__ == "__main__":
    unittest.main()
