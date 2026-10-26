"""CLI orchestration checks without loading models or calling external APIs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from app import app


class CliTests(unittest.TestCase):
    def test_query_answers_by_default_and_has_local_mode(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            snapshot_id = "20261023T000000Z-12345678"
            folder = root / snapshot_id
            folder.mkdir()
            (folder / "snapshot.json").write_text(json.dumps({"sources": [
                {"doc_id": "doc", "source_file": "note.txt"}]}), encoding="utf-8")
            chunk = {"chunk_id": "a", "doc_id": "doc", "page_start": None, "page_end": None,
                     "heading_path": [], "score": 0.8, "text": "测试证据"}
            with patch("src.retrieval.query.query", return_value=[chunk]), \
                 patch("src.qa.answer.answer_question", return_value="回答：\n测试。\n\n引用：\n[1] note.txt") as answer:
                result = CliRunner().invoke(app, ["query", "问题", "--snapshot-id", snapshot_id,
                                                  "--indexes-dir", str(root)])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("回答：", result.output)
                self.assertEqual(answer.call_count, 1)
                local = CliRunner().invoke(app, ["query", "问题", "--snapshot-id", snapshot_id,
                                                 "--indexes-dir", str(root), "--retrieval-only"])
                self.assertEqual(local.exit_code, 0, local.output)
                self.assertNotIn("回答：", local.output)
                self.assertEqual(answer.call_count, 1)

    def test_all_uses_fresh_parse_folder(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            (raw / "note.txt").write_text("测试文本。", encoding="utf-8")
            snapshot = root / "indexes" / "20261023T000000Z-12345678"
            snapshot.mkdir(parents=True)
            (snapshot / "build_report.json").write_text(json.dumps({"complete": True}), encoding="utf-8")
            seen = []

            def fake_build(parsed_dir, indexes_dir, mode):
                seen.append(parsed_dir)
                self.assertEqual(len(list(parsed_dir.glob("*/manifest.json"))), 1)
                return snapshot

            with patch("src.index.build.build", side_effect=fake_build), \
                 patch("src.qa.answer.answer_question", return_value="回答：\n测试。\n\n引用：\n无"):
                result = CliRunner().invoke(app, ["all", "问题", "--raw-dir", str(raw),
                                                  "--parsed-dir", str(root / "parsed"),
                                                  "--indexes-dir", str(root / "indexes")])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0].parent.name, "_runs")
            self.assertIn("回答：", result.output)


if __name__ == "__main__":
    unittest.main()
