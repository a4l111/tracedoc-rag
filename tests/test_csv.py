"""CSV parsing checks: encoding detection, leading zeros, literal formula text."""

import json
import tempfile
import unittest
from pathlib import Path

from src.parser.pipeline import parse_file, supported_files


class CsvParseTests(unittest.TestCase):
    def test_csv_bom_preserves_leading_zeros_and_formula_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            source = raw / "people.csv"
            source.write_bytes("﻿姓名,编号,备注\r\n甲,007,=1+1\r\n乙,008,普通\n".encode("utf-8"))

            self.assertEqual(supported_files(raw), [source])
            self.assertTrue(parse_file(source, raw, root / "parsed"))
            folder = next((root / "parsed").iterdir())
            markdown = (folder / "document.md").read_text(encoding="utf-8")
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))

            self.assertIn("# CSV 数据：people", markdown)
            self.assertIn("| 姓名 | 编号 | 备注 |", markdown)
            self.assertIn("| 甲 | 007 | =1+1 |", markdown)
            self.assertIn("| 乙 | 008 | 普通 |", markdown)
            self.assertEqual(manifest["csv"]["rows_with_values"], 3)
            self.assertEqual(manifest["csv"]["delimiter"], ",")
            self.assertIn("utf", manifest["encoding"].lower())

    def test_csv_gb18030_auto_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            source = raw / "scores.csv"
            source.write_bytes("姓名,分数\n张三,95\n李四,88\n".encode("gb18030"))

            self.assertTrue(parse_file(source, raw, root / "parsed"))
            folder = next((root / "parsed").iterdir())
            markdown = (folder / "document.md").read_text(encoding="utf-8")
            self.assertIn("张三", markdown)
            self.assertIn("| 95 |", markdown)


if __name__ == "__main__":
    unittest.main()
