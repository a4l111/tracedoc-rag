import json
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from src.parser.pipeline import parse_file, supported_files


class XlsxParseTests(unittest.TestCase):
    def test_sheet_and_row_provenance_without_formula_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            source = raw / "scores.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "成绩"
            sheet.append(["姓名", "分数"])
            sheet.append(["甲", 95])
            sheet.append(["乙", "=1+1"])
            workbook.save(source)

            self.assertEqual(supported_files(raw), [source])
            self.assertTrue(parse_file(source, raw, root / "parsed"))
            folder = next((root / "parsed").iterdir())
            markdown = (folder / "document.md").read_text(encoding="utf-8")
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("# 工作表：成绩", markdown)
            self.assertIn("行 2：甲 | 95", markdown)
            self.assertNotIn("=1+1", markdown)
            self.assertEqual(manifest["sheets"][0]["rows_with_values"], 3)
            self.assertEqual(manifest["page_map"], [])


if __name__ == "__main__":
    unittest.main()
