"""Cell-level spreadsheet snapshot parsing tests."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from src.parser import spreadsheet


class SpreadsheetTests(unittest.TestCase):
    def test_read_csv_detects_header_and_preserves_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            path.write_text("编号,姓名,分数\n001,张三,90\n002,李四,85\n", encoding="utf-8")
            sheets, warnings = spreadsheet.read_csv(path)
        self.assertEqual(len(sheets), 1)
        sheet = sheets[0]
        self.assertEqual(sheet["name"], "CSV")
        spreadsheet.detect_header(sheet)
        self.assertEqual(sheet["header_row"], 1)
        self.assertEqual(sheet["rows"][1]["cells"][0]["value"], "001")  # leading zero preserved
        self.assertTrue(any("utf-8" in w for w in warnings))

    def test_read_xlsx_reads_cached_values_and_formulas(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.xlsx"
            book = Workbook()
            ws = book.active
            ws.title = "Sheet1"
            ws.append(["姓名", "分数"])
            ws.append(["张三", 90])
            ws.append(["李四", 85])
            ws["C2"] = "=B2*2"
            book.save(path)
            sheets, _ = spreadsheet.read_xlsx(path)
        self.assertEqual(len(sheets), 1)
        sheet = sheets[0]
        spreadsheet.detect_header(sheet)
        self.assertEqual(sheet["header_row"], 1)
        cells = sheet["rows"][1]["cells"]  # row 2: 张三, 90, formula
        self.assertEqual(cells[0]["value"], "张三")
        self.assertEqual(cells[1]["value"], 90)
        self.assertEqual(cells[2]["formula"], "=B2*2")
        self.assertEqual(cells[2]["display"], "公式未计算")

    def test_snapshot_loads_artifact_and_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            parsed = Path(folder) / "parsed" / "doc-1"
            parsed.mkdir(parents=True)
            (parsed / "manifest.json").write_text(json.dumps(
                {"source_file": "a.xlsx", "sha256": "h"}), encoding="utf-8")
            (parsed / "spreadsheet.json").write_text(json.dumps(
                {"version": 1, "source_name": "a.xlsx", "source_hash": "h",
                 "sheets": [{"name": "S", "rows": []}]}, ensure_ascii=False), encoding="utf-8")
            snap = spreadsheet.snapshot(parsed.parent, "doc-1")
        self.assertEqual(snap["name"], "a.xlsx")
        self.assertEqual(snap["hash"], "h")
        self.assertEqual(snap["sheets"][0]["name"], "S")

    def test_encoding_fallback_to_gb18030(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gb.csv"
            path.write_bytes("姓名,分数\n张三,90\n".encode("gb18030"))
            sheets, _ = spreadsheet.read_csv(path)
        self.assertEqual(sheets[0]["rows"][0]["cells"][0]["value"], "姓名")


if __name__ == "__main__":
    unittest.main()
