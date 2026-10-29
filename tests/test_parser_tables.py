"""Synthetic parsing checks; no user documents are read or changed."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import fitz
from PIL import Image
from io import BytesIO

from src.parser.pipeline import parse_file
from src.parser.tables import docx_table


class TableParsingTests(unittest.TestCase):
    def test_image_only_pdf_ocr_is_opt_in_and_marked_unreviewed(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            picture = BytesIO()
            Image.new("RGB", (100, 50), "white").save(picture, format="PNG")
            document = fitz.open()
            page = document.new_page()
            page.insert_image(fitz.Rect(72, 72, 172, 122), stream=picture.getvalue())
            document.save(raw / "scan.pdf")
            document.close()

            with patch("src.parser.pipeline._ocr_text", return_value="Scanned heading") as ocr:
                self.assertTrue(parse_file(raw / "scan.pdf", raw, root / "parsed", ocr_scanned_pages=True))
            self.assertEqual(ocr.call_count, 1)
            folder = next((root / "parsed").iterdir())
            markdown = (folder / "document.md").read_text(encoding="utf-8")
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("<!-- PAGE:1 -->", markdown)
            self.assertIn("![PDF page 1 OCR source](images/page-0001.png)", markdown)
            self.assertIn("OCR:local page=1 review_status=unreviewed", markdown)
            self.assertIn("Scanned heading", markdown)
            self.assertEqual(manifest["image_assets"][-1]["ocr_character_count"], len("Scanned heading"))
            self.assertTrue((folder / manifest["image_assets"][-1]["path"]).is_file())

    def test_pdf_image_asset_has_real_page_and_nearby_text(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            picture = BytesIO()
            Image.new("RGB", (10, 10), "blue").save(picture, format="PNG")
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 70), "Figure explanation")
            page.insert_image(fitz.Rect(72, 90, 120, 138), stream=picture.getvalue())
            document.save(raw / "figure.pdf")
            document.close()
            self.assertTrue(parse_file(raw / "figure.pdf", raw, root / "parsed"))
            folder = next((root / "parsed").iterdir())
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            asset = manifest["image_assets"][0]
            self.assertEqual(asset["page"], 1)
            self.assertEqual(asset["source_doc_id"], manifest["doc_id"])
            self.assertEqual(asset["nearby_text"], "Figure explanation")

    def test_docx_table_formats(self) -> None:
        plain = [[{"text": value, "colspan": 1, "vmerge": None} for value in row]
                 for row in (("A", "B"), ("1", "2"))]
        self.assertIn("| A | B |", docx_table(plain))
        self.assertIn('colspan="2"', docx_table([[{"text": "Merged", "colspan": 2, "vmerge": None}]]))
        vertical = [[{"text": "Merged", "colspan": 1, "vmerge": "restart"}],
                    [{"text": "", "colspan": 1, "vmerge": "continue"}]]
        self.assertIn('rowspan="2"', docx_table(vertical))

    def test_pdf_table_and_page_marker(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 60), "Before table")
            x_positions = [72, 180, 288]
            y_positions = [100, 130, 160, 190]
            for x in x_positions:
                page.draw_line((x, y_positions[0]), (x, y_positions[-1]))
            for y in y_positions:
                page.draw_line((x_positions[0], y), (x_positions[-1], y))
            for y, values in zip([120, 150, 180], [("Name", "Count"), ("Alpha", "1"), ("Beta", "2")]):
                for x, value in zip([85, 193], values):
                    page.insert_text((x, y), value)
            page.insert_text((72, 220), "After table")
            document.save(raw / "sample.pdf")
            document.close()
            self.assertTrue(parse_file(raw / "sample.pdf", raw, root / "parsed"))
            folder = next((root / "parsed").iterdir())
            markdown = (folder / "document.md").read_text(encoding="utf-8")
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("<!-- PAGE:1 -->", markdown)
            self.assertIn("| Name | Count |", markdown)
            self.assertEqual(manifest["table_count"], 1)

    def test_docx_table_order_and_paragraph_ids(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            word_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            paragraph = lambda value: f"<w:p><w:r><w:t>{value}</w:t></w:r></w:p>"
            row = lambda a, b: f"<w:tr><w:tc>{paragraph(a)}</w:tc><w:tc>{paragraph(b)}</w:tc></w:tr>"
            document_xml = (f'<w:document xmlns:w="{word_ns}"><w:body>' + paragraph("Before")
                            + "<w:tbl>" + row("A", "B") + row("1", "2") + "</w:tbl>"
                            + paragraph("After") + "</w:body></w:document>")
            with ZipFile(raw / "sample.docx", "w") as archive:
                archive.writestr("word/document.xml", document_xml)
                archive.writestr("word/_rels/document.xml.rels", "<Relationships/>")
            self.assertTrue(parse_file(raw / "sample.docx", raw, root / "parsed"))
            folder = next((root / "parsed").iterdir())
            markdown = (folder / "document.md").read_text(encoding="utf-8")
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            self.assertLess(markdown.index("Before"), markdown.index("| A | B |"))
            self.assertLess(markdown.index("| A | B |"), markdown.index("After"))
            self.assertEqual(manifest["paragraph_count"], 6)
            self.assertEqual(manifest["table_count"], 1)
            self.assertEqual(len(manifest["paragraphs"]), 6)

    def test_docx_heading_style_id_from_styles_xml(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            word_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            document_xml = (f'<w:document xmlns:w="{word_ns}"><w:body><w:p>'
                            '<w:pPr><w:pStyle w:val="2"/></w:pPr><w:r><w:t>Chapter</w:t></w:r>'
                            '</w:p></w:body></w:document>')
            styles_xml = (f'<w:styles xmlns:w="{word_ns}"><w:style w:styleId="2">'
                          '<w:name w:val="heading 1"/><w:pPr><w:outlineLvl w:val="0"/>'
                          '</w:pPr></w:style></w:styles>')
            with ZipFile(raw / "heading.docx", "w") as archive:
                archive.writestr("word/document.xml", document_xml)
                archive.writestr("word/styles.xml", styles_xml)
                archive.writestr("word/_rels/document.xml.rels", "<Relationships/>")
            self.assertTrue(parse_file(raw / "heading.docx", raw, root / "parsed"))
            folder = next((root / "parsed").iterdir())
            self.assertIn("# Chapter", (folder / "document.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
