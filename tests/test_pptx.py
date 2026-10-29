"""PPTX parsing and slide provenance checks using synthetic OOXML."""

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image

from src.chunker.chunks import make_chunks
from src.parser.pipeline import parse_file
from src.qa.answer import _location


class PptxTests(unittest.TestCase):
    def test_slide_order_image_notes_and_chunk_location(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            picture = BytesIO()
            Image.new("RGB", (8, 8), "red").save(picture, format="PNG")
            p = "http://schemas.openxmlformats.org/presentationml/2006/main"
            a = "http://schemas.openxmlformats.org/drawingml/2006/main"
            r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            rel = "http://schemas.openxmlformats.org/package/2006/relationships"
            slide = lambda title: (f'<p:sld xmlns:p="{p}" xmlns:a="{a}" xmlns:r="{r}"><p:cSld><p:spTree>'
                                   f'<p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>'
                                   f'<p:txBody><a:p><a:r><a:t>{title}</a:t></a:r></a:p></p:txBody></p:sp>'
                                   '</p:spTree></p:cSld></p:sld>')
            second = slide("Second")
            second = second.replace('</p:spTree>',
                                    '<p:pic><p:blipFill><a:blip r:embed="rIdImage"/></p:blipFill></p:pic>'
                                    '<p:grpSp><p:sp><p:txBody><a:p><a:r><a:t>Grouped detail</a:t>'
                                    '</a:r></a:p></p:txBody></p:sp></p:grpSp>'
                                    '</p:spTree>')
            with ZipFile(raw / "sample.pptx", "w") as archive:
                archive.writestr("ppt/presentation.xml", f'<p:presentation xmlns:p="{p}" xmlns:r="{r}">'
                                 '<p:sldIdLst><p:sldId r:id="rId2"/><p:sldId r:id="rId1"/></p:sldIdLst>'
                                 '</p:presentation>')
                archive.writestr("ppt/_rels/presentation.xml.rels", f'<Relationships xmlns="{rel}">'
                                 '<Relationship Id="rId1" Type="slide" Target="slides/slide1.xml"/>'
                                 '<Relationship Id="rId2" Type="slide" Target="slides/slide2.xml"/>'
                                 '</Relationships>')
                archive.writestr("ppt/slides/slide1.xml", slide("First"))
                archive.writestr("ppt/slides/slide2.xml", second)
                archive.writestr("ppt/slides/_rels/slide2.xml.rels", f'<Relationships xmlns="{rel}">'
                                 '<Relationship Id="rIdImage" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image.png"/>'
                                 '<Relationship Id="rIdNotes" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide2.xml"/>'
                                 '</Relationships>')
                archive.writestr("ppt/notesSlides/notesSlide2.xml", f'<p:notes xmlns:p="{p}" xmlns:a="{a}">'
                                 '<p:cSld><p:spTree><p:sp><p:nvSpPr><p:nvPr><p:ph type="body"/>'
                                 '</p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Speaker detail</a:t>'
                                 '</a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:notes>')
                archive.writestr("ppt/media/image.png", picture.getvalue())
            with patch("src.parser.pipeline._ocr_text", return_value="Diagram OCR"):
                self.assertTrue(parse_file(raw / "sample.pptx", raw, root / "parsed"))
            folder = next((root / "parsed").iterdir())
            markdown = (folder / "document.md").read_text(encoding="utf-8")
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            self.assertLess(markdown.index("Second"), markdown.index("First"))
            self.assertIn("<!-- SLIDE:1 -->", markdown)
            self.assertIn("Speaker detail", markdown)
            self.assertIn("Diagram OCR", markdown)
            self.assertIn("Grouped detail", markdown)
            self.assertEqual(manifest["slide_count"], 2)
            self.assertEqual(manifest["page_map"], [])
            self.assertEqual(manifest["image_assets"][0]["slide"], 1)
            self.assertIn("Second", manifest["image_assets"][0]["nearby_text"])
            self.assertEqual(manifest["image_assets"][0]["source_doc_id"], manifest["doc_id"])
            self.assertTrue((folder / manifest["image_assets"][0]["path"]).is_file())
            chunks = make_chunks(markdown, manifest, "structured")
            matching = next(chunk for chunk in chunks if "Second" in chunk["text"])
            self.assertEqual((matching["slide_start"], matching["slide_end"]), (1, 1))
            self.assertIsNone(matching["page_start"])
            self.assertEqual(_location(matching)[1], "第1张幻灯片")


if __name__ == "__main__":
    unittest.main()
