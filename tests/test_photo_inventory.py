"""Photo inventory keeps observations on their source pages."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from src.parser.photo_inventory import inventory_parsed
from src.index.build import _load_chunks
from src.evaluation.metrics import _relevant_ids
from src.qa.answer import _location


class _Client:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"photos":[{"people_count":1,"estimated":false,"description":"portrait"}]}'
        ))])


class PhotoInventoryTests(unittest.TestCase):
    def test_copy_and_page_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "parsed" / "doc"
            (source / "images").mkdir(parents=True)
            Image.new("RGB", (8, 8), "white").save(source / "images" / "page.png")
            (source / "document.md").write_text("<!-- PAGE:1 -->\nText\n", encoding="utf-8")
            (source / "manifest.json").write_text(json.dumps({
                "doc_id": "doc", "sha256": "source-hash", "source_file": "sample.pdf",
                "image_assets": [{"asset_id": "page-0001-vision-render",
                "kind": "page_render_for_vision", "path": "images/page.png", "page": 1}]
            }), encoding="utf-8")
            result = inventory_parsed(root / "parsed", root / "out", client=_Client())
            self.assertEqual(result["completed_pages"], 1)
            self.assertNotIn("Photo inventory", (source / "document.md").read_text(encoding="utf-8"))
            text = (root / "out" / "doc" / "document.md").read_text(encoding="utf-8")
            self.assertEqual(text, (source / "document.md").read_text(encoding="utf-8"))
            manifest = json.loads((root / "out" / "doc" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["photo_inventory"][0]["asset_id"], "page-0001-vision-render")
            self.assertEqual(manifest["photo_inventory"][0]["review_status"], "unreviewed")
            chunks, _, errors = _load_chunks(root / "out", "structured")
            self.assertFalse(errors)
            photo_chunks = [c for c in chunks if c["heading_path"] == ["Photo inventory"]]
            self.assertEqual(len(photo_chunks), 1)
            self.assertEqual(photo_chunks[0]["page_start"], 1)
            self.assertEqual(photo_chunks[0]["image_asset_id"], "page-0001-vision-render")
            summaries = [c for c in chunks if c["heading_path"] == ["Photo inventory", "Single-person photos"]]
            self.assertEqual(summaries[0]["evidence_pages"], [1])
            self.assertIn(summaries[0]["chunk_id"], _relevant_ids(chunks, [{"doc_id": "doc", "page": 1}]))
            self.assertIn("第1页", _location(summaries[0])[1])

    def test_unknown_photo_suppresses_document_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "doc"
            root.mkdir(parents=True)
            (root / "document.md").write_text("<!-- PAGE:1 -->\nText\n", encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({
                "doc_id": "doc", "source_file": "sample.pdf", "sha256": "source-hash",
                "image_assets": [{"kind": "page_render_for_vision"}],
                "photo_inventory": [{"page": 1, "asset_id": "render", "photos": [
                    {"photo_number": 1, "people_count": None, "estimated": False}]}]
            }), encoding="utf-8")
            chunks, _, errors = _load_chunks(root.parent, "structured")
            self.assertFalse(errors)
            self.assertFalse(any(len(c["heading_path"]) == 2 and
                                 c["heading_path"][0] == "Photo inventory" for c in chunks))


if __name__ == "__main__":
    unittest.main()
