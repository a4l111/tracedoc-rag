"""Visual enrichment checks without external calls."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import fitz
from PIL import Image

from src.parser.pipeline import _doc_id, parse_file
from src.parser.vision import enrich_assets, enrich_mmlongbench


class _Completions:
    def create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="页面包含一个柱状图，标题为测试图表。"))])


class _Client:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_Completions())


class VisionTests(unittest.TestCase):
    def test_general_asset_enrichment_is_traceable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (4, 4), "white").save(images / "source.png")
            manifest = {"errors": [], "image_assets": [{"asset_id": "source", "path": "images/source.png",
                                                           "kind": "source", "nearby_text": ""}]}
            markdown = enrich_assets("![source](images/source.png)\n", root, manifest, client=_Client())
            self.assertIn("review_status=unreviewed asset_id=source", markdown)
            self.assertIn("视觉模型描述（未经人工核验）", markdown)
            self.assertFalse(manifest["image_assets"][0]["vision"]["raw_response_persisted"])

    def test_enrichment_is_separate_and_traceable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            documents = root / "benchmark" / "data" / "documents"
            documents.mkdir(parents=True)
            source = documents / "sample.pdf"
            pdf = fitz.open()
            pdf.new_page().insert_text((40, 40), "Chart source text")
            pdf.save(source)
            parsed = root / "parsed"
            self.assertTrue(parse_file(source, documents, parsed))
            doc_id = _doc_id(source, documents)
            gold = root / "gold.jsonl"
            gold.write_text(json.dumps({"question": "chart?", "relevant": [
                {"doc_id": doc_id, "page": 1}]}) + "\n", encoding="utf-8")
            output = root / "vision"
            report = enrich_mmlongbench(root / "benchmark", parsed, gold, output, client=_Client())
            self.assertEqual(report["described_pages"], 1)
            original = (parsed / doc_id / "document.md").read_text(encoding="utf-8")
            enriched = (output / doc_id / "document.md").read_text(encoding="utf-8")
            self.assertNotIn("视觉模型描述", original)
            self.assertIn("视觉模型描述（未经人工核验）", enriched)
            manifest = json.loads((output / doc_id / "manifest.json").read_text(encoding="utf-8"))
            vision = manifest["image_assets"][-1]["vision"]
            self.assertFalse(vision["raw_response_persisted"])
            self.assertEqual(vision["review_status"], "unreviewed")

    def test_all_pages_does_not_need_gold_locations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            documents = root / "benchmark" / "data" / "documents"
            documents.mkdir(parents=True)
            source = documents / "sample.pdf"
            pdf = fitz.open()
            pdf.new_page().insert_text((40, 40), "First page")
            pdf.new_page().insert_text((40, 40), "Second page")
            pdf.save(source)
            pdf.close()
            parsed = root / "parsed"
            self.assertTrue(parse_file(source, documents, parsed))
            output = root / "vision_all"
            report = enrich_mmlongbench(root / "benchmark", parsed, None, output,
                                        client=_Client(), all_pages=True)
            self.assertEqual(report["selection"], "all_pages")
            self.assertEqual(report["requested_pages"], 2)
            self.assertEqual(report["described_pages"], 2)
            markdown = next(output.glob("*/document.md")).read_text(encoding="utf-8")
            self.assertIn("asset_id=page-0001-vision-render", markdown)
            self.assertIn("asset_id=page-0002-vision-render", markdown)


if __name__ == "__main__":
    unittest.main()
