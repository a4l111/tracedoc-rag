"""Unit tests for the independent image-vector index."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

from src.index import visual


def _write_doc(parsed: Path, doc_id: str, source_file: str, images: list[dict]):
    folder = parsed / doc_id
    (folder / "images").mkdir(parents=True, exist_ok=True)
    assets = []
    for asset in images:
        (folder / asset["path"]).write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
        assets.append({"asset_id": asset.get("asset_id", "a"), "path": asset["path"],
                       "page": asset.get("page"), "slide": asset.get("slide"),
                       "kind": asset.get("kind", "embedded")})
    (folder / "manifest.json").write_text(json.dumps(
        {"doc_id": doc_id, "source_file": source_file, "image_assets": assets}), encoding="utf-8")


def _vector(dim, value=1.0):
    vec = np.zeros(dim, dtype="float32")
    vec[0] = value
    return vec / np.linalg.norm(vec)


class VisualIndexTests(unittest.TestCase):
    def test_collect_images_deduplicates_and_keeps_location(self):
        with tempfile.TemporaryDirectory() as folder:
            parsed = Path(folder) / "parsed"
            _write_doc(parsed, "doc-1", "a.pdf", [{"path": "images/x.png", "page": 3}])
            _write_doc(parsed, "doc-2", "b.pptx", [{"path": "images/y.png", "slide": 2}])
            images = visual.collect_images(parsed)
        self.assertEqual(len(images), 2)
        by_doc = {i["doc_id"]: i for i in images}
        self.assertEqual(by_doc["doc-1"]["page"], 3)
        self.assertEqual(by_doc["doc-2"]["slide"], 2)

    def test_build_and_search_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parsed = root / "parsed"
            _write_doc(parsed, "doc-1", "a.pdf", [{"path": "images/x.png", "page": 3}])
            snapshot = root / "indexes" / "20261101T000000Z-12345678"
            snapshot.mkdir(parents=True)
            with patch.object(visual, "visual_client", return_value=object()), \
                 patch.object(visual, "embed_image", return_value=_vector(visual.DIMENSION)), \
                 patch.object(visual, "load_settings", return_value={
                     "visual_embedding": {"model": visual.VISUAL_MODEL, "dimensions": visual.DIMENSION,
                                          "base_url": "https://api.moark.com/v1"}}):
                report = visual.build_visual_index(parsed, snapshot)
            self.assertEqual(report["indexed"], 1)
            self.assertFalse(report["failed"])
            self.assertTrue((snapshot / "visual_index.faiss").is_file())
            self.assertEqual(faiss.read_index(str(snapshot / "visual_index.faiss")).ntotal, 1)

            with patch.object(visual, "visual_client", return_value=object()), \
                 patch.object(visual, "embed_text", return_value=_vector(visual.DIMENSION)), \
                 patch.object(visual, "load_settings", return_value={
                     "visual_embedding": {"model": visual.VISUAL_MODEL, "dimensions": visual.DIMENSION,
                                          "base_url": "https://api.moark.com/v1"}}):
                hits = visual.visual_search(root / "indexes", snapshot.name, "问题")
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["doc_id"], "doc-1")
            self.assertEqual(hits[0]["page"], 3)

    def test_search_requires_existing_index(self):
        with tempfile.TemporaryDirectory() as folder:
            indexes = Path(folder) / "indexes"
            (indexes / "20261101T000000Z-12345678").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                visual.visual_search(indexes, "20261101T000000Z-12345678", "问题")


if __name__ == "__main__":
    unittest.main()
