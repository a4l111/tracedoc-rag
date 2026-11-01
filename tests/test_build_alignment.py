"""A failed embedding must leave one usable, aligned snapshot."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

from src.index import build as builder
from src.retrieval.query import search


class BuildAlignmentTests(unittest.TestCase):
    def test_failed_chunk_is_removed_from_every_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parsed = root / "parsed" / "doc"
            parsed.mkdir(parents=True)
            (parsed / "manifest.json").write_text(json.dumps({
                "doc_id": "doc", "source_file": "doc.txt", "sha256": "abc"}), encoding="utf-8")
            (parsed / "document.md").write_text("# One\nalpha\n# Two\nbeta\n", encoding="utf-8")

            def fake_vector(folder, route, chunks, snapshot_id, checksum, errors, download, progress):
                ids = [chunk["chunk_id"] for chunk in chunks if route != "dense" or "beta" not in chunk["text"]]
                index = faiss.IndexFlatIP(2)
                for position, _ in enumerate(ids):
                    index.add(np.array([[1.0, float(position)]], dtype="float32"))
                faiss.write_index(index, str(folder / f"{route}.faiss"))
                (folder / f"{route}_meta.json").write_text(json.dumps({
                    "snapshot_id": snapshot_id, "chunks_sha256": checksum,
                    "chunk_ids": ids, "model": route, "dimension": 2}), encoding="utf-8")
                for chunk in chunks:
                    if chunk["chunk_id"] not in ids:
                        errors.append({"stage": route, "chunk_id": chunk["chunk_id"], "type": "ValueError"})
                return {"model": route, "indexed_count": len(ids), "failed_count": len(chunks) - len(ids)}

            with patch.object(builder.moark, "client"), patch.object(builder, "_build_vector_index", side_effect=fake_vector), \
                 patch.object(builder, "MODELS", {"dense": ("dense", 2, "dense.faiss", "dense_meta.json"),
                                                  "bge_m3": ("bge", 2, "bge_m3.faiss", "bge_m3_meta.json")}):
                folder = builder.build(root / "parsed", root / "indexes")

            report = json.loads((folder / "build_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["complete"])
            self.assertTrue(report["partial"])
            self.assertEqual(report["attempted_chunk_count"], 2)
            self.assertEqual(report["chunk_count"], 1)
            chunks = [json.loads(line) for line in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
            expected_ids = [chunk["chunk_id"] for chunk in chunks]
            self.assertEqual(len(expected_ids), 1)
            for route, filename in (("dense", "dense"), ("bge_m3", "bge_m3")):
                meta = json.loads((folder / f"{filename}_meta.json").read_text(encoding="utf-8"))
                self.assertEqual(meta["chunk_ids"], expected_ids)
                self.assertEqual(faiss.read_index(str(folder / f"{route}.faiss")).ntotal, 1)
            with (folder / "bm25.pkl").open("rb") as stream:
                self.assertEqual(pickle.load(stream)["chunk_ids"], expected_ids)
            self.assertEqual(search(root / "indexes", folder.name, "bm25", "alpha")[0]["chunk_id"],
                             expected_ids[0])
            repairs = [json.loads(line) for line in (folder / "repair_log.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(repairs[0]["failed_routes"], ["dense"])
            self.assertEqual(repairs[0]["action"], "excluded_from_all_indexes")


if __name__ == "__main__":
    unittest.main()
