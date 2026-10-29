"""Acceptance report detects aligned artifacts and retrieval regression."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.acceptance import INDEX_FILES, REQUIRED_DIRS, REQUIRED_ROOT, run


class AcceptanceTests(unittest.TestCase):
    def test_report_flags_hybrid_regression(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            for name in REQUIRED_ROOT:
                (root / name).write_text("x", encoding="utf-8")
            for name in REQUIRED_DIRS:
                (root / name).mkdir(parents=True)
            parsed = root / "parsed" / "doc"
            parsed.mkdir(parents=True)
            for name in ("document.md", "errors.jsonl"):
                (parsed / name).write_text("", encoding="utf-8")
            (parsed / "manifest.json").write_text(json.dumps({"image_assets": []}), encoding="utf-8")
            snapshot_id = "20261027T000000Z-12345678"
            snapshot = root / "indexes" / snapshot_id
            snapshot.mkdir(parents=True)
            chunks = b'{"chunk_id":"a"}\n'
            for name in INDEX_FILES:
                (snapshot / name).write_bytes(b"x")
            (snapshot / "chunks.jsonl").write_bytes(chunks)
            checksum = hashlib.sha256(chunks).hexdigest()
            (snapshot / "snapshot.json").write_text(json.dumps({"snapshot_id": snapshot_id,
                                                                  "chunks_sha256": checksum}), encoding="utf-8")
            routes = {name: {"indexed_count": 1} for name in ("dense", "bm25", "bge_m3")}
            (snapshot / "build_report.json").write_text(json.dumps({"snapshot_id": snapshot_id,
                "complete": True, "chunk_count": 1, "routes": routes}), encoding="utf-8")
            evaluation = root / "evaluation.json"
            evaluation.write_text(json.dumps({"snapshots": [{"snapshot_id": snapshot_id, "k": 5,
                "question_count": 1, "metrics": {"bm25": {"recall_at_5": .2},
                "dense": {"recall_at_5": .4}, "bge_m3": {"recall_at_5": .3},
                "hybrid_reranked": {"recall_at_5": .1}}}]}), encoding="utf-8")
            folder = run(root, parsed.parent, root / "indexes", snapshot_id, evaluation, root / "acceptance")
            result = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            regression = next(item for item in result["checks"] if item["name"] == "融合质量不退化")
            self.assertEqual(regression["status"], "fail")


if __name__ == "__main__":
    unittest.main()
