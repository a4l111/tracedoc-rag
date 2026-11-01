"""HTTP contract tests for the table browse and bounded compute endpoints."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.qa import table_compute
from src.service import web


def _cell(column, value, display=None, kind="s"):
    return {"column": column, "address": f"{column}1", "value": value,
            "display": display if display is not None else str(value), "format": "General",
            "formula": "", "type": "n" if kind == "n" else "s"}


def _write_spreadsheet(parsed: Path, doc_id: str):
    folder = parsed / doc_id
    folder.mkdir(parents=True)
    header = {"number": 1, "cells": [_cell(1, "姓名"), _cell(2, "分数")]}
    rows = [header,
            {"number": 2, "cells": [_cell(1, "张三"), _cell(2, 90, kind="n")]},
            {"number": 3, "cells": [_cell(1, "李四"), _cell(2, 85, kind="n")]}]
    (folder / "manifest.json").write_text(json.dumps(
        {"doc_id": doc_id, "source_file": "data.xlsx", "source_type": "xlsx", "sha256": "h"}), encoding="utf-8")
    (folder / "spreadsheet.json").write_text(json.dumps(
        {"version": 1, "source_name": "data.xlsx", "source_hash": "h",
         "sheets": [{"name": "Sheet1", "max_row": 3, "rows": rows, "header_row": 1}]},
        ensure_ascii=False), encoding="utf-8")


class TableServiceTests(unittest.TestCase):
    def _client(self, parsed: Path):
        return TestClient(web.app)

    def test_spreadsheet_browse_404_and_200(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parsed = root / "parsed"
            with patch.object(web, "PARSED", parsed), patch.object(web, "SERVICE", root / "service"), \
                 patch.object(web, "EVENTS", root / "service" / "events.jsonl"):
                with TestClient(web.app) as client:
                    self.assertEqual(client.get("/api/source/doc-1/spreadsheet").status_code, 404)
                    _write_spreadsheet(parsed, "doc-1")
                    response = client.get("/api/source/doc-1/spreadsheet")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["sheets"][0]["name"], "Sheet1")
                    self.assertEqual(response.json()["sheets"][0]["row_count"], 3)

    def test_table_execute_returns_computed_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parsed = root / "parsed"
            _write_spreadsheet(parsed, "doc-1")
            doc = table_compute._load_doc(parsed, "doc-1")
            payload = {"version": 1, "doc_id": "doc-1", "name": doc["name"], "sheet": "Sheet1",
                       "header_row": 1, "max_row": 3, "source_hash": doc["hash"],
                       "snapshot_hash": table_compute._snapshot_hash(doc), "question": "分数总和",
                       "spec": {"operation": "sum", "conditions": [], "logic": "and", "target": "B",
                                "group": "", "identifier": "", "limit": None, "metrics": []},
                       "issued_at": int(time.time())}
            signature = table_compute._signed(payload)
            with patch.object(web, "PARSED", parsed), patch.object(web, "SERVICE", root / "service"), \
                 patch.object(web, "EVENTS", root / "service" / "events.jsonl"):
                with TestClient(web.app) as client:
                    response = client.post("/api/table/execute",
                                           json={"plan": payload, "signature": signature})
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("175", response.json()["answer"])
                    self.assertEqual(response.json()["report"]["included"], 2)

    def test_table_plan_rejects_unsupported_computation_without_model_call(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parsed = root / "parsed"
            _write_spreadsheet(parsed, "doc-1")
            with patch.object(web, "PARSED", parsed), patch.object(web, "SERVICE", root / "service"), \
                 patch.object(web, "EVENTS", root / "service" / "events.jsonl"):
                with TestClient(web.app) as client:
                    response = client.post("/api/table/plan", json={"question": "占比是多少", "doc_id": "doc-1"})
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("复杂计算", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
