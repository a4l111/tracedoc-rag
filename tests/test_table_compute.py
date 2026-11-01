"""Deterministic table-compute contract tests without model calls."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from src.qa import table_compute


def _cell(column, value, display=None, kind="s"):
    return {"column": column, "address": f"{column}{column}", "value": value,
            "display": display if display is not None else str(value), "format": "General",
            "formula": "", "type": "n" if kind == "n" else "s"}


def _make_sheet():
    header = {"number": 1, "cells": [_cell(1, "姓名"), _cell(2, "类别"), _cell(3, "分数")]}
    rows = [
        {"number": 2, "cells": [_cell(1, "张三"), _cell(2, "男"), _cell(3, 90, kind="n")]},
        {"number": 3, "cells": [_cell(1, "李四"), _cell(2, "女"), _cell(3, 85, kind="n")]},
        {"number": 4, "cells": [_cell(1, "王五"), _cell(2, "男"), _cell(3, 75, kind="n")]},
        {"number": 5, "cells": [_cell(1, "张三"), _cell(2, "男"), _cell(3, 80, kind="n")]},
    ]
    return {"name": "Sheet1", "max_row": 5, "rows": [header, *rows], "header_row": 1}


class TableComputeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.parsed = Path(self.temp.name) / "parsed"
        folder = self.parsed / "doc-1"
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps({"source_file": "data.xlsx", "sha256": "h"}),
                                              encoding="utf-8")
        (folder / "spreadsheet.json").write_text(json.dumps(
            {"version": 1, "source_name": "data.xlsx", "source_hash": "h", "sheets": [_make_sheet()]},
            ensure_ascii=False), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def _execute(self, question, spec):
        doc = table_compute._load_doc(self.parsed, "doc-1")
        payload = {"version": 1, "doc_id": "doc-1", "name": doc["name"], "sheet": "Sheet1",
                   "header_row": 1, "max_row": 5, "source_hash": doc["hash"],
                   "snapshot_hash": table_compute._snapshot_hash(doc), "question": question,
                   "spec": spec, "issued_at": int(time.time())}
        return table_compute.execute(self.parsed, payload, table_compute._signed(payload))

    def _spec(self, **overrides):
        spec = {"operation": "count", "conditions": [], "logic": "and", "target": "",
                "group": "", "identifier": "", "limit": None, "metrics": []}
        spec.update(overrides)
        return spec

    def test_count_sum_avg_min_max(self):
        self.assertEqual(self._execute("共有几条记录", self._spec())["result"], 4)
        self.assertEqual(self._execute("分数总和是多少", self._spec(operation="sum", target="C"))["result"], "330")
        self.assertEqual(self._execute("平均分是多少", self._spec(operation="avg", target="C"))["result"], "82.5")
        self.assertEqual(self._execute("最低分", self._spec(operation="min", target="C"))["result"], "75")
        self.assertEqual(self._execute("最高分", self._spec(operation="max", target="C"))["result"], "90")

    def test_group_count_and_distinct(self):
        report = self._execute("按类别统计记录数", self._spec(operation="group_count", group="B"))
        self.assertEqual(sorted((g["group"], g["count"]) for g in report["result"]), [("女", 1), ("男", 3)])
        distinct = self._execute("按姓名去重计数", self._spec(operation="distinct_count", target="A"))
        self.assertEqual(distinct["result"], 3)
        self.assertEqual(distinct["duplicates_removed"], 1)

    def test_top_returns_descending_rows(self):
        report = self._execute("分数最高的前2条", self._spec(operation="top", target="C", identifier="A", limit=2))
        self.assertEqual(report["result"], [{"row": 2, "identifier": "张三", "value": "90"},
                                            {"row": 3, "identifier": "李四", "value": "85"}])

    def test_tampered_signature_is_rejected(self):
        doc = table_compute._load_doc(self.parsed, "doc-1")
        payload = {"version": 1, "doc_id": "doc-1", "name": doc["name"], "sheet": "Sheet1", "header_row": 1,
                   "max_row": 5, "source_hash": doc["hash"], "snapshot_hash": table_compute._snapshot_hash(doc),
                   "question": "共有几条记录", "spec": self._spec(), "issued_at": int(time.time())}
        signature = table_compute._signed(payload)
        payload["spec"]["operation"] = "sum"  # tamper after signing
        with self.assertRaises(ValueError):
            table_compute.execute(self.parsed, payload, signature)

    def test_unknown_column_is_rejected(self):
        with self.assertRaises(ValueError):
            self._execute("X 列求和", self._spec(operation="sum", target="Z"))


if __name__ == "__main__":
    unittest.main()
