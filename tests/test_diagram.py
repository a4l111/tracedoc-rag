"""Unit tests for restricted graph validation and the diagram endpoint contract."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.qa import diagram
from src.service import web


class GraphDescriptionTests(unittest.TestCase):
    def test_valid_graph_emits_mermaid_with_escaped_label(self):
        raw = json.dumps({
            "description": "流程说明",
            "nodes": [{"id": "a", "label": "开始"}],
            "edges": [],
        }, ensure_ascii=False)
        text, meta = diagram.graph_description(raw)
        self.assertTrue(meta["structure_checked"])
        self.assertIn("```mermaid", text)
        self.assertIn("flowchart TD", text)
        self.assertIn('n0["开始"]', text)

    def test_non_json_passes_through_unchanged(self):
        text, meta = diagram.graph_description("这是一段普通文字")
        self.assertEqual(text, "这是一段普通文字")
        self.assertIsNone(meta)

    def test_escapes_mermaid_entity_characters(self):
        raw = json.dumps({
            "description": "d",
            "nodes": [{"id": "a", "label": 'x"y'}],
            "edges": [],
        })
        text, _ = diagram.graph_description(raw)
        self.assertNotIn('x"y', text.split("```mermaid")[1])

    def test_duplicate_node_id_rejected(self):
        raw = json.dumps({"description": "d",
                          "nodes": [{"id": "a", "label": "一"}, {"id": "a", "label": "二"}],
                          "edges": []})
        with self.assertRaises(ValueError):
            diagram.graph_description(raw)

    def test_edge_to_missing_node_rejected(self):
        raw = json.dumps({"description": "d",
                          "nodes": [{"id": "a", "label": "一"}],
                          "edges": [{"source": "a", "target": "b"}]})
        with self.assertRaises(ValueError):
            diagram.graph_description(raw)

    def test_missing_description_rejected(self):
        raw = json.dumps({"nodes": [], "edges": []})
        with self.assertRaises(ValueError):
            diagram.graph_description(raw)


class DiagramEndpointTests(unittest.TestCase):
    def test_endpoint_returns_validated_result_and_errors_without_model_call(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = {"answer": "```mermaid\nflowchart TD\n```", "mermaid": "flowchart TD",
                      "model_generated": True, "structure_checked": True, "human_verified": False}
            with patch.object(web, "SERVICE", root / "service"), \
                 patch.object(web, "EVENTS", root / "service" / "events.jsonl"), \
                 patch.object(web, "INDEXES", root / "indexes"), \
                 patch("src.service.web.latest_complete_snapshot", return_value="snapshot-1"), \
                 patch("src.service.web.diagram.generate_diagram", return_value=result) as gen:
                with TestClient(web.app) as client:
                    response = client.post("/api/diagram", json={"question": "画出流程", "snapshot_id": "snapshot-1"})
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(response.json()["model_generated"])
                    self.assertFalse(response.json()["human_verified"])
                    gen.side_effect = ValueError("图表无法通过结构校验：x")
                    failed = client.post("/api/diagram", json={"question": "画出流程"})
                    self.assertEqual(failed.status_code, 400)
                    self.assertIn("结构校验", failed.json()["detail"])


if __name__ == "__main__":
    unittest.main()
