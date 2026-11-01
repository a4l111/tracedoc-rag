"""DeepSeek answer contract tests without network calls."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.qa.answer import CancelToken, GenerationCancelled, answer_question, stream_answer


def _response(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _delta(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.snapshot_id = "snapshot"
        folder = self.root / self.snapshot_id
        folder.mkdir()
        (folder / "snapshot.json").write_text(json.dumps({"sources": [
            {"doc_id": "doc-a", "source_file": "a.pdf"},
            {"doc_id": "doc-b", "source_file": "b.pdf"},
        ]}), encoding="utf-8")
        self.chunks = [
            {"chunk_id": "a", "doc_id": "doc-a", "page_start": 1, "page_end": 1,
             "heading_path": [], "text": "结论为甲。"},
            {"chunk_id": "b", "doc_id": "doc-b", "page_start": 2, "page_end": 2,
             "heading_path": [], "text": "结论为乙。"},
        ]

    def tearDown(self):
        self.temp.cleanup()

    def _call(self, response_text: str, history=None):
        completion = Mock()
        completion.create.return_value = _response(response_text)
        client = Mock()
        client.chat.completions = completion
        with patch("src.qa.answer.get_api_key", return_value="test-key"), \
             patch("src.qa.answer.OpenAI", return_value=client):
            result = answer_question(self.root, self.snapshot_id, "结论是什么？",
                                     self.root / "config.yaml", chunks=self.chunks, history=history)
        return result, completion

    def test_valid_citations_are_rendered_from_selected_chunks(self):
        result, completion = self._call("两个来源给出不同结论：[1][2]",
                                        history=[{"question": "上一问", "answer": "旧回答[9]"}])
        self.assertIn("[1] a.pdf，第1页", result)
        self.assertIn("[2] b.pdf，第2页", result)
        prompt = completion.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("来源相互冲突", prompt)
        user_prompt = completion.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("旧回答[9]", user_prompt)
        self.assertIn("本轮 Context", user_prompt)

    def test_fabricated_citation_falls_back_to_retrieval(self):
        result, _ = self._call("结论见[3]")
        self.assertIn("不存在的引用", result)
        self.assertIn("检索 Top 5", result)

    def test_insufficient_answer_needs_no_citation(self):
        result, _ = self._call("资料不足")
        self.assertIn("回答：\n资料不足", result)
        self.assertIn("引用：\n无", result)

    def test_missing_key_returns_evidence_without_api_call(self):
        with patch("src.qa.answer.get_api_key", side_effect=ValueError("missing")), \
             patch("src.qa.answer.OpenAI") as client:
            result = answer_question(self.root, self.snapshot_id, "问题", self.root / "config.yaml",
                                     chunks=self.chunks)
        self.assertIn("缺少 DEEPSEEK_API_KEY", result)
        client.assert_not_called()

    def test_stream_yields_deltas_then_validated_done(self):
        completion = Mock()
        completion.create.return_value = iter([
            _delta("两个来源给出不同结论："),
            _delta("[1][2]"),
        ])
        client = Mock()
        client.chat.completions = completion
        with patch("src.qa.answer.get_api_key", return_value="test-key"), \
             patch("src.qa.answer.OpenAI", return_value=client):
            events = list(stream_answer(self.root, self.snapshot_id, "结论是什么？",
                                        self.root / "config.yaml", chunks=self.chunks))
        deltas = [event for event in events if event["type"] == "delta"]
        done = [event for event in events if event["type"] == "done"]
        self.assertEqual([event["text"] for event in deltas], ["两个来源给出不同结论：", "[1][2]"])
        self.assertEqual(len(done), 1)
        self.assertIn("[1] a.pdf，第1页", done[0]["answer"])
        self.assertIn("[2] b.pdf，第2页", done[0]["answer"])
        self.assertFalse(done[0]["fallback"])

    def test_stream_cancel_stops_generation(self):
        token = CancelToken()
        token.cancel()
        completion = Mock()
        completion.create.return_value = iter([_delta("不应到达的内容[1]")])
        client = Mock()
        client.chat.completions = completion
        with patch("src.qa.answer.get_api_key", return_value="test-key"), \
             patch("src.qa.answer.OpenAI", return_value=client):
            with self.assertRaises(GenerationCancelled):
                list(stream_answer(self.root, self.snapshot_id, "结论是什么？",
                                   self.root / "config.yaml", chunks=self.chunks, cancel=token))


if __name__ == "__main__":
    unittest.main()
