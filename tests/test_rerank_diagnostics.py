"""Reranker diagnostic report tests."""

import unittest

from src.evaluation.rerank_diagnostics import render_markdown


class RerankDiagnosticTests(unittest.TestCase):
    def test_report_contains_scores_but_not_candidate_text(self) -> None:
        payload = {"snapshot_id": "snapshot", "questions": [{"number": 9, "question": "Question?",
            "matches_saved_final_top5": True, "relevant_candidates": [{"chunk_id": "c1", "rrf_rank": 1,
            "initial_batch": 1, "initial_rank": 2, "initial_score": 0.5, "advanced": True,
            "final_rank": 6, "final_score": 0.4, "selected_top5": False}]}]}
        report = render_markdown(payload)
        self.assertIn("0.500000", report)
        self.assertIn("`c1`", report)
        self.assertNotIn("candidate text", report)


if __name__ == "__main__":
    unittest.main()
