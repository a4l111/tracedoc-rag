"""Candidate coverage analysis tests."""

import unittest

from src.evaluation.retrieval_analysis import _covered_groups, _rrf


class RetrievalAnalysisTests(unittest.TestCase):
    def test_rrf_preserves_route_ranks_and_orders_shared_candidate_first(self) -> None:
        rankings = {
            "bm25": [{"chunk_id": "shared"}, {"chunk_id": "bm25"}],
            "dense": [{"chunk_id": "dense"}, {"chunk_id": "shared"}],
            "bge_m3": [{"chunk_id": "shared"}],
        }
        fused = _rrf(rankings)
        self.assertEqual(fused[0]["chunk_id"], "shared")
        self.assertEqual(fused[0]["route_ranks"], {"bm25": 1, "dense": 2, "bge_m3": 1})

    def test_covered_groups_counts_locations_not_duplicate_chunks(self) -> None:
        groups = [{"a", "b"}, {"c"}]
        self.assertEqual(_covered_groups(groups, ["a", "b"]), [1])
        self.assertEqual(_covered_groups(groups, ["b", "c"]), [1, 2])


if __name__ == "__main__":
    unittest.main()
