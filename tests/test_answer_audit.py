"""Supplementary exact-answer audit tests."""

import unittest

from src.evaluation.answer_audit import _answer_terms, _normalize


class AnswerAuditTests(unittest.TestCase):
    def test_list_answer_requires_each_item(self) -> None:
        self.assertEqual(_answer_terms("['Alpha ', 'Beta']"), ["alpha", "beta"])

    def test_normalization_is_case_and_whitespace_insensitive(self) -> None:
        self.assertEqual(_normalize(" Capt.  JOHN\nSanders "), "capt. john sanders")

    def test_not_answerable_is_not_auditable(self) -> None:
        self.assertEqual(_answer_terms("Not answerable"), [])

    def test_short_numeric_answer_is_not_auditable(self) -> None:
        self.assertEqual(_answer_terms("7.0"), [])
        self.assertEqual(_answer_terms("92%"), [])


if __name__ == "__main__":
    unittest.main()
