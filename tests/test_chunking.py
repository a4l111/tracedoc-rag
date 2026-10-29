"""Fixed and heading-aware chunking must differ when headings exist."""

import unittest

from src.chunker.chunks import make_chunks


class ChunkingTests(unittest.TestCase):
    def test_table_br_rows_keep_numeric_evidence_together(self) -> None:
        markdown = ("<!-- PAGE:10 -->\n# Survey\n| Strongly Not strongly |\n"
                    "| Ages 18-29 65 46 14 27<br>30-49 54 41 26 38<br>"
                    "50-64 50 42 34 45<br>65+ 49 39 34 47 |\n")
        manifest = {"doc_id": "survey", "source_file": "survey.pdf"}
        chunks = make_chunks(markdown, manifest, "structured")
        self.assertTrue(any("Ages 18-29 65 46" in chunk["text"] and "65+ 49 39" in chunk["text"]
                            and chunk["page_start"] == 10 for chunk in chunks))
        self.assertTrue(all(len(chunk["text"]) <= 400 for chunk in chunks))

    def test_long_table_keeps_page_chart_context_with_age_rows(self) -> None:
        markdown = ("<!-- PAGE:10 -->\nWide gender gap in approval of Trump\n"
                    "% who approve or disapprove of the president\nDisapprove Approve\n"
                    "Strongly Not strongly Not strongly Strongly\n" + "Other text. " * 60 + "\n"
                    "| Total 54 42 27 39<br>Men 45 33 34 48<br>Women 64 51 20 30<br>"
                    "White 44 36 37 49<br>Black 83 65 7 13<br>Hispanic 69 49 9 23<br>"
                    "Ages 18-29 65 46 14 27<br>30-49 54 41 26 38<br>50-64 50 42 34 45<br>"
                    "65+ 49 39 34 47<br>Postgrad 68 59 16 23 |\n")
        chunks = make_chunks(markdown, {"doc_id": "survey", "source_file": "survey.pdf"}, "structured")
        self.assertTrue(any("Wide gender gap" in chunk["text"]
                            and "Ages 18-29 65 46" in chunk["text"]
                            and "65+ 49 39" in chunk["text"]
                            and chunk["page_start"] == chunk["page_end"] == 10
                            for chunk in chunks))

    def test_heading_paths_and_real_page_markers(self) -> None:
        markdown = "<!-- PAGE:1 -->\n# 第一章\n" + "甲" * 430 + "\n<!-- PAGE:2 -->\n## 小节\n" + "乙" * 80
        manifest = {"doc_id": "sample", "source_file": "sample.pdf"}
        fixed = make_chunks(markdown, manifest, "fixed")
        structured = make_chunks(markdown, manifest, "structured")
        self.assertTrue(all(not chunk["heading_path"] for chunk in fixed))
        self.assertTrue(any(chunk["heading_path"] == ["第一章", "小节"] for chunk in structured))
        self.assertTrue(any(chunk["page_start"] == 2 for chunk in structured))
        self.assertTrue(all(chunk["chunk_id"] != chunk["display_name"] for chunk in structured))


if __name__ == "__main__":
    unittest.main()
