"""Fixed and heading-aware chunking must differ when headings exist."""

import unittest

from src.chunker.chunks import make_chunks


class ChunkingTests(unittest.TestCase):
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
