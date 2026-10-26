"""Chunk Markdown produced by the parse stage, without reading raw sources."""

import hashlib
import re
from pathlib import Path
from typing import Any

PAGE = re.compile(r"^<!-- PAGE:(\d+) -->$")
SLIDE = re.compile(r"^<!-- SLIDE:(\d+) -->$")
COMMENT = re.compile(r"^<!-- (?:paragraph_id:|OCR:).+ -->$")
HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
WINDOW = 400
OVERLAP = 50


def _clean_lines(markdown: str) -> list[tuple[str, int | None, int | None]]:
    page: int | None = None
    slide: int | None = None
    lines = []
    for line in markdown.splitlines():
        match = PAGE.match(line.strip())
        if match:
            page = int(match.group(1))
            slide = None
        elif match := SLIDE.match(line.strip()):
            slide = int(match.group(1))
            page = None
        elif not COMMENT.match(line.strip()):
            lines.append((line, page, slide))
    return lines


def _windows(lines: list[tuple[str, int | None, int | None]], split: bool) -> list[tuple[str, int | None, int | None, int | None, int | None]]:
    raw_text = "\n".join(line for line, _, _ in lines)
    leading = len(raw_text) - len(raw_text.lstrip())
    text = raw_text.strip()
    if not text:
        return []
    pages = []
    slides = []
    for index, (line, page, slide) in enumerate(lines):
        pages.extend([page] * len(line))
        slides.extend([slide] * len(line))
        if index < len(lines) - 1:
            pages.append(page)
            slides.append(slide)
    pages = pages[leading:leading + len(text)]
    slides = slides[leading:leading + len(text)]
    starts = [0] if not split or len(text) <= WINDOW else range(0, len(text), WINDOW - OVERLAP)
    result = []
    for start in starts:
        if start >= len(text):
            break
        end = min(start + WINDOW, len(text))
        value = text[start:end].strip()
        if value:
            found = [p for p in pages[start:end] if p is not None]
            found_slides = [s for s in slides[start:end] if s is not None]
            result.append((value, min(found) if found else None, max(found) if found else None,
                           min(found_slides) if found_slides else None,
                           max(found_slides) if found_slides else None))
        if end == len(text):
            break
    return result


def make_chunks(markdown: str, manifest: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    if mode not in {"fixed", "structured"}:
        raise ValueError(f"Unknown chunk mode: {mode}")
    lines = _clean_lines(markdown)
    sections: list[tuple[list[str], list[tuple[str, int | None, int | None]]]] = []
    if mode == "fixed":
        sections.append(([], lines))
    else:
        headings: list[str] = []
        current: list[tuple[str, int | None, int | None]] = []
        current_path: list[str] = []
        for line, page, slide in lines:
            match = HEADING.match(line)
            if match:
                if current:
                    sections.append((current_path, current))
                level = len(match.group(1))
                headings = headings[:level - 1]
                headings.append(match.group(2).strip())
                current_path = headings.copy()
                current = []
            current.append((line, page, slide))
        if current:
            sections.append((current_path, current))

    chunks = []
    doc_id = manifest["doc_id"]
    source_name = Path(manifest["source_file"]).name
    for heading_path, section in sections:
        for value, page_start, page_end, slide_start, slide_end in _windows(section, split=True):
            ordinal = len(chunks) + 1
            identity = f"{doc_id}\0{mode}\0{ordinal}\0{value}"
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            label = f" · {' / '.join(heading_path)}" if heading_path else ""
            display_name = f"{source_name}{label} · #{ordinal}"
            chunks.append({
                "chunk_id": chunk_id,
                "display_name": display_name,
                "doc_id": doc_id,
                "text": value,
                "page_start": page_start,
                "page_end": page_end,
                "slide_start": slide_start,
                "slide_end": slide_end,
                "heading_path": heading_path,
            })
    return chunks
