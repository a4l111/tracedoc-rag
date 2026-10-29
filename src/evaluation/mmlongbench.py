"""Import the local MMLongBench-Doc PDFs and human evidence-page labels."""

import ast
import hashlib
import json
from pathlib import Path

import fitz

from src.parser.pipeline import _doc_id, parse_file


def prepare(source_dir: Path, output_dir: Path, max_documents: int = 3) -> dict:
    """Parse a deterministic small subset without touching the project's data/raw."""
    documents = source_dir / "data" / "documents"
    sample_path = source_dir / "data" / "samples.json"
    if not documents.is_dir() or not sample_path.is_file():
        raise ValueError("Expected data/documents and data/samples.json in MMLongBench-Doc")
    if max_documents < 1:
        raise ValueError("max_documents must be positive")
    samples = json.loads(sample_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("samples.json must contain a list")
    names = sorted({sample["doc_id"] for sample in samples})[:max_documents]
    for name in names:
        if Path(name).name != name or not name.lower().endswith(".pdf") or not (documents / name).is_file():
            raise ValueError(f"Missing or unsafe benchmark document: {name}")
    parsed = output_dir / "parsed"
    parsed.mkdir(parents=True, exist_ok=True)
    page_counts = {}
    failures = []
    for name in names:
        source = documents / name
        with fitz.open(source) as pdf:
            page_counts[name] = len(pdf)
        if not parse_file(source, documents, parsed):
            failures.append(name)

    gold = []
    unanswerable = []
    invalid = []
    selected = set(names)
    for row_number, sample in enumerate(samples, 1):
        name = sample["doc_id"]
        if name not in selected or name in failures:
            continue
        try:
            pages = ast.literal_eval(sample["evidence_pages"])
            if not isinstance(pages, list) or any(type(page) is not int or page < 1 or page > page_counts[name]
                                                   for page in pages):
                raise ValueError("invalid one-based evidence pages")
        except (SyntaxError, ValueError, TypeError) as exc:
            invalid.append({"row": row_number, "doc_id": name, "reason": str(exc)})
            continue
        record = {"question": sample["question"], "source_doc": name,
                  "answer": sample["answer"], "evidence_sources": sample.get("evidence_sources"),
                  "benchmark_row": row_number}
        if not pages:
            if sample["answer"] == "Not answerable":
                unanswerable.append(record)
            else:
                invalid.append({"row": row_number, "doc_id": name,
                                "reason": "answer exists but evidence_pages is empty"})
            continue
        record["relevant"] = [{"doc_id": _doc_id(documents / name, documents), "page": page}
                              for page in sorted(set(pages))]
        gold.append(record)

    gold_path = output_dir / "gold.jsonl"
    gold_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gold), encoding="utf-8")
    (output_dir / "unanswerable.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in unanswerable), encoding="utf-8")
    summary = {"source_dir": str(source_dir.resolve()), "samples_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
               "selected_documents": names, "parsed_documents": len(names) - len(failures),
               "failed_documents": failures, "answerable_questions": len(gold),
               "unanswerable_questions": len(unanswerable), "invalid_labels": invalid,
               "page_rule": "benchmark one-based evidence_pages = parser PAGE:n",
               "parsed_dir": str(parsed.resolve()), "gold_path": str(gold_path.resolve())}
    (output_dir / "import_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                     encoding="utf-8")
    return summary
