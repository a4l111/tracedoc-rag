"""Offline BM25 baseline over parsed artifacts, without model API calls."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.chunker.chunks import make_chunks
from src.evaluation.metrics import _read_gold, _relevant_groups
from src.index.build import tokenize


def _chunks(parsed_dir: Path, mode: str) -> list[dict]:
    chunks = []
    for manifest_path in sorted(parsed_dir.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        markdown = (manifest_path.parent / "document.md").read_text(encoding="utf-8")
        chunks.extend(make_chunks(markdown, manifest, mode))
    if not chunks:
        raise ValueError(f"No chunks found in {parsed_dir}")
    return chunks


def evaluate_mode(parsed_dir: Path, gold_path: Path, mode: str, k: int = 5,
                  candidate_k: int = 20) -> dict:
    """Evaluate one chunk mode with the production BM25 tokenizer."""
    if mode not in {"fixed", "structured"}:
        raise ValueError("mode must be fixed or structured")
    if k < 1 or candidate_k < k:
        raise ValueError("candidate_k must be at least k")
    chunks = _chunks(parsed_dir, mode)
    gold = _read_gold(gold_path)
    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])
    recall_total = 0.0
    candidate_recall_total = 0.0
    reciprocal_rank_total = 0.0
    details = []
    for example in gold:
        groups = _relevant_groups(chunks, example["relevant"])
        scores = bm25.get_scores(tokenize(example["question"]))
        order = sorted(range(len(chunks)), key=lambda index: (-float(scores[index]), index))
        top_ids = [chunks[index]["chunk_id"] for index in order[:k]]
        candidate_ids = [chunks[index]["chunk_id"] for index in order[:candidate_k]]
        recall = sum(bool(group.intersection(top_ids)) for group in groups) / len(groups)
        candidate_recall = sum(bool(group.intersection(candidate_ids)) for group in groups) / len(groups)
        hit_ranks = [rank for rank, chunk_id in enumerate(top_ids, 1)
                     if any(chunk_id in group for group in groups)]
        reciprocal_rank = 1 / hit_ranks[0] if hit_ranks else 0.0
        recall_total += recall
        candidate_recall_total += candidate_recall
        reciprocal_rank_total += reciprocal_rank
        details.append({
            "question": example["question"],
            "source_doc": example.get("source_doc"),
            "benchmark_row": example.get("benchmark_row"),
            "relevant": example["relevant"],
            "top_chunk_ids": top_ids,
            "recall_at_k": recall,
            "candidate_recall": candidate_recall,
            "reciprocal_rank": reciprocal_rank,
            "missed_at_k": [label for label, group in zip(example["relevant"], groups)
                            if not group.intersection(top_ids)],
            "unindexed_labels": [label for label, group in zip(example["relevant"], groups) if not group],
        })
    count = len(gold)
    chunk_bytes = "".join(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n" for chunk in chunks).encode()
    return {
        "route": "bm25_offline",
        "chunk_mode": mode,
        "chunk_count": len(chunks),
        "chunks_with_heading": sum(bool(chunk["heading_path"]) for chunk in chunks),
        "question_count": count,
        "k": k,
        "candidate_k": candidate_k,
        "recall_at_k": recall_total / count,
        "candidate_recall": candidate_recall_total / count,
        "mrr_at_k": reciprocal_rank_total / count,
        "chunks_sha256": hashlib.sha256(chunk_bytes).hexdigest(),
        "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "tokenizer": "src.index.build.tokenize (jieba, strip, lowercase)",
        "questions": details,
    }


def render(results: list[dict], parsed_dir: Path, gold_path: Path) -> str:
    lines = [
        "# Offline BM25 baseline", "",
        f"Parsed artifacts: `{parsed_dir}`  ", f"Gold labels: `{gold_path}`", "",
        "This report uses no embedding, reranker, DeepSeek, or MoArk calls. "
        "It is a BM25 and chunking baseline, not a hybrid retrieval score.", "",
        "| Chunk mode | Chunks | Questions | Recall@5 | Recall@20 | MRR@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(f"| {result['chunk_mode']} | {result['chunk_count']} | {result['question_count']} "
                     f"| {result['recall_at_k']:.4f} | {result['candidate_recall']:.4f} "
                     f"| {result['mrr_at_k']:.4f} |")
    if all(result["chunks_with_heading"] == 0 for result in results):
        lines.extend(["", "No parsed chunk has a Markdown heading. Structured chunking therefore has no "
                      "heading boundary to use on this pilot corpus."])
    lines.extend(["", "## Missed evidence locations at Top 5", ""])
    for result in results:
        lines.append(f"### {result['chunk_mode']}")
        missed = [detail for detail in result["questions"] if detail["missed_at_k"]]
        if not missed:
            lines.append("No misses.")
        for detail in missed:
            lines.append(f"- row {detail.get('benchmark_row')}: {detail['question']} — "
                         f"{json.dumps(detail['missed_at_k'], ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


def run(parsed_dir: Path, gold_path: Path, output_dir: Path) -> Path:
    """Evaluate both modes and write a reproducible report directory."""
    results = [evaluate_mode(parsed_dir, gold_path, mode) for mode in ("fixed", "structured")]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = output_dir / f"bm25-{run_id}"
    suffix = 1
    while folder.exists():
        folder = output_dir / f"bm25-{run_id}-{suffix}"
        suffix += 1
    folder.mkdir(parents=True)
    (folder / "report.json").write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    (folder / "report.md").write_text(render(results, parsed_dir, gold_path), encoding="utf-8")
    return folder
