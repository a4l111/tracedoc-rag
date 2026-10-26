"""Evaluate retrieval against human-labeled, snapshot-independent relevance data."""

import hashlib
import json
import re
import statistics
from pathlib import Path

from src.retrieval.query import ROUTES, query, search


def _read_gold(path: Path) -> list[dict]:
    examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not examples:
        raise ValueError("Gold file has no labeled questions")
    for number, example in enumerate(examples, start=1):
        if not isinstance(example.get("question"), str) or not example["question"].strip():
            raise ValueError(f"Gold line {number} needs a nonempty question")
        if not isinstance(example.get("relevant"), list) or not example["relevant"]:
            raise ValueError(f"Gold line {number} needs a nonempty relevant list")
    return examples


def _relevant_ids(chunks: list[dict], labels: list[dict]) -> set[str]:
    found: set[str] = set()
    for label in labels:
        if "chunk_id" in label:
            found.update(chunk["chunk_id"] for chunk in chunks if chunk["chunk_id"] == label["chunk_id"])
        elif "doc_id" in label:
            page = label.get("page")
            slide = label.get("slide")
            found.update(chunk["chunk_id"] for chunk in chunks
                         if chunk["doc_id"] == label["doc_id"]
                         and (page is None or chunk["page_start"] is not None
                              and chunk["page_start"] <= page <= chunk["page_end"])
                         and (slide is None or chunk.get("slide_start") is not None
                              and chunk["slide_start"] <= slide <= chunk["slide_end"]))
        else:
            raise ValueError("Each relevance label needs doc_id or chunk_id")
    return found


def _relevant_groups(chunks: list[dict], labels: list[dict]) -> list[set[str]]:
    """One gold location is one relevance unit even when it spans many chunks."""
    return [_relevant_ids(chunks, [label]) for label in labels]


def evaluate_snapshot(indexes_dir: Path, snapshot_id: str, gold_path: Path, k: int = 5) -> dict:
    """Compute macro Recall@K and MRR for each route and final reranked results."""
    if not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", snapshot_id):
        raise ValueError("Invalid snapshot_id")
    if not 1 <= k <= 5:
        raise ValueError("K must be between 1 and 5 because the final reranker returns five chunks")
    folder = indexes_dir / snapshot_id
    snapshot = json.loads((folder / "snapshot.json").read_text(encoding="utf-8"))
    report = json.loads((folder / "build_report.json").read_text(encoding="utf-8"))
    if snapshot.get("snapshot_id") != snapshot_id or report.get("snapshot_id") != snapshot_id or not report.get("complete"):
        raise ValueError("Evaluation requires a complete matching snapshot")
    chunk_bytes = (folder / "chunks.jsonl").read_bytes()
    if hashlib.sha256(chunk_bytes).hexdigest() != snapshot["chunks_sha256"]:
        raise ValueError("Chunk snapshot checksum mismatch")
    chunks = [json.loads(line) for line in chunk_bytes.splitlines()]
    examples = _read_gold(gold_path)
    routes = (*ROUTES, "hybrid_reranked")
    recall_key = f"recall_at_{k}"
    totals = {route: {recall_key: 0.0, "reciprocal_rank": 0.0} for route in routes}
    details = []
    for example in examples:
        groups = _relevant_groups(chunks, example["relevant"])
        relevant = set().union(*groups)
        results = {route: search(indexes_dir, snapshot_id, route, example["question"], top_k=k)
                   for route in ROUTES}
        results["hybrid_reranked"] = query(indexes_dir, snapshot_id, example["question"])
        question_detail = {"question": example["question"], "relevant_chunk_count": len(relevant),
                           "relevant_location_count": len(groups),
                           "unindexed_labels": [label for label, group in zip(example["relevant"], groups) if not group],
                           "relevant_chunk_ids": sorted(relevant), "routes": {}}
        for route, ranked in results.items():
            ids = [chunk["chunk_id"] for chunk in ranked[:k]]
            hits = [rank for rank, chunk_id in enumerate(ids, start=1) if chunk_id in relevant]
            recall = sum(bool(group.intersection(ids)) for group in groups) / len(groups)
            reciprocal_rank = 1 / hits[0] if hits else 0.0
            totals[route][recall_key] += recall
            totals[route]["reciprocal_rank"] += reciprocal_rank
            question_detail["routes"][route] = {"retrieved_chunk_ids": ids,
                                                 "missed_chunk_ids": sorted(relevant.difference(ids)),
                                                 "missed_labels": [label for label, group in zip(example["relevant"], groups)
                                                                   if not group.intersection(ids)],
                                                 recall_key: recall, "reciprocal_rank": reciprocal_rank}
        details.append(question_detail)
    metrics = {route: {recall_key: values[recall_key] / len(examples),
                       "mrr": values["reciprocal_rank"] / len(examples)}
               for route, values in totals.items()}
    return {"snapshot_id": snapshot_id, "chunk_mode": snapshot["chunk_mode"],
            "chunks_sha256": snapshot["chunks_sha256"],
            "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
            "model_config": snapshot.get("model_config", {}),
            "chunk_count": len(chunks), "question_count": len(examples), "k": k, "metrics": metrics,
            "questions": details}


def render_report(results: list[dict], gold_path: Path) -> str:
    lines = ["# Retrieval evaluation", "", f"Gold labels: `{gold_path}`", "",
             "Recall@K is the mean fraction of labeled source locations reached by at least one retrieved chunk. "
             "MRR is the mean reciprocal rank of the first relevant chunk.", ""]
    for result in results:
        recall_key = f"recall_at_{result['k']}"
        lines.extend([f"## {result['snapshot_id']} ({result['chunk_mode']})", "",
                      f"Chunks: {result['chunk_count']}; labeled questions: {result['question_count']}", "",
                      f"| Route | Recall@{result['k']} | MRR |", "| --- | ---: | ---: |"])
        for route, metrics in result["metrics"].items():
            lines.append(f"| {route} | {metrics[recall_key]:.4f} | {metrics['mrr']:.4f} |")
        lines.append("")
        lines.extend([f"Gold SHA256: `{result.get('gold_sha256', 'unavailable')}`  ",
                      f"Chunk SHA256: `{result.get('chunks_sha256', 'unavailable')}`", "",
                      "### Missed labels", ""])
        missed = [(question["question"], route, values.get("missed_labels", values["missed_chunk_ids"]))
                  for question in result["questions"] for route, values in question["routes"].items()
                  if values.get("missed_labels", values["missed_chunk_ids"])]
        if missed:
            for question, route, labels in missed:
                lines.append(f"- {question} · {route}: {json.dumps(labels, ensure_ascii=False)}")
        else:
            lines.append("No missed labels at this cutoff.")
        lines.append("")
    return "\n".join(lines)


def compare_chunk_snapshots(indexes_dir: Path, snapshot_ids: list[str]) -> list[dict]:
    """Compare saved chunk distributions without reading or reparsing source files."""
    if len(snapshot_ids) < 2:
        raise ValueError("Provide at least two snapshots for chunk comparison")
    results = []
    source_sets = []
    parsed_hash_sets = []
    for snapshot_id in snapshot_ids:
        if not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", snapshot_id):
            raise ValueError("Invalid snapshot_id")
        folder = indexes_dir / snapshot_id
        snapshot = json.loads((folder / "snapshot.json").read_text(encoding="utf-8"))
        if snapshot.get("snapshot_id") != snapshot_id:
            raise ValueError("Snapshot ID mismatch")
        content = (folder / "chunks.jsonl").read_bytes()
        if hashlib.sha256(content).hexdigest() != snapshot["chunks_sha256"]:
            raise ValueError("Chunk snapshot checksum mismatch")
        chunks = [json.loads(line) for line in content.splitlines()]
        if not chunks:
            raise ValueError(f"Snapshot {snapshot_id} has no chunks")
        source_sets.append({source["doc_id"]: source["sha256"] for source in snapshot["sources"]})
        parsed_hash_sets.append({source["doc_id"]: source.get("markdown_sha256") for source in snapshot["sources"]})
        lengths = [len(chunk["text"]) for chunk in chunks]
        by_document = {source["source_file"]: sum(chunk["doc_id"] == source["doc_id"] for chunk in chunks)
                       for source in snapshot["sources"]}
        results.append({"snapshot_id": snapshot_id, "chunk_mode": snapshot["chunk_mode"],
                        "chunk_count": len(chunks), "mean_characters": statistics.mean(lengths),
                        "median_characters": statistics.median(lengths),
                        "chunks_with_page": sum(chunk["page_start"] is not None for chunk in chunks),
                        "chunks_with_heading": sum(bool(chunk["heading_path"]) for chunk in chunks),
                        "chunks_by_document": by_document})
    if any(sources != source_sets[0] for sources in source_sets[1:]):
        raise ValueError("Snapshots do not contain the same source documents")
    if len({result["chunk_mode"] for result in results}) < 2:
        raise ValueError("Chunk comparison needs different chunk modes")
    if all(all(value is not None for value in hashes.values()) for hashes in parsed_hash_sets):
        if any(hashes != parsed_hash_sets[0] for hashes in parsed_hash_sets[1:]):
            raise ValueError("Snapshots were built from different parsed Markdown")
        for result in results:
            result["parsed_input_verified"] = True
    else:
        for result in results:
            result["parsed_input_verified"] = False
    return results


def render_chunk_comparison(results: list[dict]) -> str:
    lines = ["# Chunk strategy comparison", "",
             "Structural statistics from saved build snapshots. Retrieval quality requires labeled questions.", "",
             "Parsed Markdown checksums match across snapshots." if results[0]["parsed_input_verified"]
             else "Parsed Markdown checksum was not recorded in at least one older snapshot; input equality is unverified.", "",
             "| Snapshot | Mode | Chunks | Mean chars | Median chars | With page | With heading |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for result in results:
        lines.append(f"| {result['snapshot_id']} | {result['chunk_mode']} | {result['chunk_count']} "
                     f"| {result['mean_characters']:.1f} | {result['median_characters']:.1f} "
                     f"| {result['chunks_with_page']} | {result['chunks_with_heading']} |")
    lines.append("")
    return "\n".join(lines)
