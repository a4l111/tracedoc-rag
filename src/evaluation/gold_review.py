"""Create, validate, apply, and offline-rescore auditable gold-label reviews."""

import json
from copy import deepcopy
from datetime import date
from pathlib import Path

from src.evaluation.metrics import _read_gold, _relevant_groups
from src.retrieval.query import ROUTES

DECISIONS = {"keep", "replace", "add", "uncertain"}
STATUSES = {"confirmed", "needs_second_review"}


def prepare(gold_path: Path, question_numbers: list[int], output_path: Path) -> Path:
    if output_path.exists():
        raise ValueError(f"Refusing to overwrite existing review file: {output_path}")
    gold = _read_gold(gold_path)
    selected = sorted(set(question_numbers))
    if not selected or selected[0] < 1 or selected[-1] > len(gold):
        raise ValueError(f"Question numbers must be between 1 and {len(gold)}")
    records = []
    for number in selected:
        item = gold[number - 1]
        records.append({"benchmark_row": item.get("benchmark_row"), "question_number": number,
                        "question": item["question"], "source_doc": item.get("source_doc"),
                        "answer": item.get("answer"),
                        "original_pages": [label["page"] for label in item["relevant"] if "page" in label],
                        "reviewed_pages": [], "decision": "uncertain", "evidence": "",
                        "reviewer": "", "reviewed_at": str(date.today()),
                        "status": "needs_second_review", "visual_description_reviewed": False})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                           encoding="utf-8")
    return output_path


def _read_reviews(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("Review file has no records")
    return records


def _validate_reviews(gold: list[dict], reviews: list[dict]) -> dict[int, dict]:
    by_row = {item.get("benchmark_row"): item for item in gold}
    number_by_row = {item.get("benchmark_row"): number for number, item in enumerate(gold, start=1)}
    if None in by_row or len(by_row) != len(gold):
        raise ValueError("Every gold record needs a unique benchmark_row")
    validated = {}
    for review in reviews:
        row = review.get("benchmark_row")
        if type(row) is not int or row not in by_row or row in validated:
            raise ValueError(f"Unknown or duplicate benchmark_row: {row}")
        original = by_row[row]
        if review.get("question_number") != number_by_row[row]:
            raise ValueError(f"Review row {row} question_number mismatch")
        for field in ("question", "source_doc", "answer"):
            if review.get(field) != original.get(field):
                raise ValueError(f"Review row {row} does not match gold field: {field}")
        original_pages = sorted(label["page"] for label in original["relevant"] if "page" in label)
        if sorted(review.get("original_pages", [])) != original_pages:
            raise ValueError(f"Review row {row} original_pages mismatch")
        decision = review.get("decision")
        status = review.get("status")
        pages = review.get("reviewed_pages")
        if decision not in DECISIONS or status not in STATUSES:
            raise ValueError(f"Review row {row} has invalid decision or status")
        if not isinstance(pages, list) or any(type(page) is not int or page < 1 for page in pages):
            raise ValueError(f"Review row {row} has invalid reviewed_pages")
        if decision == "keep" and sorted(pages) != original_pages:
            raise ValueError(f"Review row {row} keep must preserve original_pages")
        if decision in {"replace", "add"} and not pages:
            raise ValueError(f"Review row {row} {decision} requires reviewed_pages")
        if status == "confirmed" and (not str(review.get("reviewer", "")).strip()
                                       or not str(review.get("evidence", "")).strip()):
            raise ValueError(f"Confirmed review row {row} needs reviewer and evidence")
        if not isinstance(review.get("visual_description_reviewed"), bool):
            raise ValueError(f"Review row {row} needs boolean visual_description_reviewed")
        validated[row] = review
    return validated


def _rescore(chunks: list[dict], gold: list[dict], evaluated_questions: list[dict], routes: tuple[str, ...]) -> dict:
    totals = {route: {"recall_at_5": 0.0, "reciprocal_rank": 0.0} for route in routes}
    for item, evaluated in zip(gold, evaluated_questions):
        if item["question"] != evaluated.get("question"):
            raise ValueError("Evaluation report question order does not match gold")
        groups = _relevant_groups(chunks, item["relevant"])
        for route in routes:
            ids = evaluated["routes"][route]["retrieved_chunk_ids"][:5]
            hits = [rank for rank, chunk_id in enumerate(ids, start=1)
                    if any(chunk_id in group for group in groups)]
            totals[route]["recall_at_5"] += sum(bool(group.intersection(ids)) for group in groups) / len(groups)
            totals[route]["reciprocal_rank"] += 1 / hits[0] if hits else 0.0
    count = len(gold)
    return {route: {"recall_at_5": values["recall_at_5"] / count,
                    "mrr": values["reciprocal_rank"] / count} for route, values in totals.items()}


def apply(gold_path: Path, review_path: Path, evaluation_report: Path,
          indexes_dir: Path, snapshot_id: str, output_dir: Path) -> Path:
    original = _read_gold(gold_path)
    reviews = _read_reviews(review_path)
    validated = _validate_reviews(original, reviews)
    reviewed = deepcopy(original)
    changes = []
    for item in reviewed:
        review = validated.get(item.get("benchmark_row"))
        if not review:
            continue
        before = [label["page"] for label in item["relevant"] if "page" in label]
        after = before
        applied = review["status"] == "confirmed" and review["decision"] in {"replace", "add"}
        if applied:
            after = (review["reviewed_pages"] if review["decision"] == "replace"
                     else sorted(set(before + review["reviewed_pages"])))
            doc_id = next(label["doc_id"] for label in item["relevant"] if "doc_id" in label)
            item["relevant"] = [{"doc_id": doc_id, "page": page} for page in sorted(set(after))]
        changes.append({"benchmark_row": review["benchmark_row"],
                        "question_number": review.get("question_number"), "decision": review["decision"],
                        "status": review["status"], "applied": applied, "original_pages": before,
                        "after_review_pages": after, "evidence": review["evidence"],
                        "reviewer": review["reviewer"], "reviewed_at": review.get("reviewed_at"),
                        "visual_description_reviewed": review["visual_description_reviewed"]})

    report = json.loads(evaluation_report.read_text(encoding="utf-8"))
    matches = [entry for entry in report.get("snapshots", []) if entry.get("snapshot_id") == snapshot_id]
    if len(matches) != 1 or len(matches[0].get("questions", [])) != len(original):
        raise ValueError("Evaluation report is incomplete or does not contain the selected snapshot")
    snapshot = matches[0]
    folder = indexes_dir / snapshot_id
    chunks = [json.loads(line) for line in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    routes = (*ROUTES, "hybrid_reranked")
    original_metrics = _rescore(chunks, original, snapshot["questions"], routes)
    reviewed_metrics = _rescore(chunks, reviewed, snapshot["questions"], routes)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gold_reviewed.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviewed), encoding="utf-8")
    audit = {"snapshot_id": snapshot_id, "source_gold": str(gold_path), "review_file": str(review_path),
             "reviewed_records": len(reviews), "applied_changes": sum(item["applied"] for item in changes),
             "unreviewed_visual_records": sum(not item["visual_description_reviewed"] for item in changes),
             "changes": changes, "original_metrics": original_metrics, "reviewed_metrics": reviewed_metrics,
             "warning": "Reviewed metrics are supplementary; original benchmark metrics remain authoritative."}
    (output_dir / "gold_review_report.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                                                         encoding="utf-8")
    (output_dir / "gold_review_report.md").write_text(_render_report(audit), encoding="utf-8")
    return output_dir


def _render_report(audit: dict) -> str:
    lines = ["# Gold 人工复核与离线重算", "", f"Snapshot: `{audit['snapshot_id']}`", "",
             f"- 复核记录：{audit['reviewed_records']}", f"- 实际修改：{audit['applied_changes']}",
             f"- 未完成原版视觉核验：{audit['unreviewed_visual_records']}", "",
             "## 指标对比", "", "| 路线 | 原始 Recall@5 | 复核 Recall@5 | 原始 MRR | 复核 MRR |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for route in (*ROUTES, "hybrid_reranked"):
        before, after = audit["original_metrics"][route], audit["reviewed_metrics"][route]
        lines.append(f"| {route} | {before['recall_at_5']:.4f} | {after['recall_at_5']:.4f} | "
                     f"{before['mrr']:.4f} | {after['mrr']:.4f} |")
    lines.extend(["", "## 复核记录", "", "| benchmark_row | decision | status | applied | 页码变更 | 原版视觉核验 |",
                  "| ---: | --- | --- | --- | --- | --- |"])
    for item in audit["changes"]:
        pages = f"{item['original_pages']} → {item['after_review_pages']}"
        lines.append(f"| {item['benchmark_row']} | {item['decision']} | {item['status']} | "
                     f"{'是' if item['applied'] else '否'} | {pages} | "
                     f"{'是' if item['visual_description_reviewed'] else '否'} |")
    lines.extend(["", "> 复核指标仅作补充；不得覆盖或隐去原始 MMLongBench 指标。", ""])
    return "\n".join(lines)
