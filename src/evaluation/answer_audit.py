"""Conservative supplementary audit of answer strings in saved final chunks."""

import ast
import json
import re
import unicodedata
from pathlib import Path

from src.evaluation.metrics import _read_gold


def _normalize(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).casefold()).strip()


def _answer_terms(answer: object) -> list[str]:
    if answer is None:
        return []
    parsed = answer
    if isinstance(answer, str):
        try:
            parsed = ast.literal_eval(answer)
        except (SyntaxError, ValueError):
            parsed = answer
    values = parsed if isinstance(parsed, (list, tuple)) else [parsed]
    terms = [term for value in values if (term := _normalize(value)) and term != "not answerable"]
    if any(re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", term) for term in terms):
        return []
    return terms


def audit(indexes_dir: Path, snapshot_id: str, gold_path: Path,
          evaluation_report: Path, output_dir: Path) -> Path:
    chunks = {item["chunk_id"]: item for item in (
        json.loads(line) for line in (indexes_dir / snapshot_id / "chunks.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip())}
    gold = _read_gold(gold_path)
    report = json.loads(evaluation_report.read_text(encoding="utf-8"))
    matches = [item for item in report.get("snapshots", []) if item.get("snapshot_id") == snapshot_id]
    if len(matches) != 1 or len(matches[0].get("questions", [])) != len(gold):
        raise ValueError("Evaluation report is incomplete or does not contain the selected snapshot")
    details = []
    counts = {"page_metric_hit": 0, "exact_answer_text_hit": 0,
              "page_miss_text_hit": 0, "page_hit_text_miss": 0, "not_auditable": 0}
    for number, (example, evaluated) in enumerate(zip(gold, matches[0]["questions"]), start=1):
        if evaluated.get("question") != example["question"]:
            raise ValueError(f"Question order mismatch at item {number}")
        final = evaluated["routes"]["hybrid_reranked"]
        ids = final["retrieved_chunk_ids"]
        combined = _normalize("\n".join(chunks[chunk_id]["text"] for chunk_id in ids))
        terms = _answer_terms(example.get("answer"))
        matched = [term for term in terms if term in combined]
        text_hit = bool(terms) and len(matched) == len(terms)
        page_hit = final.get("recall_at_5", 0.0) > 0
        if not terms:
            counts["not_auditable"] += 1
        if page_hit:
            counts["page_metric_hit"] += 1
        if text_hit:
            counts["exact_answer_text_hit"] += 1
        if text_hit and not page_hit:
            counts["page_miss_text_hit"] += 1
        if terms and page_hit and not text_hit:
            counts["page_hit_text_miss"] += 1
        details.append({"number": number, "question": example["question"], "answer": example.get("answer"),
                        "auditable": bool(terms), "page_metric_hit": page_hit, "exact_answer_text_hit": text_hit,
                        "matched_terms": matched, "missing_terms": [term for term in terms if term not in matched],
                        "final_chunk_ids": ids})
    payload = {"snapshot_id": snapshot_id, "summary": counts, "questions": details,
               "warning": "Exact normalized substring audit only; supplementary and never a replacement for page labels."}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "answer_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    (output_dir / "answer_audit.md").write_text(render_markdown(payload), encoding="utf-8")
    return output_dir


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = ["# 最终 Top 5 答案文本辅助审计", "", f"Snapshot: `{payload['snapshot_id']}`", "",
             "这是保守的规范化精确子串检查，只用于发现页码标签可能漏记的情况，不能替代正式页码指标。", "",
             f"- 页码指标命中：{summary['page_metric_hit']}",
             f"- 完整答案文本命中：{summary['exact_answer_text_hit']}",
             f"- 页码未命中但答案文本存在：{summary['page_miss_text_hit']}",
             f"- 页码命中但未找到完整答案文本：{summary['page_hit_text_miss']}", "",
             f"- 不适合精确文本审计：{summary['not_auditable']}", "",
             "| # | 可审计 | 页码命中 | 答案文本命中 | 缺失项 | 问题 |",
             "| ---: | --- | --- | --- | --- | --- |"]
    for item in payload["questions"]:
        missing = ", ".join(item["missing_terms"]).replace("|", "\\|") or "-"
        question = item["question"].replace("|", "\\|")
        lines.append(f"| {item['number']} | {'是' if item['auditable'] else '否'} | "
                     f"{'是' if item['page_metric_hit'] else '否'} | "
                     f"{'是' if item['exact_answer_text_hit'] else '否'} | {missing} | {question} |")
    lines.append("")
    return "\n".join(lines)
