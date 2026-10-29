"""Collect privacy-limited reranker tournament diagnostics for selected benchmark questions."""

import json
from pathlib import Path

from src.evaluation.metrics import _read_gold, _relevant_groups
from src.evaluation.retrieval_analysis import _rrf
from src.retrieval.query import ROUTES, search
from src.utils import moark


def diagnose(indexes_dir: Path, snapshot_id: str, gold_path: Path, evaluation_report: Path,
             question_numbers: list[int], output_dir: Path) -> Path:
    chunks = [json.loads(line) for line in (indexes_dir / snapshot_id / "chunks.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    gold = _read_gold(gold_path)
    selected = sorted(set(question_numbers or range(1, len(gold) + 1)))
    if not selected or selected[0] < 1 or selected[-1] > len(gold):
        raise ValueError(f"Question numbers must be between 1 and {len(gold)}")
    report = json.loads(evaluation_report.read_text(encoding="utf-8"))
    matches = [item for item in report.get("snapshots", []) if item.get("snapshot_id") == snapshot_id]
    if len(matches) != 1 or len(matches[0].get("questions", [])) != len(gold):
        raise ValueError("Evaluation report is incomplete or does not contain the selected snapshot")
    evaluated = matches[0]["questions"]
    api, endpoint = moark.rerank_client()
    results = []
    for number in selected:
        example = gold[number - 1]
        prior = evaluated[number - 1]
        if prior.get("question") != example["question"]:
            raise ValueError(f"Question order mismatch at item {number}")
        rankings = {route: search(indexes_dir, snapshot_id, route, example["question"], top_k=20)
                    for route in ROUTES}
        candidates = _rrf(rankings)
        by_id = {chunk["chunk_id"]: chunk for ranked in rankings.values() for chunk in ranked}
        scores, diagnostics = moark.rerank_with_diagnostics(
            api, example["question"], [by_id[item["chunk_id"]]["text"] for item in candidates], endpoint)
        groups = _relevant_groups(chunks, example["relevant"])
        relevant = set().union(*groups)
        candidate_rows = []
        for rrf_rank, (candidate, diagnostic, score) in enumerate(zip(candidates, diagnostics, scores), start=1):
            row = {"chunk_id": candidate["chunk_id"], "rrf_rank": rrf_rank,
                   "rrf_score": candidate["score"], "route_ranks": candidate["route_ranks"],
                   **{key: value for key, value in diagnostic.items() if key != "candidate_index"},
                   "selected_top5": bool(diagnostic["final_rank"] and diagnostic["final_rank"] <= 5),
                   "relevant": candidate["chunk_id"] in relevant}
            candidate_rows.append(row)
        current_final = [row["chunk_id"] for row in sorted(
            candidate_rows, key=lambda row: (row["final_rank"] is None, row["final_rank"] or 10**9))[:5]
            if row["final_rank"] is not None]
        saved_final = prior["routes"]["hybrid_reranked"]["retrieved_chunk_ids"]
        results.append({"number": number, "question": example["question"],
                        "saved_final_chunk_ids": saved_final, "diagnostic_final_chunk_ids": current_final,
                        "matches_saved_final_top5": current_final == saved_final,
                        "relevant_candidates": [row for row in candidate_rows if row["relevant"]],
                        "finalists": [row for row in candidate_rows if row["advanced"]]})
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"snapshot_id": snapshot_id, "rerank_strategy": moark.RERANK_STRATEGY,
               "question_numbers": selected, "questions": results,
               "privacy": "Numeric scores and chunk IDs only; no document text, prompts, or raw API responses."}
    (output_dir / "rerank_diagnostics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                                         encoding="utf-8")
    (output_dir / "rerank_diagnostics.md").write_text(render_markdown(payload), encoding="utf-8")
    return output_dir


def render_markdown(payload: dict) -> str:
    lines = ["# Reranker 分批晋级诊断", "", f"Snapshot: `{payload['snapshot_id']}`", "",
             "仅保存数值分数和 Chunk ID；不保存文档文本、Prompt 或原始 API 响应。", ""]
    for item in payload["questions"]:
        lines.extend([f"## 第 {item['number']} 题", "", item["question"], "",
                      f"与正式评测 Top 5 一致：{'是' if item['matches_saved_final_top5'] else '否'}", "",
                      "| Chunk ID | RRF排名 | 初赛批次/名次 | 初赛分数 | 晋级 | 决赛名次 | 决赛分数 | Top5 |",
                      "| --- | ---: | --- | ---: | --- | ---: | ---: | --- |"])
        if not item["relevant_candidates"]:
            lines.append("| 无相关候选 | - | - | - | - | - | - | - |")
        for row in item["relevant_candidates"]:
            final_rank = row["final_rank"] if row["final_rank"] is not None else "-"
            final_score = f"{row['final_score']:.6f}" if row["final_score"] is not None else "-"
            lines.append(f"| `{row['chunk_id']}` | {row['rrf_rank']} | {row['initial_batch']}/{row['initial_rank']} | "
                         f"{row['initial_score']:.6f} | {'是' if row['advanced'] else '否'} | {final_rank} | "
                         f"{final_score} | {'是' if row['selected_top5'] else '否'} |")
        lines.append("")
    return "\n".join(lines)
