"""Analyze candidate coverage and final reranking errors for a saved evaluation."""

import json
from pathlib import Path

from src.evaluation.metrics import _read_gold, _relevant_groups
from src.retrieval.query import ROUTES, search


def _covered_groups(groups: list[set[str]], ids: list[str]) -> list[int]:
    selected = set(ids)
    return [index + 1 for index, group in enumerate(groups) if group.intersection(selected)]


def _rrf(rankings: dict[str, list[dict]]) -> list[dict]:
    fused: dict[str, dict] = {}
    for route in ROUTES:
        for rank, chunk in enumerate(rankings[route][:20], start=1):
            entry = fused.setdefault(chunk["chunk_id"], {"chunk_id": chunk["chunk_id"], "score": 0.0,
                                                          "route_ranks": {}})
            entry["score"] += 1.0 / (60 + rank)
            entry["route_ranks"][route] = rank
    return sorted(fused.values(), key=lambda item: (-item["score"], item["chunk_id"]))[:50]


def analyze(indexes_dir: Path, snapshot_id: str, gold_path: Path,
            evaluation_report: Path, output_dir: Path) -> Path:
    """Recreate Top-20 candidates and compare them with a completed final Top-5 report."""
    folder = indexes_dir / snapshot_id
    chunks = [json.loads(line) for line in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    gold = _read_gold(gold_path)
    report = json.loads(evaluation_report.read_text(encoding="utf-8"))
    matches = [item for item in report.get("snapshots", []) if item.get("snapshot_id") == snapshot_id]
    if len(matches) != 1:
        raise ValueError("Evaluation report must contain exactly one matching snapshot")
    evaluated = matches[0]
    if evaluated.get("question_count") != len(gold) or len(evaluated.get("questions", [])) != len(gold):
        raise ValueError("Evaluation report is incomplete or does not match the gold questions")

    details = []
    counts = {"final_hit": 0, "reranker_drop": 0, "candidate_miss": 0,
              "single_route_regression": 0, "hybrid_rescue": 0}
    route_coverage = {route: 0 for route in ROUTES}
    rrf_coverage = 0
    final_coverage = 0
    for number, (example, prior) in enumerate(zip(gold, evaluated["questions"]), start=1):
        if prior.get("question") != example["question"]:
            raise ValueError(f"Question order mismatch at item {number}")
        groups = _relevant_groups(chunks, example["relevant"])
        rankings = {route: search(indexes_dir, snapshot_id, route, example["question"], top_k=20)
                    for route in ROUTES}
        route_ids = {route: [chunk["chunk_id"] for chunk in ranked] for route, ranked in rankings.items()}
        route_groups = {route: _covered_groups(groups, ids) for route, ids in route_ids.items()}
        for route, covered in route_groups.items():
            route_coverage[route] += len(covered)

        rrf = _rrf(rankings)
        rrf_ids = [item["chunk_id"] for item in rrf]
        rrf_groups = _covered_groups(groups, rrf_ids)
        final_ids = prior["routes"]["hybrid_reranked"]["retrieved_chunk_ids"]
        final_groups = _covered_groups(groups, final_ids)
        rrf_coverage += len(rrf_groups)
        final_coverage += len(final_groups)

        if final_groups:
            category = "final_hit"
        elif rrf_groups:
            category = "reranker_drop"
        else:
            category = "candidate_miss"
        counts[category] += 1

        single_top5_hit = any(_covered_groups(groups, ids[:5]) for ids in route_ids.values())
        if single_top5_hit and not final_groups:
            counts["single_route_regression"] += 1
        if final_groups and not single_top5_hit:
            counts["hybrid_rescue"] += 1

        relevant = set().union(*groups)
        relevant_candidates = [{"chunk_id": item["chunk_id"], "rrf_rank": rank,
                                "rrf_score": item["score"], "route_ranks": item["route_ranks"]}
                               for rank, item in enumerate(rrf, start=1) if item["chunk_id"] in relevant]
        details.append({"number": number, "question": example["question"], "gold": example["relevant"],
                        "category": category, "route_top20_covered_groups": route_groups,
                        "route_top5_covered_groups": {
                            route: _covered_groups(groups, ids[:5]) for route, ids in route_ids.items()},
                        "rrf_top50_covered_groups": rrf_groups,
                        "final_top5_covered_groups": final_groups,
                        "relevant_rrf_candidates": relevant_candidates,
                        "final_chunk_ids": final_ids})

    total_groups = sum(len(_relevant_groups(chunks, example["relevant"])) for example in gold)
    result = {"snapshot_id": snapshot_id, "question_count": len(gold), "gold_location_count": total_groups,
              "summary": counts,
              "location_coverage": {**{f"{route}_top20": route_coverage[route] / total_groups
                                        for route in ROUTES},
                                    "rrf_top50": rrf_coverage / total_groups,
                                    "final_top5": final_coverage / total_groups},
              "questions": details}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(result), encoding="utf-8")
    return output_dir


def render_markdown(result: dict) -> str:
    coverage = result["location_coverage"]
    summary = result["summary"]
    lines = ["# 候选覆盖与逐题排序误差分析", "", f"Snapshot: `{result['snapshot_id']}`", "",
             "## 汇总", "",
             f"- 问题数：{result['question_count']}",
             f"- 最终命中：{summary['final_hit']}",
             f"- 候选已召回但被精排淘汰：{summary['reranker_drop']}",
             f"- RRF Top 50 未召回：{summary['candidate_miss']}",
             f"- 单路 Top 5 命中但最终结果丢失：{summary['single_route_regression']}",
             f"- 三个单路 Top 5 均未命中但融合救回：{summary['hybrid_rescue']}", "",
             "## 标注位置覆盖率", "", "| 阶段 | 覆盖率 |", "| --- | ---: |"]
    labels = {"bm25_top20": "BM25 Top 20", "dense_top20": "Dense Top 20",
              "bge_m3_top20": "bge-m3 Top 20", "rrf_top50": "RRF Top 50", "final_top5": "最终 Top 5"}
    lines.extend(f"| {labels[key]} | {value:.4f} |" for key, value in coverage.items())
    lines.extend(["", "## 逐题分析", "",
                  "| # | 分类 | BM25/ Dense/ bge Top20 | 单路Top5命中 | RRF相关候选排名 | 最终命中 | 问题 |",
                  "| ---: | --- | --- | --- | --- | --- | --- |"])
    names = {"final_hit": "最终命中", "reranker_drop": "精排淘汰", "candidate_miss": "候选未召回"}
    for item in result["questions"]:
        route_hits = "/".join(str(len(item["route_top20_covered_groups"][route])) for route in ROUTES)
        top5_hits = "/".join(str(len(item["route_top5_covered_groups"][route])) for route in ROUTES)
        ranks = ", ".join(str(candidate["rrf_rank"]) for candidate in item["relevant_rrf_candidates"]) or "-"
        final_hit = "是" if item["final_top5_covered_groups"] else "否"
        question = item["question"].replace("|", "\\|")
        lines.append(f"| {item['number']} | {names[item['category']]} | {route_hits} | {top5_hits} | {ranks} | {final_hit} | {question} |")
    lines.append("")
    return "\n".join(lines)
