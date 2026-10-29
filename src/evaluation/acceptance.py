"""Reproducible acceptance checks over saved artifacts; no model calls."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_ROOT = ("app.py", "requirements.txt", "config.yaml", ".env.example", ".gitignore", "README.md")
REQUIRED_DIRS = ("src/parser", "src/chunker", "src/index", "src/retrieval", "src/qa", "src/evaluation", "src/utils")
INDEX_FILES = ("chunks.jsonl", "dense.faiss", "dense_meta.json", "bm25.pkl", "bge_m3.faiss",
               "bge_meta.json", "snapshot.json", "build_report.json")


def _check(items: list[dict], name: str, status: str, detail: str) -> None:
    items.append({"name": name, "status": status, "detail": detail})


def run(root: Path, parsed_dir: Path, indexes_dir: Path, snapshot_id: str,
        evaluation_report: Path, output_dir: Path) -> Path:
    checks: list[dict] = []
    missing = [name for name in REQUIRED_ROOT if not (root / name).is_file()]
    missing += [name for name in REQUIRED_DIRS if not (root / name).is_dir()]
    _check(checks, "项目骨架", "pass" if not missing else "fail",
           "必需文件和目录齐全" if not missing else "缺少：" + ", ".join(missing))

    manifests = sorted(parsed_dir.glob("*/manifest.json")) if parsed_dir.is_dir() else []
    valid_documents = 0
    visual_assets = 0
    unreviewed_visual_assets = 0
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest_path.parent / "document.md").is_file() and (manifest_path.parent / "errors.jsonl").is_file():
                valid_documents += 1
            for asset in manifest.get("image_assets", []):
                if asset.get("vision"):
                    visual_assets += 1
                    unreviewed_visual_assets += asset["vision"].get("review_status") == "unreviewed"
        except (OSError, ValueError):
            continue
    _check(checks, "解析产物", "pass" if valid_documents else "fail",
           f"{valid_documents} 个文档具有 document.md、manifest.json 和 errors.jsonl")
    _check(checks, "视觉溯源", "pass" if visual_assets and visual_assets == unreviewed_visual_assets else "warn",
           f"{visual_assets} 个视觉资产；{unreviewed_visual_assets} 个明确标记为未经人工核验")

    snapshot_dir = indexes_dir / snapshot_id
    missing_index = [name for name in INDEX_FILES if not (snapshot_dir / name).is_file()]
    snapshot = {}
    report = {}
    try:
        snapshot = json.loads((snapshot_dir / "snapshot.json").read_text(encoding="utf-8"))
        report = json.loads((snapshot_dir / "build_report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    complete = (not missing_index and snapshot.get("snapshot_id") == snapshot_id
                and report.get("snapshot_id") == snapshot_id and report.get("complete") is True)
    _check(checks, "完整索引快照", "pass" if complete else "fail",
           f"snapshot={snapshot_id}; missing={missing_index}; complete={report.get('complete')}")
    if complete:
        chunk_bytes = (snapshot_dir / "chunks.jsonl").read_bytes()
        checksum_ok = hashlib.sha256(chunk_bytes).hexdigest() == snapshot.get("chunks_sha256")
        routes = report.get("routes", {})
        routes_ok = all(routes.get(route, {}).get("indexed_count") == report.get("chunk_count")
                        for route in ("dense", "bm25", "bge_m3"))
        _check(checks, "共享 Chunk 与三路数量", "pass" if checksum_ok and routes_ok else "fail",
               f"checksum={checksum_ok}; routes_aligned={routes_ok}; chunks={report.get('chunk_count')}")

    evaluation = None
    try:
        payload = json.loads(evaluation_report.read_text(encoding="utf-8"))
        evaluation = next(item for item in payload["snapshots"] if item["snapshot_id"] == snapshot_id)
    except (OSError, ValueError, KeyError, StopIteration):
        pass
    if not evaluation:
        _check(checks, "正式标注评测", "fail", f"未找到匹配报告：{evaluation_report}")
    else:
        metrics = evaluation.get("metrics", {})
        route_names = {"bm25", "dense", "bge_m3", "hybrid_reranked"}
        _check(checks, "正式标注评测", "pass" if route_names.issubset(metrics) else "fail",
               f"questions={evaluation.get('question_count')}; routes={sorted(metrics)}")
        recall_key = f"recall_at_{evaluation.get('k', 5)}"
        hybrid = metrics.get("hybrid_reranked", {}).get(recall_key, 0.0)
        best_route = max((metrics.get(route, {}).get(recall_key, 0.0)
                          for route in ("bm25", "dense", "bge_m3")), default=0.0)
        _check(checks, "融合质量不退化", "pass" if hybrid >= best_route else "fail",
               f"hybrid={hybrid:.4f}; best_single_route={best_route:.4f}")

    summary = {status: sum(item["status"] == status for item in checks) for status in ("pass", "warn", "fail")}
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "snapshot_id": snapshot_id,
              "parsed_dir": str(parsed_dir), "evaluation_report": str(evaluation_report),
              "summary": summary, "checks": checks}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = output_dir / run_id
    suffix = 1
    while folder.exists():
        folder = output_dir / f"{run_id}-{suffix}"
        suffix += 1
    folder.mkdir(parents=True)
    (folder / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 项目验收报告", "", f"Snapshot: `{snapshot_id}`", "",
             f"结果：PASS {summary['pass']} / WARN {summary['warn']} / FAIL {summary['fail']}", "",
             "| 检查项 | 状态 | 证据 |", "| --- | --- | --- |"]
    lines.extend(f"| {item['name']} | {item['status'].upper()} | {item['detail'].replace('|', '/')} |"
                 for item in checks)
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder
