"""Small shared LAN interface over the existing parse/build/query pipeline."""

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.index.build import build as build_snapshot
from src.parser.pipeline import EXTENSIONS, parse_file, supported_files
from src.qa.answer import answer_question
from src.retrieval.query import latest_complete_snapshot, query as retrieve

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PARSED = ROOT / "data" / "parsed"
INDEXES = ROOT / "data" / "indexes"
SERVICE = ROOT / "data" / "service"
JOBS = SERVICE / "jobs"
EVENTS = SERVICE / "events.jsonl"
MAX_UPLOAD = 50 * 1024 * 1024
executor = ThreadPoolExecutor(max_workers=1)
lock = threading.Lock()
app = FastAPI(title="Multimodal RAG LAN")


@app.middleware("http")
async def same_origin_writes(request: Request, call_next):
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc.lower() != request.headers.get("host", "").lower():
            return JSONResponse({"detail": "跨站写入被拒绝"}, status_code=403)
    return await call_next(request)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _job(job_id: str, **changes) -> dict:
    path = JOBS / f"{job_id}.json"
    with lock:
        record = json.loads(path.read_text(encoding="utf-8"))
        record.update(changes)
        record["updated_at"] = _now()
        _write_json(path, record)
    return record


def _new_job(kind: str) -> str:
    job_id = uuid.uuid4().hex
    _write_json(JOBS / f"{job_id}.json", {
        "job_id": job_id, "kind": kind, "state": "queued", "stage": "排队中",
        "completed": 0, "total": None, "errors": 0,
        "created_at": _now(), "updated_at": _now(),
    })
    return job_id


def _event(kind: str, success: bool, elapsed_ms: int | None = None) -> None:
    SERVICE.mkdir(parents=True, exist_ok=True)
    with lock, EVENTS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": _now(), "kind": kind, "success": success,
                                 "elapsed_ms": elapsed_ms}, ensure_ascii=False) + "\n")


def _parse_job(job_id: str, paths: list[Path]) -> None:
    _job(job_id, state="running", stage="解析中", total=len(paths))
    failures = []
    for number, path in enumerate(paths, 1):
        success = parse_file(path, RAW, PARSED)
        if not success:
            failures.append(path.name)
        _job(job_id, completed=number, errors=len(failures), current_file=path.name)
    _job(job_id, state="completed" if not failures else "partial", stage="解析完成",
         failed_files=failures)
    _event("parse", not failures)


def _build_job(job_id: str, mode: str) -> None:
    _job(job_id, state="running", stage="构建索引中（模型调用进度不可逐块读取）")
    folder = build_snapshot(PARSED, INDEXES, mode, False)
    report = json.loads((folder / "build_report.json").read_text(encoding="utf-8"))
    _job(job_id, state="completed" if report["complete"] else "partial", stage="构建完成",
         snapshot_id=folder.name, completed=report["chunk_count"], total=report["chunk_count"],
         errors=report.get("error_count", 0), report=str(folder / "build_report.json"))
    _event("build", report["complete"])


def _run(job_id: str, callback, *args) -> None:
    try:
        callback(job_id, *args)
    except Exception as exc:
        _job(job_id, state="failed", stage="失败", error=type(exc).__name__)
        _event("job", False)


@app.on_event("startup")
def _recover_jobs() -> None:
    JOBS.mkdir(parents=True, exist_ok=True)
    for path in JOBS.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("state") in {"queued", "running"}:
                _job(record["job_id"], state="interrupted", stage="服务重启中断，可手动重试")
        except (OSError, ValueError, KeyError):
            continue


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/api/documents")
def documents():
    result = []
    for path in sorted(PARSED.glob("*/manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            result.append({"doc_id": manifest["doc_id"], "source_file": manifest["source_file"],
                           "source_type": manifest["source_type"], "errors": len(manifest.get("errors", [])),
                           "images": len(manifest.get("image_assets", []))})
        except (OSError, KeyError, ValueError):
            continue
    return result


@app.get("/api/snapshots")
def snapshots():
    result = []
    for path in sorted(INDEXES.glob("*/build_report.json"), reverse=True):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            result.append({"snapshot_id": path.parent.name, "complete": report["complete"],
                           "chunk_count": report["chunk_count"]})
        except (OSError, KeyError, ValueError):
            continue
    return result


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if not name or name in {".", ".."} or Path(name).suffix.lower() not in EXTENSIONS:
        raise HTTPException(400, "不支持该文件类型")
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / f"{uuid.uuid4().hex[:12]}-{name}"
    size = 0
    try:
        with target.open("wb") as stream:
            while block := await file.read(1024 * 1024):
                size += len(block)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "文件超过 50 MB")
                stream.write(block)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    job_id = _new_job("parse")
    executor.submit(_run, job_id, _parse_job, [target])
    return {"job_id": job_id, "filename": target.name}


@app.post("/api/parse")
def parse_existing():
    paths = supported_files(RAW) if RAW.is_dir() else []
    if not paths:
        raise HTTPException(400, "data/raw 没有受支持的文件")
    job_id = _new_job("parse")
    executor.submit(_run, job_id, _parse_job, paths)
    return {"job_id": job_id}


class BuildRequest(BaseModel):
    chunk_mode: str = "structured"


@app.post("/api/build")
def build(request: BuildRequest):
    if request.chunk_mode not in {"fixed", "structured"}:
        raise HTTPException(400, "chunk_mode 必须为 fixed 或 structured")
    job_id = _new_job("build")
    executor.submit(_run, job_id, _build_job, request.chunk_mode)
    return {"job_id": job_id}


@app.get("/api/jobs")
def jobs():
    records = []
    for path in sorted(JOBS.glob("*.json"), reverse=True)[:30]:
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return records


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    snapshot_id: str | None = None
    answer: bool = True


@app.post("/api/query")
def query(request: QueryRequest):
    from time import monotonic
    start = monotonic()
    try:
        snapshot_id = request.snapshot_id or latest_complete_snapshot(INDEXES)
        if not snapshot_id.replace("-", "").isalnum():
            raise ValueError("无效 snapshot ID")
        chunks = retrieve(INDEXES, snapshot_id, request.question)
        snapshot = json.loads((INDEXES / snapshot_id / "snapshot.json").read_text(encoding="utf-8"))
        names = {item["doc_id"]: item["source_file"] for item in snapshot["sources"]}
        hits = [{"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
                 "source": names[chunk["doc_id"]], "page_start": chunk["page_start"],
                 "page_end": chunk["page_end"], "slide_start": chunk.get("slide_start"),
                 "slide_end": chunk.get("slide_end"), "heading_path": chunk["heading_path"],
                 "score": chunk["score"], "text": chunk["text"]} for chunk in chunks]
        response = answer_question(INDEXES, snapshot_id, request.question, ROOT / "config.yaml", chunks=chunks) if request.answer else None
        _event("query", True, round((monotonic() - start) * 1000))
        return {"snapshot_id": snapshot_id, "answer": response, "hits": hits}
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        _event("query", False, round((monotonic() - start) * 1000))
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/source/{doc_id}")
def source(doc_id: str):
    if not doc_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "无效文档 ID")
    path = PARSED / doc_id / "document.md"
    if not path.is_file():
        raise HTTPException(404, "文档不存在")
    return FileResponse(path, media_type="text/markdown; charset=utf-8")


@app.get("/api/source/{doc_id}/images/{name}")
def source_image(doc_id: str, name: str):
    if not doc_id.replace("-", "").replace("_", "").isalnum() or Path(name).name != name:
        raise HTTPException(400, "无效资产路径")
    path = PARSED / doc_id / "images" / name
    if not path.is_file():
        raise HTTPException(404, "图片不存在")
    return FileResponse(path)


@app.get("/api/metrics")
def metrics():
    events = []
    if EVENTS.is_file():
        for line in EVENTS.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    queries = [event for event in events if event["kind"] == "query"]
    return {"queries": len(queries), "errors": sum(not item["success"] for item in queries),
            "mean_query_ms": round(sum(item.get("elapsed_ms") or 0 for item in queries) / len(queries)) if queries else None}
