"""Small shared LAN interface over the existing parse/build/query pipeline."""

import json
import os
import sys
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
from src.service.conversations import Conversations
from src.utils.model_settings import DEFAULTS, delete_secret, public_settings, save_role, set_secret

ROOT = (Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[2])
BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", ROOT))
RAW = ROOT / "data" / "raw"
PARSED = ROOT / "data" / "parsed"
INDEXES = ROOT / "data" / "indexes"
SERVICE = ROOT / "data" / "service"
JOBS = SERVICE / "jobs"
EVENTS = SERVICE / "events.jsonl"
CATEGORIES = SERVICE / "categories.json"
MAX_UPLOAD = 50 * 1024 * 1024
executor = ThreadPoolExecutor(max_workers=1)
lock = threading.Lock()
app = FastAPI(title="Multimodal RAG LAN")


def _require_local_admin(request: Request) -> None:
    host = request.client.host if request.client else ""
    if host not in {"127.0.0.1", "::1", "testclient"}:
        raise HTTPException(403, "模型与密钥设置只能在服务所在电脑上管理")


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


def _new_job(kind: str, retry: dict | None = None) -> str:
    job_id = uuid.uuid4().hex
    _write_json(JOBS / f"{job_id}.json", {
        "job_id": job_id, "kind": kind, "state": "queued", "stage": "排队中",
        "completed": 0, "total": None, "errors": 0,
        "retry": retry,
        "created_at": _now(), "updated_at": _now(),
    })
    return job_id


def _event(kind: str, success: bool, elapsed_ms: int | None = None) -> None:
    SERVICE.mkdir(parents=True, exist_ok=True)
    with lock, EVENTS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": _now(), "kind": kind, "success": success,
                                 "elapsed_ms": elapsed_ms}, ensure_ascii=False) + "\n")


def _categories() -> dict[str, str]:
    try:
        value = json.loads(CATEGORIES.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _assets(doc_id: str, page_start: int | None, page_end: int | None,
            slide_start: int | None, slide_end: int | None,
            source_sha256: str | None = None) -> list[dict]:
    try:
        manifest = json.loads((PARSED / doc_id / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if source_sha256 and manifest.get("sha256") != source_sha256:
        return []
    result = []
    for asset in manifest.get("image_assets", []):
        number = asset.get("page") if page_start is not None else asset.get("slide")
        first, last = (page_start, page_end) if page_start is not None else (slide_start, slide_end)
        name = Path(str(asset.get("path", ""))).name
        if (isinstance(number, int) and first is not None and first <= number <= last and name
                and (PARSED / doc_id / "images" / name).is_file()):
            result.append({"name": name, "kind": asset.get("kind"), "page": asset.get("page"),
                           "slide": asset.get("slide"),
                           "url": f"/api/source/{doc_id}/images/{name}"})
    return result[:20]


def _parse_job(job_id: str, paths: list[Path], describe_images: bool = False) -> None:
    _job(job_id, state="running", stage="解析中", total=len(paths))
    failures = []
    for number, path in enumerate(paths, 1):
        success = parse_file(path, RAW, PARSED, describe_images=describe_images)
        if not success:
            failures.append(path.name)
        _job(job_id, completed=number, errors=len(failures), current_file=path.name)
    _job(job_id, state="completed" if not failures else "partial", stage="解析完成",
         failed_files=failures)
    _event("parse", not failures)


def _build_job(job_id: str, mode: str) -> None:
    _job(job_id, state="running", stage="准备构建")
    stages = {"chunking": "切块", "bm25": "BM25 分词", "dense": "Qwen3 向量化", "bge_m3": "bge-m3 向量化"}
    def progress(stage: str, completed: int, total: int, errors: int) -> None:
        _job(job_id, stage=stages.get(stage, stage), completed=completed, total=total, errors=errors)
    folder = build_snapshot(PARSED, INDEXES, mode, False, on_progress=progress)
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
    categories = _categories()
    for path in sorted(PARSED.glob("*/manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            result.append({"doc_id": manifest["doc_id"], "source_file": manifest["source_file"],
                           "source_type": manifest["source_type"], "errors": len(manifest.get("errors", [])),
                           "images": len(manifest.get("image_assets", [])),
                           "category": categories.get(manifest["doc_id"], "未分类")})
        except (OSError, KeyError, ValueError):
            continue
    return result


class CategoryRequest(BaseModel):
    category: str = Field(min_length=1, max_length=50)


@app.put("/api/documents/{doc_id}/category")
def set_category(doc_id: str, body: CategoryRequest):
    if not doc_id.replace("-", "").replace("_", "").isalnum() or not (PARSED / doc_id / "manifest.json").is_file():
        raise HTTPException(404, "文档不存在")
    category = body.category.strip()
    if not category:
        raise HTTPException(400, "分类不能为空")
    with lock:
        categories = _categories()
        categories[doc_id] = category
        _write_json(CATEGORIES, categories)
    return {"doc_id": doc_id, "category": category}


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
async def upload(describe_images: bool = False, file: UploadFile = File(...)):
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
    relative = target.relative_to(RAW).as_posix()
    job_id = _new_job("parse", {"paths": [relative], "describe_images": describe_images})
    executor.submit(_run, job_id, _parse_job, [target], describe_images)
    return {"job_id": job_id, "filename": target.name}


@app.post("/api/parse")
def parse_existing(describe_images: bool = False):
    paths = supported_files(RAW) if RAW.is_dir() else []
    if not paths:
        raise HTTPException(400, "data/raw 没有受支持的文件")
    relative_paths = [path.relative_to(RAW).as_posix() for path in paths]
    job_id = _new_job("parse", {"paths": relative_paths, "describe_images": describe_images})
    executor.submit(_run, job_id, _parse_job, paths, describe_images)
    return {"job_id": job_id}


class BuildRequest(BaseModel):
    chunk_mode: str = "structured"


@app.post("/api/build")
def build(request: BuildRequest):
    if request.chunk_mode not in {"fixed", "structured"}:
        raise HTTPException(400, "chunk_mode 必须为 fixed 或 structured")
    job_id = _new_job("build", {"chunk_mode": request.chunk_mode})
    executor.submit(_run, job_id, _build_job, request.chunk_mode)
    return {"job_id": job_id}


@app.get("/api/jobs")
def jobs():
    records = []
    for path in JOBS.glob("*.json"):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(records, key=lambda item: item.get("updated_at", ""), reverse=True)[:30]


def _load_job(job_id: str) -> dict:
    if not job_id.isalnum():
        raise HTTPException(404, "任务不存在")
    path = JOBS / f"{job_id}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(404, "任务不存在")


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str):
    record = _load_job(job_id)
    if record.get("state") not in {"failed", "partial", "interrupted"}:
        raise HTTPException(409, "只有失败、部分成功或中断的任务可以重试")
    retry = record.get("retry")
    if not isinstance(retry, dict):
        raise HTTPException(409, "旧任务缺少可重试参数，请重新发起")
    if record.get("kind") == "build":
        mode = retry.get("chunk_mode")
        if mode not in {"fixed", "structured"}:
            raise HTTPException(409, "构建重试参数无效")
        new_id = _new_job("build", retry)
        executor.submit(_run, new_id, _build_job, mode)
    elif record.get("kind") == "parse":
        names = retry.get("paths")
        if not isinstance(names, list) or not names:
            raise HTTPException(409, "解析重试参数无效")
        paths = []
        raw_root = RAW.resolve()
        for name in names:
            candidate = (RAW / str(name)).resolve()
            if candidate.parent != raw_root and raw_root not in candidate.parents:
                raise HTTPException(409, "解析重试路径越界")
            if not candidate.is_file():
                raise HTTPException(409, f"待重试文件不存在：{name}")
            paths.append(candidate)
        describe = bool(retry.get("describe_images", False))
        new_id = _new_job("parse", retry)
        executor.submit(_run, new_id, _parse_job, paths, describe)
    else:
        raise HTTPException(409, "该任务类型不支持重试")
    return {"job_id": new_id, "retried_from": job_id}


@app.get("/api/jobs/{job_id}/errors")
def job_errors(job_id: str):
    record = _load_job(job_id)
    if record.get("kind") == "parse":
        return {"job_id": job_id, "count": record.get("errors", 0),
                "errors": [{"type": "parse_failed", "file": name}
                           for name in record.get("failed_files", [])[:100]]}
    snapshot_id = record.get("snapshot_id")
    if record.get("kind") != "build" or not isinstance(snapshot_id, str):
        return {"job_id": job_id, "count": record.get("errors", 0), "errors": []}
    if Path(snapshot_id).name != snapshot_id:
        raise HTTPException(409, "任务 snapshot_id 无效")
    error_path = INDEXES / snapshot_id / "errors.jsonl"
    errors = []
    if error_path.is_file():
        for line in error_path.read_text(encoding="utf-8").splitlines()[:100]:
            try:
                item = json.loads(line)
                errors.append({key: item[key] for key in ("stage", "type", "message", "chunk_id", "doc_id")
                               if key in item})
            except ValueError:
                continue
    return {"job_id": job_id, "count": record.get("errors", len(errors)), "errors": errors}


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    snapshot_id: str | None = None
    answer: bool = True
    conversation_id: str | None = None


class ModelSettingsRequest(BaseModel):
    role: str
    provider: str
    base_url: str
    model: str
    timeout: float = Field(ge=1, le=300)
    api_key: str | None = Field(default=None, max_length=4096)
    clear_key: bool = False


@app.get("/api/admin/model-settings")
def model_settings(request: Request):
    _require_local_admin(request)
    return public_settings()


@app.put("/api/admin/model-settings")
def update_model_settings(body: ModelSettingsRequest, request: Request):
    _require_local_admin(request)
    if body.role not in DEFAULTS:
        raise HTTPException(400, "未知模型角色")
    if body.api_key and body.clear_key:
        raise HTTPException(400, "不能同时设置和清除密钥")
    try:
        save_role(body.role, {"provider": body.provider, "base_url": body.base_url,
                              "model": body.model, "timeout": body.timeout})
        if body.api_key:
            set_secret(body.role, body.api_key)
        elif body.clear_key:
            delete_secret(body.role)
        return public_settings()[body.role]
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/query")
def query(request: QueryRequest):
    from time import monotonic
    start = monotonic()
    try:
        snapshot_id = request.snapshot_id or latest_complete_snapshot(INDEXES)
        if not snapshot_id.replace("-", "").isalnum():
            raise ValueError("无效 snapshot ID")
        conversations = Conversations(SERVICE / "conversations.sqlite3")
        history = conversations.history(request.conversation_id)
        previous_questions = "\n".join(item["question"] for item in history[-6:])[-8000:]
        retrieval_question = (previous_questions + "\n当前问题：" + request.question) if previous_questions else request.question
        chunks = retrieve(INDEXES, snapshot_id, retrieval_question)
        snapshot = json.loads((INDEXES / snapshot_id / "snapshot.json").read_text(encoding="utf-8"))
        names = {item["doc_id"]: item["source_file"] for item in snapshot["sources"]}
        source_hashes = {item["doc_id"]: item.get("sha256") for item in snapshot["sources"]}
        hits = [{"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
                 "source": names[chunk["doc_id"]], "page_start": chunk["page_start"],
                 "page_end": chunk["page_end"], "slide_start": chunk.get("slide_start"),
                 "slide_end": chunk.get("slide_end"), "heading_path": chunk["heading_path"],
                 "score": chunk["score"], "text": chunk["text"],
                 "images": _assets(chunk["doc_id"], chunk["page_start"], chunk["page_end"],
                                   chunk.get("slide_start"), chunk.get("slide_end"),
                                   source_hashes.get(chunk["doc_id"]))} for chunk in chunks]
        config_path = ROOT / "config.yaml"
        if not config_path.is_file():
            config_path = BUNDLE_ROOT / "config.yaml"
        usage: dict[str, int] = {}
        response = answer_question(INDEXES, snapshot_id, request.question, config_path,
                                   chunks=chunks, history=history,
                                   on_usage=usage.update) if request.answer else None
        conversation_id, turn_id = conversations.append(request.conversation_id, request.question,
                                                         response or "仅检索，未调用 DeepSeek", snapshot_id, hits)
        _event("query", True, round((monotonic() - start) * 1000))
        if usage:
            _token_event(usage)
        return {"snapshot_id": snapshot_id, "answer": response, "hits": hits,
                "conversation_id": conversation_id, "turn_id": turn_id, "usage": usage or None}
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
    token_events = [event for event in events if event["kind"] == "deepseek_usage"]
    feedback = Conversations(SERVICE / "conversations.sqlite3").feedback_counts()
    return {"queries": len(queries), "errors": sum(not item["success"] for item in queries),
            "mean_query_ms": round(sum(item.get("elapsed_ms") or 0 for item in queries) / len(queries)) if queries else None,
            "deepseek_tokens": sum(item.get("total_tokens", 0) for item in token_events),
            "token_reported_calls": len(token_events), "feedback": feedback}


def _token_event(usage: dict[str, int]) -> None:
    SERVICE.mkdir(parents=True, exist_ok=True)
    with lock, EVENTS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": _now(), "kind": "deepseek_usage", **usage}) + "\n")


class FeedbackRequest(BaseModel):
    rating: int


@app.post("/api/turns/{turn_id}/feedback")
def save_feedback(turn_id: int, body: FeedbackRequest):
    try:
        Conversations(SERVICE / "conversations.sqlite3").rate(turn_id, body.rating)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"turn_id": turn_id, "rating": body.rating}


def _conversation_id(value: str) -> str:
    if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
        raise HTTPException(400, "无效会话 ID")
    return value


@app.get("/api/conversations")
def conversation_list():
    return Conversations(SERVICE / "conversations.sqlite3").list()


@app.get("/api/conversations/{conversation_id}")
def conversation_get(conversation_id: str):
    try:
        return Conversations(SERVICE / "conversations.sqlite3").get(_conversation_id(conversation_id))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.delete("/api/conversations/{conversation_id}")
def conversation_delete(conversation_id: str):
    try:
        Conversations(SERVICE / "conversations.sqlite3").delete(_conversation_id(conversation_id))
        return {"deleted": True}
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
