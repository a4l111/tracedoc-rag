"""Build one immutable chunk snapshot and three indexes from parsed artifacts."""

import hashlib
import json
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.progress import track

from src.chunker.chunks import make_chunks
from src.utils import moark

MODELS = {
    "dense": (moark.MODEL, moark.DIMENSION, "dense.faiss", "dense_meta.json"),
    "bge_m3": ("BAAI/bge-m3", 1024, "bge_m3.faiss", "bge_meta.json"),
}
LOCAL_BGE_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / "bge-m3"


def model_source(route: str) -> tuple[str, str, str | None]:
    """Return model loader input and identity for this build."""
    if route != "bge_m3":
        raise ValueError("Only bge-m3 uses local model weights")
    weight = LOCAL_BGE_DIR / "pytorch_model.bin"
    if weight.is_file():
        digest = hashlib.sha256()
        with weight.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return str(LOCAL_BGE_DIR), "local", digest.hexdigest()
    return MODELS[route][0], "huggingface", None


def tokenize(text: str) -> list[str]:
    """The exact tokenizer shared by BM25 build and query."""
    return [word.strip().lower() for word in jieba.cut(text) if word.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_chunks(parsed_dir: Path, mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    chunks = []
    sources = []
    errors = []
    for manifest_path in sorted(parsed_dir.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            markdown_path = manifest_path.parent / "document.md"
            markdown = markdown_path.read_text(encoding="utf-8")
            file_chunks = make_chunks(markdown, manifest, mode)
            chunks.extend(file_chunks)
            sources.append({"doc_id": manifest["doc_id"], "sha256": manifest["sha256"],
                            "source_file": manifest["source_file"],
                            "markdown_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
                            "chunk_count": len(file_chunks)})
        except Exception as exc:
            errors.append({"stage": "chunking", "file": str(manifest_path),
                           "type": type(exc).__name__, "message": str(exc)})
    return chunks, sources, errors


def _build_vector_index(
    folder: Path, route: str, chunks: list[dict[str, Any]], snapshot_id: str,
    snapshot_sha256: str, errors: list[dict[str, Any]], download_models: bool,
    on_progress: Callable[[str, int, int, int], None] | None = None,
) -> dict[str, Any]:
    model_name, dimension, index_name, meta_name = MODELS[route]
    index = faiss.IndexFlatIP(dimension)
    successful_ids = []
    load_error = None
    loader_name, source_kind, weight_sha256 = model_source(route) if route != "dense" else (None, "moark", None)
    try:
        if route == "dense":
            model = moark.client()
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(loader_name, local_files_only=not download_models)
    except Exception as exc:
        model = None
        load_error = f"{type(exc).__name__}: {exc}"
    if route == "dense" and model is not None:
        processed = 0
        for start in track(range(0, len(chunks), moark.EMBED_BATCH_SIZE),
                           description=f"Embedding {route}", console=Console()):
            batch = chunks[start:start + moark.EMBED_BATCH_SIZE]
            try:
                vectors = (moark.embed_many(model, [chunk["text"] for chunk in batch])
                           if len(batch) > 1 else [moark.embed(model, batch[0]["text"])])
                for chunk, vector in zip(batch, vectors):
                    index.add(vector.reshape(1, dimension))
                    successful_ids.append(chunk["chunk_id"])
            except Exception:
                # Retry individually so one bad chunk cannot discard the rest of a batch.
                for chunk in batch:
                    try:
                        vector = moark.embed(model, chunk["text"])
                        index.add(vector.reshape(1, dimension))
                        successful_ids.append(chunk["chunk_id"])
                    except Exception as exc:
                        errors.append({"stage": route, "chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
                                       "type": type(exc).__name__, "message": str(exc)})
            processed += len(batch)
            if on_progress:
                on_progress(route, processed, len(chunks), len(errors))
    else:
      for processed, chunk in enumerate(track(chunks, description=f"Embedding {route}", console=Console()), 1):
        try:
            if model is None:
                raise RuntimeError(f"Model unavailable: {load_error}")
            vector = np.asarray(model.encode(chunk["text"], normalize_embeddings=True), dtype="float32")
            if vector.shape != (dimension,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                raise ValueError(f"Invalid embedding shape or values: {vector.shape}")
            index.add(vector.reshape(1, dimension))
            successful_ids.append(chunk["chunk_id"])
        except Exception as exc:
            errors.append({"stage": route, "chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
                           "type": type(exc).__name__, "message": str(exc)})
        if on_progress:
            on_progress(route, processed, len(chunks), len(errors))
    faiss.write_index(index, str(folder / index_name))
    meta = {"snapshot_id": snapshot_id, "chunks_sha256": snapshot_sha256,
            "model": model_name, "dimension": dimension, "chunk_ids": successful_ids,
            "model_source": source_kind, "base_url": moark.embedding_base_url() if route == "dense" else None,
            "weight_sha256": weight_sha256,
            "indexed_count": len(successful_ids), "failed_count": len(chunks) - len(successful_ids),
            "model_load_error": load_error}
    _write_json(folder / meta_name, meta)
    return {"model": model_name, "indexed_count": len(successful_ids),
            "failed_count": len(chunks) - len(successful_ids), "model_load_error": load_error}


def build(parsed_dir: Path, indexes_dir: Path, mode: str = "structured", download_models: bool = False,
          on_progress: Callable[[str, int, int, int], None] | None = None) -> Path:
    if not parsed_dir.is_dir() or not any(parsed_dir.glob("*/manifest.json")):
        raise ValueError(f"No parsed manifests found in {parsed_dir}")
    if mode not in {"fixed", "structured"}:
        raise ValueError(f"Unknown chunk mode: {mode}")
    moark.client()  # Check the remote credential before creating a snapshot.
    snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = indexes_dir / snapshot_id
    folder.mkdir(parents=True, exist_ok=False)
    chunks, sources, errors = _load_chunks(parsed_dir, mode)
    if not chunks:
        raise ValueError("No usable chunks found in parsed artifacts")
    if on_progress:
        on_progress("chunking", len(chunks), len(chunks), len(errors))
    chunk_bytes = b"".join((json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8") for chunk in chunks)
    (folder / "chunks.jsonl").write_bytes(chunk_bytes)
    chunks_sha256 = hashlib.sha256(chunk_bytes).hexdigest()
    snapshot = {"snapshot_id": snapshot_id, "created_at": datetime.now(timezone.utc).isoformat(),
                "chunk_mode": mode, "chunk_count": len(chunks), "chunks_sha256": chunks_sha256,
                "sources": sources,
                "model_config": {"dense": {"provider": "moark", "model": moark.MODEL,
                                           "dimension": moark.DIMENSION, "base_url": moark.BASE_URL},
                                 "reranker": {"provider": "moark", "model": moark.RERANK_MODEL}}}
    _write_json(folder / "snapshot.json", snapshot)

    tokens = [tokenize(chunk["text"]) for chunk in track(chunks, description="Tokenizing BM25", console=Console())]
    if on_progress:
        on_progress("bm25", len(chunks), len(chunks), len(errors))
    bm25 = BM25Okapi(tokens)
    with (folder / "bm25.pkl").open("wb") as stream:
        pickle.dump({"snapshot_id": snapshot_id, "chunks_sha256": chunks_sha256,
                     "chunk_ids": [chunk["chunk_id"] for chunk in chunks], "bm25": bm25}, stream)

    routes = {"bm25": {"indexed_count": len(chunks), "failed_count": 0}}
    for route in MODELS:
        routes[route] = _build_vector_index(folder, route, chunks, snapshot_id,
                                            chunks_sha256, errors, download_models, on_progress)
    with (folder / "errors.jsonl").open("w", encoding="utf-8") as stream:
        for error in errors:
            stream.write(json.dumps(error, ensure_ascii=False) + "\n")
    _write_json(folder / "build_report.json", {"snapshot_id": snapshot_id, "chunk_count": len(chunks),
                                                "chunk_mode": mode, "routes": routes,
                                                "error_count": len(errors),
                                                "complete": all(routes[name]["indexed_count"] == len(chunks) for name in routes)})
    return folder
