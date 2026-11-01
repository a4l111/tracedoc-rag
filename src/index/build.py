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
from src.index import visual
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
            for observation in manifest.get("photo_inventory", []):
                photos = observation.get("photos", [])
                if not photos:
                    continue
                page = observation["page"]
                details = "; ".join(
                    f"photo {photo['photo_number']}: {photo['people_count'] if photo['people_count'] is not None else 'unknown'} "
                    f"{'estimated' if photo['estimated'] else 'observed'} people, "
                    f"single person {photo['people_count'] == 1 and not photo['estimated']}"
                    for photo in photos)
                text = (f"Photo inventory, page {page}, source asset {observation['asset_id']}; "
                        f"model generated and unreviewed. {details}")
                chunk_id = hashlib.sha256(
                    f"{manifest['doc_id']}\0photo-inventory\0{page}\0{text}".encode("utf-8")
                ).hexdigest()[:24]
                file_chunks.append({"chunk_id": chunk_id,
                                    "display_name": f"{Path(manifest['source_file']).name} · photos · page {page}",
                                    "doc_id": manifest["doc_id"], "text": text,
                                    "page_start": page, "page_end": page,
                                    "slide_start": None, "slide_end": None,
                                    "heading_path": ["Photo inventory"],
                                    "image_asset_id": observation["asset_id"],
                                    "review_status": "unreviewed"})
            observations = manifest.get("photo_inventory", [])
            expected_photo_pages = sum(asset.get("kind") == "page_render_for_vision"
                                       for asset in manifest.get("image_assets", []))
            inventory_complete = len(observations) == expected_photo_pages and all(
                photo.get("people_count") is not None
                for entry in observations for photo in entry.get("photos", []))
            singles = [(entry["page"], photo["photo_number"], entry["asset_id"])
                       for entry in observations for photo in entry.get("photos", [])
                       if photo.get("people_count") == 1 and not photo.get("estimated")]
            counts = [(photo["people_count"], entry["page"], photo["photo_number"],
                       entry["asset_id"], photo.get("estimated", False))
                      for entry in observations for photo in entry.get("photos", [])
                      if photo.get("people_count") is not None]
            summaries = []
            if inventory_complete and singles:
                references = ", ".join(f"page {page} photo {number}" for page, number, _ in singles)
                summaries.append(("Single-person photos", sorted({page for page, _, _ in singles}),
                                  [asset for _, _, asset in singles],
                                  f"Across all photographs in this document, {len(singles)} photos contain "
                                  f"exactly one visible person: {references}. Model-generated photo counts; "
                                  "not manually reviewed."))
            if inventory_complete and counts:
                winner = max(counts)
                if sum(count == winner[0] for count, *_ in counts) == 1:
                    count, page, number, asset, estimated = winner
                    summaries.append(("Most people in a photo", [page], [asset],
                                      f"Across all photographs in this document, the photo with the most "
                                      f"visible people is on page {page}, photo {number}: "
                                      f"{count} {'estimated' if estimated else 'observed'} people. "
                                      "Model-generated photo count; not manually reviewed."))
            for title, pages, assets, text in summaries:
                chunk_id = hashlib.sha256(
                    f"{manifest['doc_id']}\0photo-summary\0{title}\0{text}".encode("utf-8")
                ).hexdigest()[:24]
                file_chunks.append({"chunk_id": chunk_id,
                                    "display_name": f"{Path(manifest['source_file']).name} · {title}",
                                    "doc_id": manifest["doc_id"], "text": text,
                                    "page_start": None, "page_end": None,
                                    "slide_start": None, "slide_end": None,
                                    "heading_path": ["Photo inventory", title],
                                    "evidence_pages": pages, "image_asset_ids": assets,
                                    "review_status": "unreviewed"})
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
    attempted_count = len(chunks)
    routes = {}
    for route in MODELS:
        routes[route] = _build_vector_index(folder, route, chunks, snapshot_id,
                                            "", errors, download_models, on_progress)
    metadata = {route: json.loads((folder / MODELS[route][3]).read_text(encoding="utf-8"))
                for route in MODELS}
    successful_ids = {route: set(meta["chunk_ids"]) for route, meta in metadata.items()}
    common_ids = set.intersection(*successful_ids.values())
    usable_chunks = [chunk for chunk in chunks if chunk["chunk_id"] in common_ids]
    if not usable_chunks:
        errors.append({"stage": "alignment", "type": "NoCommonChunks",
                       "message": "No chunk has valid embeddings in every vector route"})
    chunk_bytes = b"".join((json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")
                           for chunk in usable_chunks)
    (folder / "chunks.jsonl").write_bytes(chunk_bytes)
    chunks_sha256 = hashlib.sha256(chunk_bytes).hexdigest()
    aligned_ids = [chunk["chunk_id"] for chunk in usable_chunks]
    repair_records = []
    for route, meta in metadata.items():
        _, dimension, index_name, meta_name = MODELS[route]
        old_ids = meta["chunk_ids"]
        positions = {chunk_id: position for position, chunk_id in enumerate(old_ids)}
        old_index = faiss.read_index(str(folder / index_name))
        aligned_index = faiss.IndexFlatIP(dimension)
        for chunk_id in aligned_ids:
            aligned_index.add(old_index.reconstruct(positions[chunk_id]).reshape(1, dimension))
        faiss.write_index(aligned_index, str(folder / index_name))
        meta.update({"chunks_sha256": chunks_sha256, "chunk_ids": aligned_ids,
                     "indexed_count": len(aligned_ids), "failed_count": attempted_count - len(aligned_ids)})
        _write_json(folder / meta_name, meta)
        routes[route].update({"indexed_count": len(aligned_ids),
                              "failed_count": attempted_count - len(aligned_ids)})
    for chunk in chunks:
        if chunk["chunk_id"] not in common_ids:
            repair_records.append({"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
                                   "failed_routes": [route for route, ids in successful_ids.items()
                                                     if chunk["chunk_id"] not in ids],
                                   "action": "excluded_from_all_indexes",
                                   "result": "three_route_alignment_preserved"})
    for error in errors:
        if error.get("stage") == "chunking":
            repair_records.append({"file": error.get("file"), "failed_stage": "chunking",
                                   "action": "skipped_source_for_this_snapshot",
                                   "result": "other_sources_continue"})
    with (folder / "repair_log.jsonl").open("w", encoding="utf-8") as stream:
        for record in repair_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    visual_result: dict[str, Any] | None = None
    try:
        visual_result = visual.build_visual_index(parsed_dir, folder, on_progress)
    except Exception as exc:
        visual_result = {"failed": True, "error": f"{type(exc).__name__}: {exc}"}
        errors.append({"stage": "visual", "type": type(exc).__name__, "message": str(exc)})
    snapshot = {"snapshot_id": snapshot_id, "created_at": datetime.now(timezone.utc).isoformat(),
                "chunk_mode": mode, "chunk_count": len(usable_chunks), "attempted_chunk_count": attempted_count,
                "excluded_chunk_count": attempted_count - len(usable_chunks), "chunks_sha256": chunks_sha256,
                "sources": sources,
                "model_config": {"dense": {"provider": "moark", "model": moark.MODEL,
                                           "dimension": moark.DIMENSION, "base_url": moark.BASE_URL},
                                 "reranker": {"provider": "moark", "model": moark.RERANK_MODEL},
                                 "visual_embedding": {"provider": "moark", "model": visual.VISUAL_MODEL,
                                                      "dimension": visual.DIMENSION}}}
    _write_json(folder / "snapshot.json", snapshot)
    if usable_chunks:
        tokens = [tokenize(chunk["text"]) for chunk in track(usable_chunks, description="Tokenizing BM25", console=Console())]
        bm25 = BM25Okapi(tokens)
        with (folder / "bm25.pkl").open("wb") as stream:
            pickle.dump({"snapshot_id": snapshot_id, "chunks_sha256": chunks_sha256,
                         "chunk_ids": aligned_ids, "bm25": bm25}, stream)
    if on_progress:
        on_progress("bm25", len(usable_chunks), len(usable_chunks), len(errors))
    routes["bm25"] = {"indexed_count": len(usable_chunks),
                      "failed_count": attempted_count - len(usable_chunks)}
    with (folder / "errors.jsonl").open("w", encoding="utf-8") as stream:
        for error in errors:
            stream.write(json.dumps(error, ensure_ascii=False) + "\n")
    _write_json(folder / "build_report.json", {"snapshot_id": snapshot_id, "chunk_count": len(usable_chunks),
                                                "attempted_chunk_count": attempted_count,
                                                "excluded_chunk_count": attempted_count - len(usable_chunks),
                                                "chunk_mode": mode, "routes": routes,
                                                "visual": visual_result,
                                                "error_count": len(errors),
                                                "partial": bool(errors or repair_records),
                                                "complete": bool(usable_chunks) and all(
                                                    routes[name]["indexed_count"] == len(usable_chunks)
                                                    for name in routes)})
    return folder
