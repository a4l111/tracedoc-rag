"""Search exactly one verified build snapshot."""

import hashlib
import json
import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

from src.index.build import MODELS, LOCAL_BGE_DIR, tokenize
from src.utils import moark

ROUTES = ("bm25", "dense", "bge_m3")


@lru_cache(maxsize=2)
def _bge_model(loader_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(loader_name, local_files_only=True)


def search(indexes_dir: Path, snapshot_id: str, route: str, question: str, top_k: int = 5) -> list[dict]:
    if not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", snapshot_id):
        raise ValueError("Invalid snapshot_id")
    if route not in {"bm25", *MODELS}:
        raise ValueError(f"Unknown route: {route}")
    if not question.strip() or top_k < 1:
        raise ValueError("Question must be nonempty and top_k must be positive")
    folder = indexes_dir / snapshot_id
    snapshot = json.loads((folder / "snapshot.json").read_text(encoding="utf-8"))
    if snapshot["snapshot_id"] != snapshot_id:
        raise ValueError("Snapshot ID mismatch")
    chunk_bytes = (folder / "chunks.jsonl").read_bytes()
    if hashlib.sha256(chunk_bytes).hexdigest() != snapshot["chunks_sha256"]:
        raise ValueError("Chunk snapshot checksum mismatch")
    chunks = [json.loads(line) for line in chunk_bytes.splitlines()]
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    if route == "bm25":
        # Only load pickle files produced locally by this project.
        with (folder / "bm25.pkl").open("rb") as stream:
            state = pickle.load(stream)
        if state["snapshot_id"] != snapshot_id or state["chunks_sha256"] != snapshot["chunks_sha256"]:
            raise ValueError("BM25 snapshot mismatch")
        scores = state["bm25"].get_scores(tokenize(question))
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        return [{**by_id[state["chunk_ids"][i]], "score": float(scores[i])} for i in ranked]

    _, _, index_name, meta_name = MODELS[route]
    meta = json.loads((folder / meta_name).read_text(encoding="utf-8"))
    if meta["snapshot_id"] != snapshot_id or meta["chunks_sha256"] != snapshot["chunks_sha256"]:
        raise ValueError("Vector index snapshot mismatch")
    model_name = meta["model"]
    dimension = meta["dimension"]
    allowed_models = {"dense": {moark.MODEL},
                      "bge_m3": {"BAAI/bge-m3"}}
    if model_name not in allowed_models[route]:
        raise ValueError("Unexpected model for this route")
    if route == "dense" and (meta.get("model_source") != "moark" or
                              meta.get("base_url") != moark.embedding_base_url()):
        raise ValueError("Dense index was not built with the configured MoArk service")
    index = faiss.read_index(str(folder / index_name))
    if index.ntotal != len(meta["chunk_ids"]) or index.d != dimension:
        raise ValueError("Vector index metadata mismatch")
    if index.ntotal == 0:
        raise ValueError(f"{route} has no indexed chunks in this snapshot")
    if route == "dense":
        vector = moark.embed(moark.client(), question)
    else:
        loader_name = model_name
        if meta.get("model_source") == "local":
            weight = LOCAL_BGE_DIR / "pytorch_model.bin"
            if not weight.is_file():
                raise ValueError("Local bge-m3 weights are missing")
            digest = hashlib.sha256()
            with weight.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != meta.get("weight_sha256"):
                raise ValueError("Local bge-m3 weights do not match this snapshot")
            loader_name = str(LOCAL_BGE_DIR)
        model = _bge_model(loader_name)
        vector = np.asarray(model.encode(question, normalize_embeddings=True), dtype="float32")
    if vector.shape != (dimension,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError("Invalid query embedding")
    scores, positions = index.search(vector.reshape(1, dimension), min(top_k, index.ntotal))
    return [{**by_id[meta["chunk_ids"][int(position)]], "score": float(score)}
            for score, position in zip(scores[0], positions[0]) if position >= 0]


def latest_complete_snapshot(indexes_dir: Path) -> str:
    """Select the newest complete build snapshot without reading source documents."""
    for folder in sorted(indexes_dir.iterdir(), reverse=True) if indexes_dir.is_dir() else []:
        report_path = folder / "build_report.json"
        if not folder.is_dir() or not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        meta_path = folder / "dense_meta.json"
        if (report.get("snapshot_id") == folder.name and report.get("complete")
                and report.get("routes", {}).get("dense", {}).get("model") == MODELS["dense"][0]
                and meta_path.is_file()):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            snapshot_path = folder / "snapshot.json"
            if not snapshot_path.is_file():
                continue
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if (meta.get("model_source") == "moark" and meta.get("base_url") == moark.embedding_base_url()
                    and snapshot.get("model_config", {}).get("reranker") ==
                    {"provider": "moark", "model": moark.RERANK_MODEL}):
                return folder.name
    raise FileNotFoundError("No complete MoArk Qwen3-Embedding-8B snapshot found; run build")


def fuse_and_rerank(question: str, rankings: dict[str, list[dict]],
                    on_step: Callable[[str], None] | None = None) -> list[dict]:
    """Fuse three precomputed Top-20 rankings with RRF and rerank the Top 50."""
    fused: dict[str, dict] = {}
    for route in ROUTES:
        if route not in rankings:
            raise ValueError(f"Missing ranking: {route}")
        for rank, chunk in enumerate(rankings[route][:20], start=1):
            entry = fused.setdefault(chunk["chunk_id"], {"chunk": chunk, "rrf_score": 0.0})
            entry["rrf_score"] += 1.0 / (60 + rank)
    candidates = sorted(fused.values(), key=lambda item: (-item["rrf_score"], item["chunk"]["chunk_id"]))[:50]
    if not candidates:
        return []

    if on_step:
        on_step("Reranking candidates")
    rerank_api, rerank_endpoint = moark.rerank_client()
    scores = moark.rerank(rerank_api, question, [item["chunk"]["text"] for item in candidates], rerank_endpoint)
    if (len(scores) != len(candidates) or np.isnan(scores).any() or np.isposinf(scores).any()
            or np.isfinite(scores).sum() < min(5, len(candidates))):
        raise ValueError("Invalid reranker scores")
    ranked = sorted(zip(candidates, scores), key=lambda pair: (-float(pair[1]), pair[0]["chunk"]["chunk_id"]))
    return [{**item["chunk"], "score": float(score)} for item, score in ranked[:5]]


_TECH_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def _technical_terms(question: str) -> str:
    """Extract ASCII technical/English terms for a conservative re-query."""
    terms = " ".join(_TECH_TERM_RE.findall(question)).strip()
    return terms if terms and terms != question.strip() else ""


def query(indexes_dir: Path, snapshot_id: str, question: str,
          on_step: Callable[[str], None] | None = None, expand: bool = True) -> list[dict]:
    """Retrieve 20 per route, fuse 50 by RRF, and rerank to five.

    When the first pass returns fewer than two chunks, conservatively re-query
    with the technical/English terms extracted from the question and merge any
    new chunks. Original terms are always retained and no terms are invented.
    """
    if not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", snapshot_id):
        raise ValueError("Invalid snapshot_id")
    if not question.strip():
        raise ValueError("Question must be nonempty")
    report = json.loads((indexes_dir / snapshot_id / "build_report.json").read_text(encoding="utf-8"))
    if report.get("snapshot_id") != snapshot_id or not report.get("complete"):
        raise ValueError("Query requires a complete matching snapshot")
    if report.get("routes", {}).get("dense", {}).get("model") != MODELS["dense"][0]:
        raise ValueError("Snapshot uses an unsupported dense model; rebuild with Qwen3-Embedding-8B")
    snapshot = json.loads((indexes_dir / snapshot_id / "snapshot.json").read_text(encoding="utf-8"))
    if snapshot.get("model_config", {}).get("reranker") != {"provider": "moark", "model": moark.RERANK_MODEL}:
        raise ValueError("Snapshot reranker configuration does not match MoArk; rebuild")
    rankings = {}
    for route in ROUTES:
        if on_step:
            on_step(f"Searching {route}")
        rankings[route] = search(indexes_dir, snapshot_id, route, question, top_k=20)
    hits = fuse_and_rerank(question, rankings, on_step)
    if expand and len(hits) < 2:
        technical = _technical_terms(question)
        if technical:
            if on_step:
                on_step("Expanding query with technical terms")
            extra_rankings = {}
            for route in ROUTES:
                extra_rankings[route] = search(indexes_dir, snapshot_id, route, technical, top_k=20)
            extra = fuse_and_rerank(technical, extra_rankings, on_step)
            known = {chunk["chunk_id"] for chunk in hits}
            hits = (hits + [chunk for chunk in extra if chunk["chunk_id"] not in known])[:5]
    return hits
