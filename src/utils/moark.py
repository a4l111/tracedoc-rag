"""Qwen3 text embeddings and reranking served by MoArk APIs."""

from typing import Any, Dict

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from src.utils.model_settings import get_api_key, load_settings

BASE_URL = "https://api.moark.com/v1"
MODEL = "Qwen3-Embedding-8B"
DIMENSION = 4096
RERANK_MODEL = "Qwen3-Reranker-8B"
RERANK_BATCH_SIZE = 20
RERANK_STRATEGY = "batched-tournament-v1"
EMBED_BATCH_SIZE = 16


def embedding_base_url() -> str:
    return load_settings()["embedding"]["base_url"]


def client() -> OpenAI:
    load_dotenv()
    settings = load_settings()["embedding"]
    return OpenAI(api_key=get_api_key("embedding"), base_url=settings["base_url"],
                  timeout=settings["timeout"], max_retries=2)


def rerank_client() -> tuple[OpenAI, str]:
    """MoArk serves reranking at /v1/rerank with the same access token."""
    load_dotenv()
    settings = load_settings()["reranker"]
    return OpenAI(api_key=get_api_key("reranker"), base_url=settings["base_url"],
                  timeout=settings["timeout"], max_retries=2), "/rerank"


def embed(client: OpenAI, text: str) -> np.ndarray:
    response = client.embeddings.create(model=load_settings()["embedding"]["model"], input=text, dimensions=DIMENSION)
    vector = np.asarray(response.data[0].embedding, dtype="float32")
    norm = np.linalg.norm(vector)
    if vector.shape != (DIMENSION,) or not np.isfinite(vector).all() or not np.isfinite(norm) or norm == 0:
        raise ValueError(f"Invalid MoArk embedding: shape={vector.shape}")
    return vector / norm


def embed_many(client: OpenAI, texts: list[str]) -> list[np.ndarray]:
    """Embed a batch while preserving input order and validating every vector."""
    if not texts:
        return []
    response = client.embeddings.create(model=load_settings()["embedding"]["model"], input=texts, dimensions=DIMENSION)
    vectors: list[np.ndarray | None] = [None] * len(texts)
    for fallback_index, item in enumerate(response.data):
        index = getattr(item, "index", fallback_index)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= len(texts):
            raise ValueError("Invalid MoArk embedding result index")
        if vectors[index] is not None:
            raise ValueError("Duplicate MoArk embedding result index")
        vector = np.asarray(item.embedding, dtype="float32")
        norm = np.linalg.norm(vector)
        if vector.shape != (DIMENSION,) or not np.isfinite(vector).all() or not np.isfinite(norm) or norm == 0:
            raise ValueError(f"Invalid MoArk embedding: shape={vector.shape}")
        vectors[index] = vector / norm
    if any(vector is None for vector in vectors):
        raise ValueError("MoArk embedding omitted inputs")
    return [vector for vector in vectors if vector is not None]


def _rerank_batch(client: OpenAI, question: str, documents: list[str], endpoint: str) -> np.ndarray:
    response = client.post(endpoint, cast_to=Dict[str, Any], body={
        "model": load_settings()["reranker"]["model"], "query": question, "documents": documents,
    })
    scores = np.empty(len(documents), dtype="float32")
    results = response.get("results", [])
    seen = set()
    for result in results:
        index = result.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= len(documents) or index in seen:
            raise ValueError("Invalid MoArk rerank result index")
        score = float(result["relevance_score"])
        if not np.isfinite(score):
            raise ValueError("Invalid MoArk rerank score")
        scores[index] = score
        seen.add(index)
    if len(seen) != len(documents):
        raise ValueError("MoArk rerank omitted candidates")
    return scores


def rerank(client: OpenAI, question: str, documents: list[str], endpoint: str) -> np.ndarray:
    """Return comparable finalist scores; large candidate sets use a two-stage tournament."""
    scores, _ = rerank_with_diagnostics(client, question, documents, endpoint)
    return scores


def rerank_with_diagnostics(client: OpenAI, question: str, documents: list[str],
                            endpoint: str) -> tuple[np.ndarray, list[dict]]:
    """Rerank while returning numeric tournament diagnostics without document text."""
    if not documents:
        return np.array([], dtype="float32"), []
    if len(documents) <= RERANK_BATCH_SIZE:
        scores = _rerank_batch(client, question, documents, endpoint)
        order = sorted(range(len(documents)), key=lambda index: (-float(scores[index]), index))
        ranks = {index: rank for rank, index in enumerate(order, start=1)}
        return scores, [{"candidate_index": index, "initial_batch": 1,
                         "initial_score": float(scores[index]), "initial_rank": ranks[index],
                         "advanced": True, "final_score": float(scores[index]),
                         "final_rank": ranks[index]} for index in range(len(documents))]
    batch_count = (len(documents) + RERANK_BATCH_SIZE - 1) // RERANK_BATCH_SIZE
    finalists_per_batch = max(5, RERANK_BATCH_SIZE // batch_count)
    finalists: list[int] = []
    diagnostics: list[dict] = []
    for batch_number, start in enumerate(range(0, len(documents), RERANK_BATCH_SIZE), start=1):
        batch = documents[start:start + RERANK_BATCH_SIZE]
        local_scores = _rerank_batch(client, question, batch, endpoint)
        local_order = sorted(range(len(batch)), key=lambda index: (-float(local_scores[index]), index))
        local_ranks = {index: rank for rank, index in enumerate(local_order, start=1)}
        advanced = set(local_order[:finalists_per_batch])
        finalists.extend(start + index for index in local_order if index in advanced)
        diagnostics.extend({"candidate_index": start + index, "initial_batch": batch_number,
                            "initial_score": float(local_scores[index]), "initial_rank": local_ranks[index],
                            "advanced": index in advanced, "final_score": None, "final_rank": None}
                           for index in range(len(batch)))
    if len(finalists) > RERANK_BATCH_SIZE:
        raise ValueError("Rerank tournament produced too many finalists")
    final_scores = _rerank_batch(client, question, [documents[index] for index in finalists], endpoint)
    scores = np.full(len(documents), -np.inf, dtype="float32")
    for index, score in zip(finalists, final_scores):
        scores[index] = score
    final_order = sorted(range(len(finalists)), key=lambda index: (-float(final_scores[index]), index))
    final_ranks = {finalists[index]: rank for rank, index in enumerate(final_order, start=1)}
    final_values = {index: float(score) for index, score in zip(finalists, final_scores)}
    for item in diagnostics:
        index = item["candidate_index"]
        if index in final_values:
            item["final_score"] = final_values[index]
            item["final_rank"] = final_ranks[index]
    return scores, diagnostics
