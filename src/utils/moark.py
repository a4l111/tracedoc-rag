"""Qwen3 text embeddings and reranking served by MoArk APIs."""

import os
from typing import Any, Dict

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://api.moark.com/v1"
MODEL = "Qwen3-Embedding-8B"
DIMENSION = 4096
RERANK_MODEL = "Qwen3-Reranker-8B"
RERANK_BATCH_SIZE = 20


def embedding_base_url() -> str:
    return BASE_URL


def client() -> OpenAI:
    load_dotenv()
    key = os.getenv("MOARK_API_KEY", "").strip()
    if not key or key == "your_key_here":
        raise ValueError("MOARK_API_KEY is missing; set it in .env or the environment")
    return OpenAI(api_key=key, base_url=embedding_base_url(), timeout=30.0, max_retries=2)


def rerank_client() -> tuple[OpenAI, str]:
    """MoArk serves reranking at /v1/rerank with the same access token."""
    return client(), "/rerank"


def embed(client: OpenAI, text: str) -> np.ndarray:
    response = client.embeddings.create(model=MODEL, input=text, dimensions=DIMENSION)
    vector = np.asarray(response.data[0].embedding, dtype="float32")
    norm = np.linalg.norm(vector)
    if vector.shape != (DIMENSION,) or not np.isfinite(vector).all() or not np.isfinite(norm) or norm == 0:
        raise ValueError(f"Invalid MoArk embedding: shape={vector.shape}")
    return vector / norm


def rerank(client: OpenAI, question: str, documents: list[str], endpoint: str) -> np.ndarray:
    if not documents:
        return np.array([], dtype="float32")
    scores = np.empty(len(documents), dtype="float32")
    for start in range(0, len(documents), RERANK_BATCH_SIZE):
        batch = documents[start:start + RERANK_BATCH_SIZE]
        response = client.post(endpoint, cast_to=Dict[str, Any], body={
            "model": RERANK_MODEL, "query": question, "documents": batch,
        })
        results = response.get("results", [])
        seen = set()
        for result in results:
            index = result.get("index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= len(batch) or index in seen:
                raise ValueError("Invalid MoArk rerank result index")
            score = float(result["relevance_score"])
            if not np.isfinite(score):
                raise ValueError("Invalid MoArk rerank score")
            scores[start + index] = score
            seen.add(index)
        if len(seen) != len(batch):
            raise ValueError("MoArk rerank omitted candidates")
    return scores
