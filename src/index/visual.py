"""Independent image vectors for text-to-image retrieval.

Image vectors live beside the text indexes in a snapshot but are optional: a
snapshot stays complete for text retrieval even when the visual embedding model
is unavailable. Citation locations (page/slide and the image asset path) stay
intact; the vectors only add a visual route, they never alter text chunks.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from src.utils.model_settings import get_api_key, load_settings

VISUAL_MODEL = "Qwen3-VL-Embedding-8B"
DIMENSION = 2048
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def visual_client() -> OpenAI:
    load_dotenv()
    settings = load_settings()["visual_embedding"]
    return OpenAI(api_key=get_api_key("visual_embedding"), base_url=settings["base_url"],
                  timeout=settings["timeout"], max_retries=2)


def _post_embeddings(client: OpenAI, inputs: list[dict]) -> list[np.ndarray]:
    settings = load_settings()["visual_embedding"]
    payload: dict[str, Any] = {"model": settings["model"], "input": inputs, "encoding_format": "float"}
    if settings.get("dimensions"):
        payload["dimensions"] = settings["dimensions"]
    response = client.post("/embeddings", cast_to=dict, body=payload)
    rows = sorted(response["data"], key=lambda row: row["index"])
    if [row["index"] for row in rows] != list(range(len(inputs))):
        raise ValueError("图文向量返回索引或数量不一致")
    vectors = []
    dimension = settings.get("dimensions", DIMENSION)
    for row in rows:
        vector = np.asarray(row["embedding"], dtype="float32")
        norm = np.linalg.norm(vector)
        if vector.shape != (dimension,) or not np.isfinite(vector).all() or not np.isfinite(norm) or norm == 0:
            raise ValueError(f"Invalid visual embedding: shape={vector.shape}")
        vectors.append(vector / norm)
    return vectors


def embed_image(client: OpenAI, path: Path) -> np.ndarray:
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("图片超过20MB，未发送至图文向量服务")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return _post_embeddings(client, [{"image": encoded}])[0]


def embed_text(client: OpenAI, text: str) -> np.ndarray:
    return _post_embeddings(client, [{"text": text}])[0]


def collect_images(parsed_dir: Path) -> list[dict]:
    """Return one entry per unique image asset across all parsed documents."""
    images: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for manifest_path in sorted(parsed_dir.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        doc_id = manifest.get("doc_id", manifest_path.parent.name)
        for asset in manifest.get("image_assets", []):
            path = asset.get("path", "")
            if not path or (doc_id, path) in seen:
                continue
            seen.add((doc_id, path))
            images.append({"doc_id": doc_id, "path": path,
                           "source_file": manifest.get("source_file", ""),
                           "page": asset.get("page"), "slide": asset.get("slide"),
                           "kind": asset.get("kind"), "asset_id": asset.get("asset_id")})
    return images


def build_visual_index(parsed_dir: Path, snapshot_folder: Path,
                       on_progress: Callable[[str, int, int, int], None] | None = None) -> dict:
    """Embed every image asset and write a FAISS index plus metadata into the snapshot."""
    images = collect_images(parsed_dir)
    if not images:
        return {"model": VISUAL_MODEL, "dimension": DIMENSION, "images": 0, "indexed": 0, "skipped": 0}
    client = visual_client()
    index = faiss.IndexFlatIP(DIMENSION)
    indexed: list[dict] = []
    failures = 0
    for number, image in enumerate(images, start=1):
        full = parsed_dir / image["doc_id"] / image["path"]
        try:
            vector = embed_image(client, full)
            index.add(vector.reshape(1, DIMENSION))
            image["image_sha256"] = hashlib.sha256(full.read_bytes()).hexdigest()
            indexed.append(image)
        except Exception as exc:
            failures += 1
            (snapshot_folder / "visual_errors.jsonl").open("a", encoding="utf-8").write(
                json.dumps({"doc_id": image["doc_id"], "path": image["path"],
                            "type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False) + "\n")
        if on_progress:
            on_progress("visual", number, len(images), failures)
    if not indexed:
        return {"model": VISUAL_MODEL, "dimension": DIMENSION, "images": len(images), "indexed": 0,
                "skipped": len(images), "failed": True}
    faiss.write_index(index, str(snapshot_folder / "visual_index.faiss"))
    images_bytes = ("\n".join(json.dumps(item, ensure_ascii=False) for item in indexed)).encode("utf-8")
    (snapshot_folder / "visual_images.jsonl").write_bytes(images_bytes)
    meta = {"snapshot_id": snapshot_folder.name, "model": VISUAL_MODEL, "dimension": DIMENSION,
            "base_url": load_settings()["visual_embedding"]["base_url"],
            "images_sha256": hashlib.sha256(images_bytes).hexdigest(),
            "indexed_count": len(indexed), "failed_count": failures}
    (snapshot_folder / "visual_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                                       encoding="utf-8")
    return {"model": VISUAL_MODEL, "dimension": DIMENSION, "images": len(images),
            "indexed": len(indexed), "skipped": failures, "failed": False}


def visual_search(indexes_dir: Path, snapshot_id: str, question: str, top_k: int = 5) -> list[dict]:
    """Embed a text query and return the top matching image assets with scores."""
    folder = indexes_dir / snapshot_id
    meta_path = folder / "visual_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError("该快照没有图文向量索引")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("snapshot_id") != snapshot_id or meta.get("model") != VISUAL_MODEL:
        raise ValueError("图文向量索引与快照或模型不匹配")
    if meta.get("base_url") != load_settings()["visual_embedding"]["base_url"]:
        raise ValueError("图文向量索引使用的服务与当前配置不一致")
    images_bytes = (folder / "visual_images.jsonl").read_bytes()
    if hashlib.sha256(images_bytes).hexdigest() != meta["images_sha256"]:
        raise ValueError("图文向量索引校验和不匹配")
    images = [json.loads(line) for line in images_bytes.splitlines()]
    index = faiss.read_index(str(folder / "visual_index.faiss"))
    if index.ntotal != len(images) or index.d != DIMENSION:
        raise ValueError("图文向量索引元数据不一致")
    vector = embed_text(visual_client(), question)
    scores, positions = index.search(vector.reshape(1, DIMENSION), min(top_k, index.ntotal))
    return [{**images[int(position)], "score": float(score)}
            for score, position in zip(scores[0], positions[0]) if position >= 0]
