"""Count people in page photographs while preserving page image provenance."""

import base64
import json
import shutil
from pathlib import Path

from PIL import Image

from src.parser.vision import VISION_MODEL, _client


def _photos_on_page(client, image_path: Path, model: str) -> list[dict]:
    with Image.open(image_path) as image:
        mime = Image.MIME.get(image.format, "image/png")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": (
                "只统计此文档页面中每一张独立的真实照片里可见的人。忽略正文提到的人、绘图人物、图标和重复缩略图。"
                "请从上到下、从左到右逐张列出。人数看不清时用 null；多人拥挤无法精确计数时给保守估计并设 estimated=true。"
                "严格输出 JSON 对象：{\"photos\":[{\"people_count\":整数或null,\"estimated\":布尔值,"
                "\"description\":\"简短可见内容\"}]}。没有照片则 photos 为空数组。不要猜测图片外的人。"
            )},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]}], temperature=0, max_tokens=550,
    )
    raw = response.choices[0].message.content if response.choices else None
    if not raw:
        raise ValueError("Photo inventory model returned an empty response")
    value = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("photos"), list):
        raise ValueError("Photo inventory response must contain a photos list")
    photos = []
    for number, photo in enumerate(parsed["photos"], 1):
        count = photo.get("people_count")
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("Invalid photo people_count")
        if type(photo.get("estimated")) is not bool:
            raise ValueError("Invalid photo estimated flag")
        photos.append({"photo_number": number, "people_count": count,
                       "estimated": photo["estimated"],
                       "description": str(photo.get("description", ""))[:160]})
    return photos


def inventory_parsed(parsed_dir: Path, output_dir: Path, client=None,
                     model: str = VISION_MODEL, on_progress=None) -> dict:
    """Copy parsed artifacts and add page-scoped, unreviewed photo observations."""
    if not parsed_dir.is_dir() or not any(parsed_dir.glob("*/manifest.json")):
        raise ValueError("No parsed manifests found")
    if output_dir.exists():
        raise ValueError(f"Output already exists: {output_dir}")
    api = client or _client()
    shutil.copytree(parsed_dir, output_dir)
    manifests = sorted(output_dir.glob("*/manifest.json"))
    total = sum(sum(a.get("kind") == "page_render_for_vision"
                    for a in json.loads(p.read_text(encoding="utf-8")).get("image_assets", []))
                for p in manifests)
    completed = 0
    errors = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        observations = []
        for asset in manifest.get("image_assets", []):
            if asset.get("kind") != "page_render_for_vision":
                continue
            page = asset["page"]
            try:
                photos = _photos_on_page(api, manifest_path.parent / asset["path"], model)
                observation = {"page": page, "asset_id": asset["asset_id"],
                               "photos": photos, "model": model, "review_status": "unreviewed"}
                observations.append(observation)
                completed += 1
            except Exception as exc:
                error = {"stage": "photo_inventory", "doc_id": manifest["doc_id"], "page": page,
                         "asset_id": asset.get("asset_id"), "type": type(exc).__name__, "message": str(exc)}
                errors.append(error)
                with (manifest_path.parent / "errors.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(error, ensure_ascii=False) + "\n")
            if on_progress:
                on_progress(completed + len(errors), total)
        manifest["photo_inventory"] = observations
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"requested_pages": total, "completed_pages": completed, "failed_pages": len(errors),
              "errors": errors, "model": model, "review_status": "unreviewed",
              "raw_responses_persisted": False}
    (output_dir / "photo_inventory_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
