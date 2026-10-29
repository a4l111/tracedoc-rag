"""Make an isolated parsed excerpt for the page-10 table retrieval experiment."""

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/benchmarks/mmlongbench/parsed_vision/05-03-18-political-release-f05a693718c1"
TARGET_ROOT = ROOT / "data/validation/q12_table"
TARGET = TARGET_ROOT / "parsed" / SOURCE.name


def main() -> None:
    markdown = (SOURCE / "document.md").read_text(encoding="utf-8")
    start = markdown.index("<!-- PAGE:6 -->")
    end = markdown.index("<!-- PAGE:12 -->")
    excerpt = markdown[start:end]
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    manifest["page_map"] = [item for item in manifest["page_map"] if 6 <= item["page"] <= 11]
    manifest["image_assets"] = [item for item in manifest["image_assets"]
                                if 6 <= item.get("page", 0) <= 11]
    manifest["errors"] = []
    manifest["experiment"] = "Existing parsed PDF excerpt, physical pages 6-11; no raw PDF reparsing"
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "images").mkdir(exist_ok=True)
    for asset in manifest["image_assets"]:
        source_asset = SOURCE / asset["path"]
        if not source_asset.is_file():
            raise FileNotFoundError(source_asset)
        shutil.copy2(source_asset, TARGET / asset["path"])
    (TARGET / "document.md").write_text(excerpt, encoding="utf-8")
    (TARGET / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (TARGET / "errors.jsonl").write_text("", encoding="utf-8")

    reviewed = ROOT / "data/benchmarks/mmlongbench_page_corrected/reviewed_q12/gold_reviewed.jsonl"
    question = next(json.loads(line) for line in reviewed.read_text(encoding="utf-8").splitlines()
                    if json.loads(line).get("benchmark_row") == 215)
    (TARGET_ROOT / "gold.jsonl").write_text(json.dumps(question, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Prepared {TARGET}; pages={len(manifest['page_map'])}; assets={len(manifest['image_assets'])}")


if __name__ == "__main__":
    main()
