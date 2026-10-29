"""Prepare a small, source-linked heading corpus from an existing parsed PPTX."""

import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/validation/pptx/parsed_fixed/265178_History_Of_Internet-f1d5a7738c0a"
TARGET = ROOT / "data/validation/chunk_modes/parsed/network-evolution-slides-2-4"
TITLE_SUFFIXES = ("The Beginning of Networking", "TCP/IP Creates the Internet",
                  "From the Internet to the World Wide Web")


def main() -> None:
    original = (SOURCE / "document.md").read_text(encoding="utf-8")
    markdown = original.split("<!-- SLIDE:2 -->", 1)[1].split("<!-- SLIDE:5 -->", 1)[0]
    markdown = "<!-- SLIDE:2 -->" + markdown
    for suffix in TITLE_SUFFIXES:
        pattern = re.compile(r"(?m)^(?!#)([^\n]*" + re.escape(suffix) + r")$")
        markdown, count = pattern.subn(r"## \1", markdown)
        if count != 1:
            raise ValueError(f"Expected one heading ending in: {suffix}")
    if any(token in markdown for token in ("265178", "Jessie Zhang", "Stu ID")):
        raise ValueError("Personal cover information leaked into the sample")

    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    manifest["doc_id"] = TARGET.name
    manifest["source_file"] = "network_evolution_slides_2_4.pptx (parsed excerpt)"
    manifest["slide_map"] = [item for item in manifest.get("slide_map", []) if item.get("slide") in (2, 3, 4)]
    manifest["image_assets"] = [item for item in manifest["image_assets"] if item.get("slide") in (2, 3, 4)]
    for asset in manifest["image_assets"]:
        asset["source_doc_id"] = manifest["doc_id"]
    manifest["errors"] = []
    manifest["experiment"] = {
        "source": "existing parsed PPTX, slides 2-4 only",
        "change": "promoted three actual slide titles to Markdown level-2 headings",
        "source_markdown_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
    }

    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "images").mkdir(exist_ok=True)
    for asset in manifest["image_assets"]:
        source_asset = SOURCE / asset["path"]
        if not source_asset.is_file():
            raise FileNotFoundError(source_asset)
        shutil.copy2(source_asset, TARGET / asset["path"])
    (TARGET / "document.md").write_text(markdown, encoding="utf-8")
    (TARGET / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (TARGET / "errors.jsonl").write_text("", encoding="utf-8")

    labels = [
        ("Which technology enabled efficient data transmission in ARPANET?", 2),
        ("In what year did ARPANET officially adopt TCP/IP?", 3),
        ("Who proposed the World Wide Web at CERN in 1989?", 4),
    ]
    gold = [{"question": question, "relevant": [{"doc_id": manifest["doc_id"], "slide": slide}]}
            for question, slide in labels]
    gold_path = TARGET.parents[1] / "gold.jsonl"
    gold_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in gold), encoding="utf-8")
    print(f"Prepared {TARGET}; {len(manifest['image_assets'])} linked images; {len(gold)} questions")


if __name__ == "__main__":
    main()
