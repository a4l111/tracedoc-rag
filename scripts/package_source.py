"""Package tracked source plus the current reviewable, untracked additions."""

import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release/TraceDocRAG-source-20261031.zip"
ADDITIONS = (
    "docs/TraceDoc_RAG_项目介绍_29题八节版_第12题截图前置.docx",
    "docs/figures/retrieval_18_questions.png",
    "docs/figures/web_chat_demo_20261031.png",
    "docs/figures/cited_answer_q18.png",
    "scripts/make_evaluation_figure.py",
    "scripts/package_source.py",
    "src/parser/photo_inventory.py",
    "tests/test_build_alignment.py",
    "tests/test_photo_inventory.py",
    "tests/test_xlsx.py",
)


def main() -> None:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    paths = sorted(set(filter(None, tracked)) | set(ADDITIONS))
    for relative in paths:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(relative)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in paths:
            archive.write(ROOT / relative, relative)
        archive.writestr("SOURCE_STATE.txt", f"base_commit={commit}\nstate=working-tree snapshot with uncommitted changes\n")
    print(f"{OUTPUT} ({len(paths)} files)")


if __name__ == "__main__":
    main()
