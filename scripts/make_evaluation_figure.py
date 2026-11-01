"""Render the saved 18-question evaluation metrics for the project report."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORT = (ROOT / "data/benchmarks/mmlongbench_page_corrected/evaluation_photo_reviewed_full"
          / "20261030T090632Z-0940a301/report.json")
OUTPUT = ROOT / "docs/figures/retrieval_18_questions.png"


def main() -> None:
    result = json.loads(REPORT.read_text(encoding="utf-8"))["snapshots"][0]
    metrics = result["metrics"]
    rows = [("BM25", metrics["bm25"]["recall_at_5"]),
            ("Qwen3 Dense", metrics["dense"]["recall_at_5"]),
            ("bge-m3", metrics["bge_m3"]["recall_at_5"]),
            ("RRF + Reranker", metrics["hybrid_reranked"]["recall_at_5"])]
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = lambda size: ImageFont.truetype(str(font_path), size)
    canvas = Image.new("RGB", (1200, 610), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((65, 32), "18 题检索实验 · Recall@5", font=font(37), fill="#17324d")
    draw.text((65, 88), f"快照 {result['snapshot_id']}   |   {result['chunk_count']} Chunk   |   原始 Gold",
              font=font(21), fill="#526579")
    for index, (label, value) in enumerate(rows):
        top = 155 + index * 95
        draw.text((65, top + 12), label, font=font(26), fill="#17324d")
        draw.rounded_rectangle((325, top, 1045, top + 52), radius=10, fill="#e7eef5")
        draw.rounded_rectangle((325, top, 325 + round(720 * value), top + 52), radius=10,
                               fill="#177f9d" if index == 3 else "#4076a9")
        draw.text((1060, top + 10), f"{value:.4f}", font=font(26), fill="#17324d")
    draw.line((65, 550, 1135, 550), fill="#c9d5e1", width=2)
    draw.text((65, 562), "指标仅衡量证据位置命中；跨页汇总改变 Chunk 粒度，不能据此断言模型本身提升。",
              font=font(18), fill="#526579")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
