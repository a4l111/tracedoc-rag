"""Draw the four-value prompt pipeline diagram (说明性示意，非系统输出截图)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

TEAL = "#145b65"
INK = "#1c3040"
MUTED = "#607d84"
FILLS = {"input": "#eef4f5", "prompt": "#e5f2f3", "raw": "#f5f6f7",
         "citations": "#fff6e9", "final": "#edf7f6"}

BOXES = [
    ("输入", "用户问题 + 检索 Top 5\n（编号化 Context）", "input"),
    ("值一 · Prompt 指令", "只依据 Context 回答\n关键结论强制 [1][2] 引用\n不编造引用\n资料不足时明确说明", "prompt"),
    ("值二 · 原始模型响应", "DeepSeek 返回的原始文本\n（独立保存，不经改写）", "raw"),
    ("值三 · 解析出的引用编号", "从响应中解析 [1][2] 等编号\n逐条对应检索片段", "citations"),
    ("值四 · 最终展示答案", "带引用的最终答案\n（展示给用户）", "final"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 9.0), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")

    y = 12.2
    box_h = 1.9
    gap = 0.42
    x0, x1 = 1.1, 8.9
    for title, body, key in BOXES:
        box = FancyBboxPatch((x0, y - box_h), x1 - x0, box_h,
                             boxstyle="round,pad=0.08,rounding_size=0.18",
                             linewidth=1.4, edgecolor=TEAL, facecolor=FILLS[key])
        ax.add_patch(box)
        ax.text(x0 + 0.32, y - 0.30, title, fontsize=13, fontweight="bold",
                color=INK, ha="left", va="top")
        ax.text(x0 + 0.32, y - 0.72, body, fontsize=10.5, color=INK,
                ha="left", va="top", linespacing=1.5)
        if key != "final":
            ax.add_patch(FancyArrowPatch((5.0, y - box_h - 0.03), (5.0, y - box_h - gap + 0.03),
                                         arrowstyle="-|>", mutation_scale=18,
                                         linewidth=1.6, color=TEAL))
        y -= box_h + gap

    ax.text(5.0, 0.25, "四个值分离存储、互不混淆：Prompt 指令、原始响应、解析引用号与最终答案各为独立值。",
            fontsize=10.5, color=MUTED, ha="center", va="center")
    fig.savefig("docs/figures/prompt_pipeline.png", bbox_inches="tight",
                facecolor="white", pad_inches=0.25)
    plt.close(fig)
    print("prompt_pipeline.png written")


if __name__ == "__main__":
    main()
