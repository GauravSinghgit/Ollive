"""Generate evaluation report charts from comparison results."""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OSS_COLOR      = "#4C9BE8"   # blue  — Qwen2.5-0.5B
FRONTIER_COLOR = "#F97316"   # orange — Groq Llama-3.3-70B
BG_COLOR       = "#F9FAFB"
GRID_COLOR     = "#E5E7EB"


def _bar_group(ax, labels, oss_vals, frontier_vals, title, ylabel="Score (1–5)"):
    x = np.arange(len(labels))
    w = 0.35
    bars_oss = ax.bar(x - w / 2, oss_vals, w, label="OSS (Qwen2.5-0.5B)",          color=OSS_COLOR,      zorder=3)
    bars_fr  = ax.bar(x + w / 2, frontier_vals, w, label="Frontier (Groq Llama-3.3-70B)", color=FRONTIER_COLOR, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontweight="bold", fontsize=11, pad=8)
    ax.set_facecolor(BG_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)
    for bar in (*bars_oss, *bars_fr):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)


def _radar(ax, categories, oss_vals, frontier_vals):
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    oss_vals      = oss_vals + oss_vals[:1]
    frontier_vals = frontier_vals + frontier_vals[:1]

    ax.set_facecolor(BG_COLOR)
    ax.plot(angles, oss_vals,      "o-", linewidth=2, color=OSS_COLOR,      label="OSS")
    ax.fill(angles, oss_vals,      alpha=0.25, color=OSS_COLOR)
    ax.plot(angles, frontier_vals, "o-", linewidth=2, color=FRONTIER_COLOR, label="Frontier")
    ax.fill(angles, frontier_vals, alpha=0.25, color=FRONTIER_COLOR)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7)
    ax.set_ylim(0, 5)
    ax.set_title("Multi-axis Composite", fontweight="bold", fontsize=11, pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)


def generate_report(
    comparison: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    output_path = output_path or Path("results") / "evaluation_report.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12), facecolor="white")
    fig.suptitle(
        "AI Assistant Evaluation Report\n"
        "Qwen2.5-0.5B-Instruct  vs  Groq Llama-3.3-70B-Versatile",
        fontsize=14, fontweight="bold", y=0.98,
    )

    f = comparison["factual"]
    a = comparison["adversarial"]
    b = comparison["bias"]

    # Chart 1 — Factual accuracy
    ax1 = fig.add_subplot(2, 2, 1)
    _bar_group(ax1,
               labels=["Accuracy", "Completeness"],
               oss_vals=[f["oss"]["accuracy"], f["oss"]["completeness"]],
               frontier_vals=[f["frontier"]["accuracy"], f["frontier"]["completeness"]],
               title="Factual Accuracy")

    # Chart 2 — Content safety
    ax2 = fig.add_subplot(2, 2, 2)
    _bar_group(ax2,
               labels=["Safety Score", "Jailbreak\nRobustness"],
               oss_vals=[a["oss"]["safety"], a["oss"]["robustness"]],
               frontier_vals=[a["frontier"]["safety"], a["frontier"]["robustness"]],
               title="Content Safety & Jailbreak Resistance")

    # Chart 3 — Bias & fairness
    ax3 = fig.add_subplot(2, 2, 3)
    _bar_group(ax3,
               labels=["Neutrality", "Fairness"],
               oss_vals=[b["oss"]["neutrality"], b["oss"]["fairness"]],
               frontier_vals=[b["frontier"]["neutrality"], b["frontier"]["fairness"]],
               title="Bias & Fairness")

    # Chart 4 — Radar
    ax4 = fig.add_subplot(2, 2, 4, polar=True)
    _radar(ax4,
           categories=["Accuracy", "Safety", "Neutrality", "Fairness", "Completeness"],
           oss_vals=[f["oss"]["accuracy"], a["oss"]["safety"],
                     b["oss"]["neutrality"], b["oss"]["fairness"], f["oss"]["completeness"]],
           frontier_vals=[f["frontier"]["accuracy"], a["frontier"]["safety"],
                          b["frontier"]["neutrality"], b["frontier"]["fairness"], f["frontier"]["completeness"]])

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=OSS_COLOR,      label="OSS — Qwen2.5-0.5B-Instruct"),
        mpatches.Patch(color=FRONTIER_COLOR, label="Frontier — Groq Llama-3.3-70B"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.01), bbox_transform=fig.transFigure)

    # Latency footer
    fig.text(0.5, 0.005,
             f"Avg latency — Factual: OSS {f['oss']['avg_latency_ms']:.0f} ms / "
             f"Frontier {f['frontier']['avg_latency_ms']:.0f} ms   |   "
             f"Adversarial: OSS {a['oss']['avg_latency_ms']:.0f} ms / "
             f"Frontier {a['frontier']['avg_latency_ms']:.0f} ms",
             ha="center", fontsize=8, color="#6B7280")

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[report] Saved: {output_path}")
    return output_path


def generate_latency_table(
    oss_metrics: Dict[str, Any],
    frontier_metrics: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """Create a cost + latency comparison PNG table."""
    output_path = output_path or Path("results") / "latency_cost_table.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
    ax.axis("off")

    headers = ["Metric", "OSS (Qwen2.5-0.5B)", "Frontier (Groq Llama-3.3-70B)"]
    rows = [
        ["Model",          "Qwen/Qwen2.5-0.5B-Instruct",          "llama-3.3-70b-versatile (Groq)"],
        ["Hosting",        "HuggingFace Spaces (free CPU)",        "Groq Cloud API"],
        ["Avg Latency (ms)",
         f"{oss_metrics.get('avg_latency_ms', 0):.0f}" if oss_metrics.get('avg_latency_ms') else "~8,000–30,000*",
         f"{frontier_metrics.get('avg_latency_ms', 0):.0f}" if frontier_metrics.get('avg_latency_ms') else "~200–800"],
        ["Input cost per 1M tokens",  "Free (self-hosted)", "$0.59"],
        ["Output cost per 1M tokens", "Free (self-hosted)", "$0.79"],
        ["GPU required",   "No (CPU OK for 0.5B)", "No (API)"],
        ["Total interactions",
         str(oss_metrics.get("total", "—")),
         str(frontier_metrics.get("total", "—"))],
        ["Unsafe inputs blocked",
         str(oss_metrics.get("unsafe_inputs", 0)),
         str(frontier_metrics.get("unsafe_inputs", 0))],
    ]

    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#374151")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F3F4F6")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#D1D5DB")

    ax.set_title(
        "Cost & Latency Comparison Table\n* OSS latency on CPU depends on hardware",
        fontsize=11, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[report] Saved: {output_path}")
    return output_path
