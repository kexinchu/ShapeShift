"""Bar charts for the LLaDA-8B main scheduler comparison table.

Each cell is throughput / mean latency / P95 latency (tok/s, s, s).
Style follows plot_fifo_shapeshift.ipynb.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

FIGSIZE = (6.4 * 0.9, 4.8 * 0.6)
HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

WORKLOADS = [
    "ShareGPT",
    "Multitask",
    "OpenOrca",
    "HotpotQA",
    "Alpaca",
]

# (throughput tok/s, mean latency s, P95 latency s)
FCFS = [
    (27.2, 296.3, 452.0),
    (30.7, 259.3, 401.6),
    (85.0, 97.8, 155.4),
    (84.6, 96.0, 145.4),
    (408.7, 20.0, 30.1),
]
# ShapeShift-Shape = length_bucket in dlmserve_mainmatrix (seed13/17/23 avg)
SHAPESHIFT_SHAPE = [
    (40.7, 185.9, 302.6),
    (45.1, 168.6, 273.4),
    (108.4, 72.5, 116.6),
    (100.5, 78.3, 122.3),
    (448.5, 17.6, 27.4),
]
SHAPESHIFT_AGING = [
    (41.3, 186.1, 298.0),
    (43.0, 179.8, 285.8),
    (108.1, 70.6, 119.6),
    (100.0, 78.9, 123.2),
    (447.1, 17.7, 27.5),
]
SHAPESHIFT = [
    (42.5, 183.0, 288.9),
    (45.1, 169.8, 274.2),
    (183.0, 47.0, 83.5),
    (109.9, 71.6, 110.9),
    (441.0, 18.1, 27.9),
]


POLICIES = [
    ("FCFS", FCFS, "#7f8c8d"),
    ("ShapeShift-Shape", SHAPESHIFT_SHAPE, "#2980b9"),
    ("ShapeShift-Aging", SHAPESHIFT_AGING, "#5dade2"),
    ("ShapeShift", SHAPESHIFT, "#27ae60"),
]


def _metric(rows: list[tuple[float, float, float]], idx: int) -> list[float]:
    return [row[idx] for row in rows]


def plot_grouped_bars(
    metric_idx: int,
    ylabel: str,
    save_path: Path,
    *,
    annotate_policy: str | None = None,
    annotate_vs: str = "FCFS",
) -> None:
    """Grouped bar chart for one metric across policies and workloads."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    n_workloads = len(WORKLOADS)
    n_policies = len(POLICIES)
    x = np.arange(n_workloads)
    width = 0.18
    offsets = (np.arange(n_policies) - (n_policies - 1) / 2) * width

    policy_data = {name: _metric(rows, metric_idx) for name, rows, _ in POLICIES}
    baseline = policy_data[annotate_vs]

    bars_for_annotate = None
    for offset, (name, rows, color) in zip(offsets, POLICIES):
        values = policy_data[name]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        if annotate_policy and name == annotate_policy:
            bars_for_annotate = (bars, values)

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOADS, rotation=0, ha="center")
    if "Throughput" in ylabel:
        ax.legend(loc="upper left", framealpha=0.9, ncol=2, bbox_to_anchor=(0, 1.1))
    else:
        ax.legend(loc="upper right", framealpha=0.9, ncol=2, bbox_to_anchor=(1, 1.1))
    ymax = max(max(policy_data[name]) for name, _, _ in POLICIES)
    ax.set_ylim(0, ymax * 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if bars_for_annotate is not None:
        bars, values = bars_for_annotate
        for i, bar in enumerate(bars):
            if metric_idx == 0:
                speedup = values[i] / baseline[i]
                text = f"{speedup:.2f}\u00d7"
                color = "#2c3e50"
            else:
                reduction = (1 - values[i] / baseline[i]) * 100
                text = f"   {reduction:.0f}%"
                color = "#27ae60"
            ax.annotate(
                text,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                color=color,
            )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def main() -> None:
    plot_grouped_bars(
        0,
        "Throughput (Tok/s)",
        FIGURES / "bar_llada8b_ablation_throughput.pdf",
        annotate_policy="ShapeShift",
    )
    plot_grouped_bars(
        1,
        "Mean Latency (s)",
        FIGURES / "bar_llada8b_ablation_mean.pdf",
        annotate_policy="ShapeShift",
    )
    plot_grouped_bars(
        2,
        "P95 Latency (s)",
        FIGURES / "bar_llada8b_ablation_p95.pdf",
        annotate_policy="ShapeShift",
    )
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
