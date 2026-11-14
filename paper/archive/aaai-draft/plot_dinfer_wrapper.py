"""Bar charts for the dInfer wrapper comparison table.

Each cell is throughput / mean latency / P95 latency (tok/s, s, s).
Style follows plot_ablation_llada8b.py.
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

FIGSIZE = (6.4 * 0.7, 4.8 * 0.5)
HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

WORKLOADS = [
    "ShareGPT",
    "Multitask",
    "OpenOrca",
    "HotpotQA",
]

# (throughput tok/s, mean latency s, P95 latency s)
FCFS = [
    (5.12, 682.6, 1200.8),
    (4.45, 959.4, 1381.0),
    (13.77, 246.6, 446.2),
    (67.20, 48.4, 91.4),
]
SHAPESHIFT = [
    (43.05, 106.4, 142.7),
    (13.33, 198.9, 460.9),
    (36.71, 82.4, 167.4),
    (91.46, 41.1, 67.2),
]

POLICIES = [
    ("FCFS", FCFS, "#7f8c8d"),
    ("ShapeShift", SHAPESHIFT, "#27ae60"),
]


def _metric(rows: list[tuple[float, float, float]], idx: int) -> list[float]:
    return [row[idx] for row in rows]


def plot_grouped_bars(
    metric_idx: int,
    ylabel: str,
    save_path: Path,
    *,
    annotate_policy: str | None = "ShapeShift",
    annotate_vs: str = "FCFS",
) -> None:
    """Grouped bar chart for one metric across policies and workloads."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    n_workloads = len(WORKLOADS)
    n_policies = len(POLICIES)
    x = np.arange(n_workloads)
    width = 0.32
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
    ax.set_xticklabels(WORKLOADS, rotation=10, ha="center")
    if "Throughput" in ylabel:
        ax.legend(loc="upper left", framealpha=0.9, ncol=1)
    else:
        ax.legend(loc="upper right", framealpha=0.9, ncol=1)
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
        FIGURES / "bar_dinfer_wrapper_throughput.pdf",
    )
    plot_grouped_bars(
        1,
        "Mean Latency (s)",
        FIGURES / "bar_dinfer_wrapper_mean.pdf",
    )
    plot_grouped_bars(
        2,
        "P95 Latency (s)",
        FIGURES / "bar_dinfer_wrapper_p95.pdf",
    )
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
