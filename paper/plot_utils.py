"""Shared matplotlib helpers for ShapeShift paper figures."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

PolicyRows = Sequence[tuple[float, ...]]
Policies = Sequence[tuple[str, PolicyRows, str]]


def metric_values(rows: PolicyRows, idx: int) -> list[float]:
    return [row[idx] for row in rows]


def plot_grouped_bars_on_ax(
    ax: plt.Axes,
    workloads: Sequence[str],
    policies: Policies,
    metric_idx: int,
    ylabel: str,
    *,
    annotate_policy: str | None = "ShapeShift",
    annotate_vs: str = "FCFS",
    show_legend: bool = False,
    bar_width: float = 0.18,
    xtick_rotation: int = 0,
    panel_title: str | None = None,
) -> None:
    """Draw one grouped bar chart on the given axes."""
    n_workloads = len(workloads)
    n_policies = len(policies)
    x = np.arange(n_workloads)
    offsets = (np.arange(n_policies) - (n_policies - 1) / 2) * bar_width

    policy_data = {name: metric_values(rows, metric_idx) for name, rows, _ in policies}
    baseline = policy_data[annotate_vs]

    bars_for_annotate = None
    for offset, (name, _rows, color) in zip(offsets, policies):
        values = policy_data[name]
        bars = ax.bar(
            x + offset,
            values,
            bar_width,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        if annotate_policy and name == annotate_policy:
            bars_for_annotate = (bars, values)

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=xtick_rotation, ha="center")
    if panel_title:
        ax.set_title(panel_title, fontsize=11, pad=4)
    ymax = max(max(policy_data[name]) for name, _, _ in policies)
    ax.set_ylim(0, ymax * 1.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        if "Throughput" in ylabel:
            ax.legend(loc="upper left", framealpha=0.9, ncol=2)
        else:
            ax.legend(loc="upper right", framealpha=0.9, ncol=2)

    if bars_for_annotate is not None:
        bars, values = bars_for_annotate
        for i, bar in enumerate(bars):
            if metric_idx == 0:
                speedup = values[i] / baseline[i]
                text = f"{speedup:.2f}\u00d7"
                color = "#2c3e50"
            else:
                reduction = (1 - values[i] / baseline[i]) * 100
                text = f"{reduction:.0f}%"
                color = "#27ae60"
            ax.annotate(
                text,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )


def save_combined_row(
    panels: Sequence[tuple[int, str, str]],
    workloads: Sequence[str],
    policies: Policies,
    out_path: Path,
    *,
    figsize: tuple[float, float] = (11.5, 3.4),
    annotate_policy: str | None = "ShapeShift",
    bar_width: float = 0.18,
    xtick_rotation: int = 0,
) -> None:
    """Save a single-row multi-panel figure."""
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharey=False)
    if len(panels) == 1:
        axes = [axes]

    for ax, (metric_idx, ylabel, title), show_legend in zip(
        axes,
        panels,
        [True] + [False] * (len(panels) - 1),
    ):
        plot_grouped_bars_on_ax(
            ax,
            workloads,
            policies,
            metric_idx,
            ylabel,
            annotate_policy=annotate_policy,
            show_legend=show_legend,
            bar_width=bar_width,
            xtick_rotation=xtick_rotation,
            panel_title=title,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
