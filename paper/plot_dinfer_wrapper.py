"""dInfer wrapper comparison (combined figure)."""
from __future__ import annotations

from plot_utils import FIGURES, save_combined_row

WORKLOADS = [
    "ShareGPT",
    "Multitask",
    "OpenOrca",
    "HotpotQA",
]

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


def main() -> None:
    panels = [
        (0, "Throughput (Tok/s)", "(a) Throughput"),
        (1, "Mean Latency (s)", "(b) Mean latency"),
        (2, "P95 Latency (s)", "(c) P95 latency"),
    ]
    save_combined_row(
        panels,
        WORKLOADS,
        POLICIES,
        FIGURES / "fig_dinfer_combined.pdf",
        figsize=(10.8, 3.2),
        bar_width=0.32,
        xtick_rotation=12,
    )
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
