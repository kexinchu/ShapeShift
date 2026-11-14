"""Bar charts for the LLaDA-8B scheduler comparison (ablation + combined figure)."""
from __future__ import annotations

from plot_utils import FIGURES, save_combined_row

WORKLOADS = [
    "ShareGPT",
    "Multitask",
    "OpenOrca",
    "HotpotQA",
    "Alpaca",
]

FCFS = [
    (27.2, 296.3, 452.0),
    (30.7, 259.3, 401.6),
    (85.0, 97.8, 155.4),
    (84.6, 96.0, 145.4),
    (408.7, 20.0, 30.1),
]
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
        FIGURES / "fig_ablation_combined.pdf",
        figsize=(11.8, 3.5),
    )
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
