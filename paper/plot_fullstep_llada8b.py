"""Longer-horizon validation (combined figure)."""
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
    (20.70, 296.8),
    (7.87, 781.2),
    (10.53, 583.3),
    (7.89, 778.8),
    (102.55, 59.9),
]
SHAPESHIFT_SHAPE = [
    (27.14, 226.4),
    (11.90, 516.2),
    (19.81, 310.1),
    (12.21, 503.1),
    (116.64, 52.7),
]
SHAPESHIFT_AGING = [
    (27.13, 226.5),
    (11.51, 533.7),
    (19.73, 311.4),
    (11.88, 517.1),
    (115.83, 53.0),
]
SHAPESHIFT = [
    (29.31, 209.6),
    (15.14, 405.8),
    (30.09, 204.2),
    (14.29, 430.1),
    (116.61, 52.7),
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
        (1, "P95 Latency (s)", "(b) P95 latency"),
    ]
    save_combined_row(
        panels,
        WORKLOADS,
        POLICIES,
        FIGURES / "fig_fullstep_combined.pdf",
        figsize=(8.8, 3.5),
        xtick_rotation=12,
    )
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
