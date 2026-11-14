"""Stitch per-checkpoint bar charts into one 3x4 main-results figure."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

MODELS = [
    ("LLaDA-8B", "llada8b"),
    ("Dream-7B", "dream7b"),
    ("CoDA-1.7B", "coda17b"),
    ("LLaDA2.0-mini", "llada2mini"),
]
METRICS = [
    ("throughput", "Throughput"),
    ("mean", "Mean latency"),
    ("p95", "P95 latency"),
]


def pdf_to_array(pdf_path: Path, dpi: int = 220) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-singlefile", "-r", str(dpi), str(pdf_path), str(prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return np.array(Image.open(f"{prefix}.png"))


def main() -> None:
    fig, axes = plt.subplots(3, 4, figsize=(11.8, 7.2))

    for row, (metric_slug, row_label) in enumerate(METRICS):
        for col, (model_name, model_slug) in enumerate(MODELS):
            ax = axes[row, col]
            pdf = FIGURES / f"bar_{model_slug}_{metric_slug}.pdf"
            ax.imshow(pdf_to_array(pdf))
            ax.axis("off")
            if row == 0:
                ax.set_title(model_name, fontsize=11, pad=3)
            if col == 0:
                ax.text(
                    -0.08,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                )

    plt.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.03, wspace=0.05, hspace=0.08)
    out = FIGURES / "fig_main_online_combined.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
