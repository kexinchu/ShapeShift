"""Plot main online scheduler comparisons from dlmserve CSV outputs.

The script merges existing mainmatrix runs with completion runs that only
contain additional policies such as token_budget. It emits a compact metrics CSV
and one PDF per metric for paper figures.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CSA = ROOT / "cache-sparse-attention"
RESULTS = CSA / "results"
FIGURES = CSA / "papers" / "active-set-runtime-aaai" / "figures"

WORKLOADS = [
    (
        "ShareGPT",
        [
            "dlmserve_mainmatrix_sharegpt_n96_b8_s8_seed*_v1",
            "dlmserve_completion_tokenbudget_sharegpt_n96_b8_s8_seed*_v1",
        ],
    ),
    (
        "ShareGPT mixed",
        [
            "dlmserve_mainmatrix_sharegpt_mixed_n96_b8_s8_seed*_v1",
            "dlmserve_completion_tokenbudget_sharegpt_mixed_n96_b8_s8_seed*_v1",
        ],
    ),
    (
        "Multitask",
        [
            "dlmserve_mainmatrix_multitask_n96_b8_s8_seed*_v1",
            "dlmserve_completion_tokenbudget_multitask_n96_b8_s8_seed*_v1",
        ],
    ),
    (
        "OpenOrca/FLAN",
        [
            "dlmserve_mainmatrix_instruction_mix_n96_b8_s8_seed*_v1",
            "dlmserve_completion_tokenbudget_instruction_mix_n96_b8_s8_seed*_v1",
        ],
    ),
]

POLICIES = [
    ("fifo", "FCFS"),
    ("length_bucket", "ShapeShift-Shape"),
    ("aging_bucket", "ShapeShift-Aging"),
    ("token_budget", "ShapeShift-Budget"),
    ("shapeshift", "ShapeShift"),
    ("length_sorted_admission", "Offline length-sorted"),
]
POLICY_ALIASES = {
    "shape_only": "length_bucket",
    "shape_aging_old": "aging_bucket",
}


def normalize_policy_name(policy: str) -> str:
    return POLICY_ALIASES.get(str(policy).strip().lower(), str(policy).strip().lower())


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(var)


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "reason" in rows[0]:
        return []
    return rows


def read_rows(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "done":
                continue
            policy = normalize_policy_name(row.get("policy", ""))
            try:
                out[policy].append(float(row["response_latency_ms"]) / 1000.0)
            except (KeyError, TypeError, ValueError):
                continue
    return out


def discover(prefixes: list[str]) -> list[Path]:
    paths: list[Path] = []
    for prefix in prefixes:
        paths.extend(RESULTS.glob(prefix + "_summary.csv"))
    return sorted(set(paths))


def collect() -> list[dict[str, object]]:
    metrics: list[dict[str, object]] = []
    for workload, prefixes in WORKLOADS:
        values: dict[str, dict[str, list[float]]] = {
            policy: {"tok_s": [], "p95_s": [], "mean_s": []}
            for policy, _ in POLICIES
        }
        for summary_path in discover(prefixes):
            rows_by_policy = read_rows(
                summary_path.with_name(summary_path.name.replace("_summary.csv", "_rows.csv"))
            )
            for row in read_summary(summary_path):
                policy = normalize_policy_name(row.get("policy", ""))
                if policy not in values:
                    continue
                try:
                    values[policy]["tok_s"].append(float(row["tok_per_s"]))
                    values[policy]["p95_s"].append(float(row["p95_latency_ms"]) / 1000.0)
                except (KeyError, TypeError, ValueError):
                    continue
                if rows_by_policy.get(policy):
                    values[policy]["mean_s"].append(sum(rows_by_policy[policy]) / len(rows_by_policy[policy]))

        for policy, label in POLICIES:
            if not values[policy]["tok_s"]:
                continue
            tok_mean, tok_std = mean_std(values[policy]["tok_s"])
            p95_mean, p95_std = mean_std(values[policy]["p95_s"])
            mean_lat, mean_lat_std = mean_std(values[policy]["mean_s"])
            metrics.append(
                {
                    "workload": workload,
                    "policy": policy,
                    "policy_label": label,
                    "tok_s": tok_mean,
                    "tok_s_std": tok_std,
                    "mean_latency_s": mean_lat,
                    "mean_latency_s_std": mean_lat_std,
                    "p95_latency_s": p95_mean,
                    "p95_latency_s_std": p95_std,
                    "n_runs": len(values[policy]["tok_s"]),
                }
            )
    return metrics


def write_metrics(metrics: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def plot_metric(metrics: list[dict[str, object]], metric: str, err: str, ylabel: str, out_pdf: Path) -> None:
    workloads = [name for name, _ in WORKLOADS]
    policy_labels = [label for _, label in POLICIES]
    policy_keys = [key for key, _ in POLICIES]
    by_key = {(row["workload"], row["policy"]): row for row in metrics}

    width = 0.18
    x = list(range(len(workloads)))
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    colors = ["#808080", "#6E9BC8", "#2E7D5B", "#B08D2F"]
    for idx, (policy, label) in enumerate(zip(policy_keys, policy_labels)):
        vals = []
        errs = []
        for workload in workloads:
            row = by_key.get((workload, policy))
            vals.append(float(row[metric]) if row else float("nan"))
            errs.append(float(row[err]) if row else 0.0)
        xpos = [v + (idx - 1.5) * width for v in x]
        ax.bar(xpos, vals, width=width, label=label, color=colors[idx], yerr=errs, capsize=2, linewidth=0)

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=15, ha="right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout(pad=0.4)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_prefix", default="llada8b_policy")
    args = parser.parse_args()

    metrics = collect()
    if not metrics:
        raise RuntimeError("no metrics found")
    csv_out = RESULTS / f"{args.out_prefix}_metrics.csv"
    write_metrics(metrics, csv_out)
    plot_metric(
        metrics,
        "tok_s",
        "tok_s_std",
        "Throughput (tok/s)",
        FIGURES / f"{args.out_prefix}_throughput.pdf",
    )
    plot_metric(
        metrics,
        "mean_latency_s",
        "mean_latency_s_std",
        "Mean latency (s)",
        FIGURES / f"{args.out_prefix}_mean_latency.pdf",
    )
    plot_metric(
        metrics,
        "p95_latency_s",
        "p95_latency_s_std",
        "P95 latency (s)",
        FIGURES / f"{args.out_prefix}_p95_latency.pdf",
    )
    print(f"wrote {csv_out}")


if __name__ == "__main__":
    main()
