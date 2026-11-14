"""Aggregate main-matrix summaries into CSV and LaTeX rows."""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "cache-sparse-attention" / "results"

INPUT_PATTERNS = [
    (
        "ShareGPT multiturn",
        ["dlmserve_mainmatrix_sharegpt_n96_b8_s8_seed*_v1_summary.csv"],
    ),
    (
        "HotpotQA/Wikipedia",
        ["dlmserve_mainmatrix_hotpotqa_n96_b8_s8_seed*_v1_summary.csv"],
    ),
    (
        "Multitask mix",
        ["dlmserve_mainmatrix_multitask_n96_b8_s8_seed*_v1_summary.csv"],
    ),
    (
        "ShareGPT mixed",
        ["dlmserve_mainmatrix_sharegpt_mixed_n96_b8_s8_seed*_v1_summary.csv"],
    ),
    (
        "Alpaca instruction",
        [
            "dlmserve_mainmatrix_alpaca_n96_b8_s8_seed*_v1_summary.csv",
            "dlmserve_mainmatrix_alpaca_n96_b8_s8_seed*_v2_summary.csv",
        ],
    ),
    (
        "OpenOrca/FLAN instr.",
        [
            "dlmserve_mainmatrix_instruction_mix_downloaded_20260603*_summary.csv",
            "dlmserve_mainmatrix_instruction_mix_n96_b8_s8_seed*_v1_summary.csv",
        ],
    ),
]

DINFER_INPUTS = {
    "ShareGPT multiturn": {
        "exp": "dinfer_table2_sharegpt_multiturn_n96_b8_config1_long",
        "trace": RESULTS / "dinfer_table2_sharegpt_multiturn_n96_b8_config1_long" / "dinfer_table2_sharegpt_multiturn_n96_b8_config1_long_dinfer_sharegpt_multiturn_n96_seed13.json_1_threshold_0.95_0.jsonl",
    },
    "HotpotQA/Wikipedia": {
        "exp": "dinfer_table2_hotpotqa_wikipedia_n96_b8_config1_fixed",
        "trace": RESULTS / "dinfer_table2_hotpotqa_wikipedia_n96_b8_config1_fixed" / "dinfer_table2_hotpotqa_wikipedia_n96_b8_config1_fixed_dinfer_hotpotqa_wikipedia_n96_seed13.json_1_threshold_0.95_0.jsonl",
    },
    "Multitask mix": {
        "exp": "dinfer_table2_multitask_mix_n96_b8_config1_long",
        "trace": RESULTS / "dinfer_table2_multitask_mix_n96_b8_config1_long" / "dinfer_table2_multitask_mix_n96_b8_config1_long_dinfer_multitask_mix_n96_seed13.json_1_threshold_0.95_0.jsonl",
    },
    "ShareGPT mixed": {
        "exp": "dinfer_table2_sharegpt_mixed_n96_b8_config1_long",
        "trace": RESULTS / "dinfer_table2_sharegpt_mixed_n96_b8_config1_long" / "dinfer_table2_sharegpt_mixed_n96_b8_config1_long_dinfer_sharegpt_mixed_n96_seed13.json_1_threshold_0.95_0.jsonl",
    },
    "Alpaca instruction": {
        "exp": "dinfer_table2_alpaca_instruction_n96_b8_config1_long",
        "trace": RESULTS / "dinfer_table2_alpaca_instruction_n96_b8_config1_long" / "dinfer_table2_alpaca_instruction_n96_b8_config1_long_dinfer_alpaca_instruction_n96_seed13.json_1_threshold_0.95_0.jsonl",
    },
    "OpenOrca/FLAN instr.": {
        "exp": "dinfer_table2_instruction_mix_n96_b8_config1_fixedstats",
        "trace": RESULTS / "dinfer_table2_instruction_mix_n96_b8_config1_fixedstats" / "dinfer_table2_instruction_mix_n96_b8_config1_fixedstats_dinfer_instruction_mix_n96_seed13.json_1_threshold_0.95_0.jsonl",
    },
}

POLICY_NAMES = {
    "fifo": "dlmserve FIFO",
    "length_bucket": "Shape-only",
    "aging_bucket": "ShapeShift-Aging",
    "token_budget": "ShapeShift",
    "shapeshift": "ShapeShift",
    "length_sorted_admission": "Offline length-sorted",
}
POLICY_ALIASES = {
    "shape_only": "length_bucket",
    "shape_aging_old": "aging_bucket",
}


def normalize_policy_name(policy: str) -> str:
    return POLICY_ALIASES.get(str(policy).strip().lower(), str(policy).strip().lower())


def discover_summary_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        if "*" in pattern or "?" in pattern or "[" in pattern:
            paths.extend(sorted(RESULTS.glob(pattern)))
        else:
            p = RESULTS / pattern
            if p.exists():
                paths.append(p)
    return sorted(set(paths))


def infer_seed(path: Path, row: dict[str, str]) -> int | None:
    m = re.search(r"_seed(\d+)_", path.name)
    if m:
        return int(m.group(1))
    if "seed" in row:
        try:
            return int(float(row["seed"]))
        except (TypeError, ValueError):
            return None
    return None


def aggregate(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return mean, math.sqrt(variance)


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "reason" in rows[0]:
        return []
    return rows


def read_json_objects(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    text = path.read_text()
    decoder = json.JSONDecoder()
    pos = 0
    rows: list[dict[str, object]] = []
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        obj, next_pos = decoder.raw_decode(text, pos)
        rows.append(obj)
        pos = next_pos
    return rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def read_dinfer_tps(exp_name: str) -> float | None:
    results_path = ROOT / "results.txt"
    if not results_path.exists():
        return None
    match: list[str] = []
    for line in results_path.read_text().splitlines():
        if line.startswith(exp_name + " "):
            match = line.split()
    if not match:
        return None
    # dInfer benchmark_dataset.py prints:
    # exp config decoder threshold prefix total_forward wall avg_tokens fps tps ...
    return float(match[9])


def make_dinfer_row(workload: str, fifo_tok: float, fifo_p95_s: float) -> dict[str, object] | None:
    spec = DINFER_INPUTS.get(workload)
    if not spec:
        return None
    trace_rows = read_json_objects(spec["trace"])
    if not trace_rows:
        return None
    service_times = [
        float(row["generated_length"]) / max(1e-9, float(row["tps"]))
        for row in trace_rows
    ]
    tok_s = read_dinfer_tps(str(spec["exp"]))
    if tok_s is None:
        tok_s = sum(float(row["generated_length"]) for row in trace_rows) / sum(service_times)
    return {
        "workload": workload,
        "policy": "dInfer/LLaDA offline",
        "tok_s": tok_s,
        "p95_s": percentile(service_times, 0.95),
        "speedup": tok_s / max(1e-9, fifo_tok),
        "p95_ratio": percentile(service_times, 0.95) / max(1e-9, fifo_p95_s),
        "mean_prompt_len": "",
        "min_prompt_len": "",
        "max_prompt_len": "",
        "n_completed": len(trace_rows),
        "n_failed": 0,
        "tok_s_std": 0.0,
        "p95_s_std": 0.0,
        "speedup_std": 0.0,
        "p95_ratio_std": 0.0,
        "n_seeds": 1,
    }


def main() -> None:
    out_rows: list[dict[str, object]] = []
    for workload, pattern in INPUT_PATTERNS:
        file_rows_by_seed: dict[int, list[dict[str, str]]] = {}
        for path in discover_summary_paths(pattern):
            rows = read_summary(path)
            if not rows:
                continue
            seed = infer_seed(path, rows[0])
            if seed is None:
                continue
            file_rows_by_seed[seed] = rows

        if not file_rows_by_seed:
            continue

        grouped: dict[str, list[dict[str, float | str | int]]] = {k: [] for k in POLICY_NAMES}

        for seed, rows in sorted(file_rows_by_seed.items()):
            try:
                fifo = next(row for row in rows if row["policy"] == "fifo")
            except StopIteration:
                continue
            fifo_tok = float(fifo["tok_per_s"])
            fifo_p95_s = float(fifo["p95_latency_ms"]) / 1000.0

            for row in rows:
                policy = row["policy"]
                policy = normalize_policy_name(policy)
                if policy not in POLICY_NAMES:
                    continue
                tok_s = float(row["tok_per_s"])
                p95_s = float(row["p95_latency_ms"]) / 1000.0
                grouped[policy].append(
                    {
                        "seed": seed,
                        "tok_s": tok_s,
                        "p95_s": p95_s,
                        "speedup": tok_s / max(1e-9, fifo_tok),
                        "p95_ratio": p95_s / max(1e-9, fifo_p95_s),
                        "mean_prompt_len": float(row["mean_prompt_len"]),
                        "min_prompt_len": int(float(row["min_prompt_len"])),
                        "max_prompt_len": int(float(row["max_prompt_len"])),
                        "n_completed": int(float(row["n_completed"])),
                        "n_failed": int(float(row["n_failed"])),
                    }
                )

        for policy, rows in grouped.items():
            if not rows:
                continue
            tok_s_mean, tok_s_std = aggregate([float(row["tok_s"]) for row in rows])
            p95_mean, p95_std = aggregate([float(row["p95_s"]) for row in rows])
            speedup_mean, speedup_std = aggregate([float(row["speedup"]) for row in rows])
            p95_ratio_mean, p95_ratio_std = aggregate([float(row["p95_ratio"]) for row in rows])

            sample0 = rows[0]
            out_rows.append(
                {
                    "workload": workload,
                    "policy": POLICY_NAMES[policy],
                    "tok_s": tok_s_mean,
                    "p95_s": p95_mean,
                    "speedup": speedup_mean,
                    "p95_ratio": p95_ratio_mean,
                    "mean_prompt_len": sum(row["mean_prompt_len"] for row in rows) / len(rows),
                    "min_prompt_len": sample0["min_prompt_len"],
                    "max_prompt_len": sample0["max_prompt_len"],
                    "n_completed": sample0["n_completed"],
                    "n_failed": sample0["n_failed"],
                    "tok_s_std": tok_s_std,
                    "p95_s_std": p95_std,
                    "speedup_std": speedup_std,
                    "p95_ratio_std": p95_ratio_std,
                    "n_seeds": len(rows),
                }
            )

            if policy == "fifo":
                dinfer_row = make_dinfer_row(
                    workload,
                    tok_s_mean,
                    p95_mean,
                )
                if dinfer_row:
                    out_rows.append(dinfer_row)

    csv_out = RESULTS / "dlmserve_mainmatrix_aggregate_v1.csv"
    if out_rows:
        with csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)

    tex_out = RESULTS / "dlmserve_mainmatrix_aggregate_v1.tex"
    main_rows = [
        row for row in out_rows if str(row["policy"]) != "dInfer/LLaDA offline"
    ]

    def fmt_mean_std(row: dict[str, object], key: str, std_key: str, digits: int = 1) -> str:
        value = float(row[key])
        std = float(row.get(std_key, 0.0))
        if int(row.get("n_seeds", 1)) <= 1 or std == 0.0:
            return f"{value:.{digits}f}"
        return f"{value:.{digits}f}$\\pm${std:.{digits}f}"

    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Workload & Policy & Tok/s & P95(s) & Speed & P95/FIFO \\\\",
        "\\midrule",
    ]
    for row in main_rows:
        speed = float(row["speedup"])
        speed_std = float(row.get("speedup_std", 0.0))
        p95_ratio = float(row["p95_ratio"])
        p95_ratio_std = float(row.get("p95_ratio_std", 0.0))
        if int(row.get("n_seeds", 1)) > 1 and speed_std > 0.0:
            speed_text = f"{speed:.2f}$\\pm${speed_std:.2f}$\\times$"
        else:
            speed_text = f"{speed:.2f}$\\times$"
        if int(row.get("n_seeds", 1)) > 1 and p95_ratio_std > 0.0:
            p95_ratio_text = f"{p95_ratio:.2f}$\\pm${p95_ratio_std:.2f}"
        else:
            p95_ratio_text = f"{p95_ratio:.2f}"
        lines.append(
            f"{row['workload']} & {row['policy']} & "
            f"{fmt_mean_std(row, 'tok_s', 'tok_s_std')} & "
            f"{fmt_mean_std(row, 'p95_s', 'p95_s_std')} & "
            f"{speed_text} & {p95_ratio_text} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    tex_out.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_out}")
    print(f"wrote {tex_out}")

    summary_out = RESULTS / "dlmserve_mainmatrix_workload_stats_v1.csv"
    stat_rows: list[dict[str, object]] = []
    for workload in sorted({row["workload"] for row in out_rows}):
        fifo_rows = [
            row
            for row in out_rows
            if row["workload"] == workload and row["policy"] == POLICY_NAMES["fifo"]
        ]
        if not fifo_rows:
            continue
        row = fifo_rows[0]
        stat_rows.append(
            {
                "workload": workload,
                "mean_prompt_len": row["mean_prompt_len"],
                "min_prompt_len": row["min_prompt_len"],
                "max_prompt_len": row["max_prompt_len"],
                "n_completed": row["n_completed"],
                "n_failed": row["n_failed"],
            }
        )
    if stat_rows:
        with summary_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stat_rows)
        print(f"wrote {summary_out}")


if __name__ == "__main__":
    main()
