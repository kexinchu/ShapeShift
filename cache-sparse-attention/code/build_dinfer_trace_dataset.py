"""Build a dInfer-compatible dataset from a length trace CSV."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / "cache-sparse-attention" / "results" / "real_length_trace_sharegpt5k_v1_rows.csv"
RESULTS = ROOT / "cache-sparse-attention" / "results"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, default=TRACE)
    p.add_argument("--source", default="sharegpt_multiturn_accum")
    p.add_argument("--n_requests", type=int, default=24)
    p.add_argument("--max_total_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", type=Path, default=RESULTS / "dinfer_trace_multiturn_n24_seed13.json")
    args = p.parse_args()

    rows = []
    with args.trace.open(newline="") as f:
        for row in csv.DictReader(f):
            row_source = row.get("source", row.get("task", ""))
            if row_source != args.source:
                continue
            total_len = int(float(row.get("total_len") or 0))
            prompt_len = int(float(row["prompt_len"]))
            if total_len <= 0:
                total_len = prompt_len + 128
            if total_len <= args.max_total_len:
                rows.append(prompt_len)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n_requests]

    details = []
    for idx, prompt_len in enumerate(rows):
        prompt = " ".join(["the"] * max(1, prompt_len))
        details.append({"id": idx, "prompt": prompt, "target_prompt_len": prompt_len})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"details": details}, indent=2))
    print(f"wrote {args.out} ({len(details)} rows)")


if __name__ == "__main__":
    main()
