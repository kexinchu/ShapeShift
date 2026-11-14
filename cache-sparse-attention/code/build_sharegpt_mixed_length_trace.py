"""Build a ShareGPT-derived mixed short/long serving length trace.

The input rows already contain tokenizer-measured prompt lengths. This script
keeps the workload real-data-derived while making the serving condition explicit:
a multi-tenant queue with short single-turn chat requests mixed with longer
multi-turn conversations.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "cache-sparse-attention" / "results"
DEFAULT_IN = RESULTS / "real_length_trace_sharegpt5k_v1_rows.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_IN)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", default="sharegpt_mixed_short_long")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--n_short", type=int, default=48)
    p.add_argument("--n_long", type=int, default=48)
    p.add_argument("--short_max", type=int, default=512)
    p.add_argument("--long_min", type=int, default=1024)
    p.add_argument("--long_max", type=int, default=2048)
    args = p.parse_args()

    rows = read_rows(args.input)
    short_rows = [
        row
        for row in rows
        if row["source"] == "sharegpt_single_turn"
        and int(row["total_len"]) <= args.short_max
    ]
    long_rows = [
        row
        for row in rows
        if row["source"] == "sharegpt_multiturn_accum"
        and args.long_min <= int(row["total_len"]) <= args.long_max
    ]
    if len(short_rows) < args.n_short:
        raise ValueError(f"not enough short rows: {len(short_rows)} < {args.n_short}")
    if len(long_rows) < args.n_long:
        raise ValueError(f"not enough long rows: {len(long_rows)} < {args.n_long}")

    rng = random.Random(args.seed)
    rng.shuffle(short_rows)
    rng.shuffle(long_rows)
    selected = short_rows[: args.n_short] + long_rows[: args.n_long]
    rng.shuffle(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        fieldnames = ["source", "request_uid", "prompt_len", "gen_len", "total_len", "origin_source"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(selected):
            writer.writerow(
                {
                    "source": args.source,
                    "request_uid": f"mix_{idx:04d}_{row['request_uid']}",
                    "prompt_len": row["prompt_len"],
                    "gen_len": row["gen_len"],
                    "total_len": row["total_len"],
                    "origin_source": row["source"],
                }
            )

    prompt_lens = [int(row["prompt_len"]) for row in selected]
    print(f"wrote {args.output}")
    print(
        f"n={len(prompt_lens)} min={min(prompt_lens)} "
        f"p50={sorted(prompt_lens)[len(prompt_lens)//2]} max={max(prompt_lens)}"
    )


if __name__ == "__main__":
    main()
