"""Build a dlmserve length trace from an existing per-request rows CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--gen-length", type=int, default=128)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.rows.open(newline="") as src, args.out.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=["source", "prompt_len", "total_len"])
        writer.writeheader()
        seen: set[int] = set()
        for row in reader:
            prompt_len = int(row["prompt_len"])
            arrival = row.get("arrival_ms", "")
            key = hash((prompt_len, arrival))
            if row.get("policy") != "fifo" or key in seen:
                continue
            seen.add(key)
            writer.writerow(
                {
                    "source": args.source,
                    "prompt_len": prompt_len,
                    "total_len": prompt_len + args.gen_length,
                }
            )


if __name__ == "__main__":
    main()
