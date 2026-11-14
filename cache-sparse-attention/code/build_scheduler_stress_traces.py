"""Build compact synthetic length traces for scheduler stress experiments."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "cache-sparse-attention" / "results"


def add_rows(rows: list[dict], *, source: str, lengths: list[int], gen_length: int, seed: int) -> None:
    rng = random.Random(seed)
    shuffled = list(lengths)
    rng.shuffle(shuffled)
    for idx, prompt_len in enumerate(shuffled):
        rows.append(
            {
                "request_uid": f"{source}-{idx}",
                "source": source,
                "prompt_len": int(prompt_len),
                "total_len": int(prompt_len) + gen_length,
            }
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=RESULTS / "scheduler_stress_traces_v1.csv")
    p.add_argument("--seed", type=int, default=20260626)
    p.add_argument("--gen_length", type=int, default=128)
    args = p.parse_args()

    rows: list[dict] = []

    # Fairness stress: many shape-compatible short requests, a medium cluster,
    # and a few long outliers that a shape-only scheduler can defer.
    fairness = [64] * 32 + [512] * 16 + [1792] * 16
    add_rows(rows, source="fairness_bimodal", lengths=fairness, gen_length=args.gen_length, seed=args.seed)

    # Heterogeneity sweep with the same request count and mean-ish scale but
    # progressively wider length spread.
    low = [512, 544, 576, 608] * 12
    mid = [256] * 16 + [768] * 16 + [1280] * 16
    high = [64] * 16 + [512] * 16 + [1792] * 16
    add_rows(rows, source="hetero_low", lengths=low, gen_length=args.gen_length, seed=args.seed + 1)
    add_rows(rows, source="hetero_mid", lengths=mid, gen_length=args.gen_length, seed=args.seed + 2)
    add_rows(rows, source="hetero_high", lengths=high, gen_length=args.gen_length, seed=args.seed + 3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["request_uid", "source", "prompt_len", "total_len"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} rows={len(rows)}")


if __name__ == "__main__":
    main()
