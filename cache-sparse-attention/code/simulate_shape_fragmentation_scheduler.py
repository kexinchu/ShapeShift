"""Trace-driven shape-fragmentation simulator for dLLM serving.

This tests whether online batching can waste transformer-body work by mixing
short and long requests into the same rectangular tensor. It is intentionally
simple: every request has a fixed total sequence length, each batch costs
`batch_size * max_total_len`, and policies differ only in which waiting
requests are grouped.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "cache-sparse-attention" / "results"


@dataclass
class Req:
    uid: int
    task: str
    total_len: int
    arrival: float


def load_lengths(path: Path, *, source_filter: set[str] | None, min_len: int, max_len: int) -> list[dict]:
    seen = set()
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if "total_len" in row and ("baseline_hash" not in row or "prompt_idx" not in row):
                key = (row.get("source", row.get("task", "trace")), row.get("request_uid", len(rows)))
                if key in seen:
                    continue
                seen.add(key)
                source = row.get("source", row.get("task", "trace"))
                total_len = int(row["total_len"])
                if source_filter and source not in source_filter:
                    continue
                if total_len < min_len or (max_len > 0 and total_len > max_len):
                    continue
                rows.append({"task": source, "total_len": total_len})
                continue
            if source_filter and row["task"] not in source_filter:
                continue
            total_len = int(row["total_len"])
            if total_len < min_len or (max_len > 0 and total_len > max_len):
                continue
            key = (row["task"], row["prompt_idx"], row["baseline_hash"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"task": row["task"], "total_len": total_len})
    return rows


def synthetic_lengths(profile: str, *, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        if profile == "longctx_mixture":
            u = rng.random()
            if u < 0.55:
                total_len = rng.randint(512, 1024)
            elif u < 0.85:
                total_len = rng.randint(1024, 2048)
            else:
                total_len = rng.randint(2048, 4096)
        elif profile == "multitenant_wide":
            u = rng.random()
            if u < 0.40:
                total_len = rng.randint(160, 384)
            elif u < 0.70:
                total_len = rng.randint(512, 1024)
            elif u < 0.90:
                total_len = rng.randint(1024, 2048)
            else:
                total_len = rng.randint(3072, 4096)
        elif profile == "loguniform_128_4096":
            lo, hi = 128, 4096
            total_len = int(round(2 ** rng.uniform(math_log2(lo), math_log2(hi))))
        else:
            raise ValueError(profile)
        rows.append({"task": profile, "total_len": total_len})
    return rows


def math_log2(x: int) -> float:
    import math

    return math.log2(x)


def make_workload(base: list[dict], *, n_requests: int, arrival: str, rate: float, seed: int) -> list[Req]:
    rng = random.Random(seed)
    samples = [base[i % len(base)] for i in range(n_requests)]
    rng.shuffle(samples)
    now = 0.0
    reqs = []
    for uid, row in enumerate(samples):
        if arrival == "all_at_once":
            at = 0.0
        elif arrival == "constant":
            at = uid / rate
        elif arrival == "poisson":
            now += rng.expovariate(rate)
            at = now
        else:
            raise ValueError(arrival)
        reqs.append(Req(uid=uid, task=row["task"], total_len=row["total_len"], arrival=at))
    return sorted(reqs, key=lambda r: (r.arrival, r.uid))


def pick_batch(waiting: list[Req], *, policy: str, max_batch: int, tolerance: int) -> list[Req]:
    if policy == "fifo":
        return waiting[:max_batch]
    if policy == "length_sorted":
        return sorted(waiting, key=lambda r: (r.total_len, r.arrival, r.uid))[:max_batch]
    if policy == "length_bucket":
        seed = min(waiting, key=lambda r: (r.total_len, r.arrival, r.uid))
        compatible = [r for r in waiting if abs(r.total_len - seed.total_len) <= tolerance]
        if len(compatible) >= max_batch:
            return sorted(compatible, key=lambda r: (r.arrival, r.uid))[:max_batch]
        # Fill with nearest lengths if the bucket is underfull.
        return sorted(waiting, key=lambda r: (abs(r.total_len - seed.total_len), r.arrival, r.uid))[:max_batch]
    if policy == "oldest_bucket":
        seed = min(waiting, key=lambda r: (r.arrival, r.uid))
        compatible = [r for r in waiting if abs(r.total_len - seed.total_len) <= tolerance]
        if len(compatible) >= max_batch:
            return sorted(compatible, key=lambda r: (r.arrival, r.uid))[:max_batch]
        return sorted(waiting, key=lambda r: (abs(r.total_len - seed.total_len), r.arrival, r.uid))[:max_batch]
    raise ValueError(policy)


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]


def simulate(reqs: list[Req], *, policy: str, max_batch: int, tolerance: int) -> dict:
    future = deque(reqs)
    waiting: list[Req] = []
    now = 0.0
    done_lat = []
    full_rows = 0
    true_rows = 0
    n_batches = 0
    batch_sizes = []
    while future or waiting:
        if not waiting and future and future[0].arrival > now:
            now = future[0].arrival
        while future and future[0].arrival <= now:
            waiting.append(future.popleft())
        batch = pick_batch(waiting, policy=policy, max_batch=max_batch, tolerance=tolerance)
        chosen = set(r.uid for r in batch)
        waiting = [r for r in waiting if r.uid not in chosen]
        max_len = max(r.total_len for r in batch)
        batch_full = len(batch) * max_len
        batch_true = sum(r.total_len for r in batch)
        full_rows += batch_full
        true_rows += batch_true
        n_batches += 1
        batch_sizes.append(len(batch))
        # Normalize time by one 128-step request token-row unit; only ratios
        # matter, so this can be arbitrary.
        service = batch_full / 1000.0
        now += service
        for r in batch:
            done_lat.append(now - r.arrival)
    return {
        "policy": policy,
        "max_batch": max_batch,
        "tolerance": tolerance,
        "n_requests": len(reqs),
        "n_batches": n_batches,
        "makespan": now,
        "req_per_time": len(reqs) / now,
        "mean_batch_size": sum(batch_sizes) / len(batch_sizes),
        "padding_ratio": full_rows / true_rows,
        "padding_waste_frac": 1.0 - true_rows / full_rows,
        "mean_latency": sum(done_lat) / len(done_lat),
        "p95_latency": percentile(done_lat, 0.95),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=Path, default=RESULTS / "exact_length_mixed_combined_mmlu_gsm_math_n768_mb16_v1_rows.csv")
    p.add_argument("--out", type=Path, default=RESULTS / "shape_fragmentation_scheduler_v1.csv")
    p.add_argument("--synthetic_profile", choices=["", "longctx_mixture", "multitenant_wide", "loguniform_128_4096"], default="")
    p.add_argument("--source_filter", default="")
    p.add_argument("--min_len", type=int, default=0)
    p.add_argument("--max_len", type=int, default=0)
    p.add_argument("--n_requests", type=int, default=768)
    p.add_argument("--arrival", choices=["all_at_once", "constant", "poisson"], default="poisson")
    p.add_argument("--rate", type=float, default=8.0)
    p.add_argument("--max_batches", default="4,8,16")
    p.add_argument("--tolerances", default="0,16,32,64")
    p.add_argument("--seeds", default="1,2,3,4,5")
    args = p.parse_args()

    source_filter = set(x for x in args.source_filter.split(",") if x)
    base = (
        synthetic_lengths(args.synthetic_profile, n=args.n_requests, seed=0)
        if args.synthetic_profile
        else load_lengths(args.rows, source_filter=source_filter or None, min_len=args.min_len, max_len=args.max_len)
    )
    if not base:
        raise ValueError("no requests after filtering")
    policies = ["fifo", "length_sorted", "length_bucket", "oldest_bucket"]
    out_rows = []
    for seed in [int(x) for x in args.seeds.split(",") if x]:
        reqs = make_workload(base, n_requests=args.n_requests, arrival=args.arrival, rate=args.rate, seed=seed)
        for max_batch in [int(x) for x in args.max_batches.split(",") if x]:
            for tol in [int(x) for x in args.tolerances.split(",") if x]:
                for policy in policies:
                    if policy in ("fifo", "length_sorted") and tol != 0:
                        continue
                    row = simulate(reqs, policy=policy, max_batch=max_batch, tolerance=tol)
                    row.update({"seed": seed, "arrival": args.arrival, "rate": args.rate})
                    out_rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {args.out} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
