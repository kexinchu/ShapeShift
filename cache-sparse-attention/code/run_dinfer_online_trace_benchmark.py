"""Trace-driven online wrapper around dInfer DiffusionLLMServing.

This benchmark is intentionally separate from dInfer's offline
``benchmark_dataset.py``.  It replays the same request-length traces used by the
dlmserve online experiments and dispatches synchronous dInfer batches according
to a simple online policy.  This gives request-level latency and throughput under
the same arrival process, while making clear that dInfer itself is used as the
body-side generation engine.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
CSA = ROOT / "cache-sparse-attention"
RESULTS = CSA / "results"
DINFER = CSA / "external" / "dInfer"

sys.path.insert(0, str(DINFER / "python"))

from dinfer.decoding.serving import DiffusionLLMServing, SamplingParams  # noqa: E402


POLICIES = ("fifo", "length_bucket", "aging_bucket", "length_sorted_admission")
BLOCK_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048)


@dataclass
class Request:
    index: int
    prompt_len: int
    prompt_ids: torch.Tensor
    arrival_ms: float
    start_ms: float = float("nan")
    end_ms: float = float("nan")
    output_tokens: int = 0
    status: str = "queued"
    error: str = ""


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def length_bucket(prompt_len: int) -> int:
    for bucket in BLOCK_BUCKETS:
        if prompt_len <= bucket:
            return bucket
    return BLOCK_BUCKETS[-1]


def load_lengths(
    path: Path,
    *,
    source: str,
    n_requests: int,
    min_total_len: int,
    max_total_len: int,
    seed: int,
) -> list[int]:
    rows: list[int] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row_source = row.get("source", row.get("task", ""))
            if row_source != source:
                continue
            total_len = int(row["total_len"])
            if total_len < min_total_len or total_len > max_total_len:
                continue
            rows.append(int(row["prompt_len"]))

    if len(rows) < n_requests:
        raise ValueError(f"not enough trace rows after filtering: {len(rows)} < {n_requests}")

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n_requests]


def make_prompt_ids(tokenizer: AutoTokenizer, target_tokens: int) -> torch.Tensor:
    text = " ".join(["the"] * max(1, target_tokens))
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) > target_tokens:
        ids = ids[:target_tokens]
    if not ids:
        ids = [tokenizer.eos_token_id or 0]
    return torch.tensor(ids, dtype=torch.long)


def select_batch(
    ready: list[Request],
    *,
    policy: str,
    max_batch: int,
    now_ms: float,
    shape_slo_ms: float,
) -> list[Request]:
    if policy == "fifo":
        return sorted(ready, key=lambda r: (r.arrival_ms, r.index))[:max_batch]
    if policy == "length_sorted_admission":
        return sorted(ready, key=lambda r: (r.prompt_len, r.arrival_ms, r.index))[:max_batch]

    if policy == "aging_bucket":
        overdue = [r for r in ready if now_ms - r.arrival_ms >= shape_slo_ms]
        if overdue:
            anchor = min(overdue, key=lambda r: (r.arrival_ms, r.index))
        else:
            anchor = min(ready, key=lambda r: (r.prompt_len, r.arrival_ms, r.index))
    elif policy == "length_bucket":
        anchor = min(ready, key=lambda r: (r.prompt_len, r.arrival_ms, r.index))
    else:
        raise ValueError(f"unknown policy={policy}")

    return sorted(
        ready,
        key=lambda r: (abs(r.prompt_len - anchor.prompt_len), r.arrival_ms, r.index),
    )[:max_batch]


def pad_batch(batch: list[Request], pad_id: int) -> torch.Tensor:
    max_len = max(req.prompt_ids.shape[0] for req in batch)
    out = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, req in enumerate(batch):
        out[i, : req.prompt_ids.shape[0]] = req.prompt_ids
    return out


def run_policy(
    *,
    serving: DiffusionLLMServing,
    policy: str,
    base_requests: list[Request],
    max_batch: int,
    gen_length: int,
    block_length: int,
    arrival_gap_ms: float,
    shape_slo_ms: float,
    pad_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requests = [
        Request(
            index=req.index,
            prompt_len=req.prompt_len,
            prompt_ids=req.prompt_ids,
            arrival_ms=req.index * arrival_gap_ms,
        )
        for req in base_requests
    ]
    pending = sorted(requests, key=lambda r: (r.arrival_ms, r.index))
    ready: list[Request] = []
    done: list[Request] = []
    now_ms = 0.0
    start_wall = time.perf_counter() * 1000.0
    batch_rows: list[dict[str, Any]] = []

    while pending or ready:
        if not ready and pending:
            now_ms = max(now_ms, pending[0].arrival_ms)
        while pending and pending[0].arrival_ms <= now_ms:
            ready.append(pending.pop(0))

        batch = select_batch(
            ready,
            policy=policy,
            max_batch=max_batch,
            now_ms=now_ms,
            shape_slo_ms=shape_slo_ms,
        )
        for req in batch:
            ready.remove(req)

        batch_input = pad_batch(batch, pad_id)
        start_ms = now_ms
        wall_start = time.perf_counter() * 1000.0
        try:
            out = serving.generate(batch_input, gen_length=gen_length, block_length=block_length)
        except Exception as exc:  # keep benchmark auditable
            service_ms = time.perf_counter() * 1000.0 - wall_start
            for req in batch:
                req.start_ms = start_ms
                req.end_ms = start_ms + service_ms
                req.status = "failed"
                req.error = repr(exc)
                done.append(req)
        else:
            service_ms = time.perf_counter() * 1000.0 - wall_start
            for req in batch:
                req.start_ms = start_ms
                req.end_ms = start_ms + service_ms
                req.status = "done"
                req.output_tokens = gen_length
                done.append(req)
            del out

        now_ms += service_ms
        while pending and pending[0].arrival_ms <= now_ms:
            ready.append(pending.pop(0))

        batch_rows.append(
            {
                "policy": policy,
                "batch_size": len(batch),
                "batch_max_prompt_len": max(req.prompt_len for req in batch),
                "batch_mean_prompt_len": sum(req.prompt_len for req in batch) / len(batch),
                "service_ms": service_ms,
                "start_ms": start_ms,
                "end_ms": now_ms,
            }
        )

    rows: list[dict[str, Any]] = []
    for req in sorted(done, key=lambda r: r.index):
        rows.append(
            {
                "policy": policy,
                "request_index": req.index,
                "prompt_len": req.prompt_len,
                "length_bucket": length_bucket(req.prompt_len),
                "arrival_ms": req.arrival_ms,
                "start_latency_ms": req.start_ms - req.arrival_ms,
                "response_latency_ms": req.end_ms - req.arrival_ms,
                "output_tokens": req.output_tokens,
                "status": req.status,
                "error": req.error,
            }
        )

    completed = [row for row in rows if row["status"] == "done"]
    latencies = [float(row["response_latency_ms"]) for row in completed]
    tokens = [int(row["output_tokens"]) for row in completed]
    wall_ms = max((row["response_latency_ms"] + row["arrival_ms"] for row in rows), default=0.0)
    if rows:
        wall_ms -= min(float(row["arrival_ms"]) for row in rows)
    elapsed_wall_ms = time.perf_counter() * 1000.0 - start_wall
    summary = {
        "policy": policy,
        "max_batch": max_batch,
        "n_requests": len(requests),
        "gen_length": gen_length,
        "block_length": block_length,
        "arrival_gap_ms": arrival_gap_ms,
        "shape_slo_ms": shape_slo_ms,
        "wall_ms": wall_ms,
        "elapsed_wall_ms": elapsed_wall_ms,
        "req_per_s": len(completed) / max(1e-9, wall_ms / 1000.0),
        "tok_per_s": sum(tokens) / max(1e-9, wall_ms / 1000.0),
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
        "n_completed": len(completed),
        "n_failed": len(requests) - len(completed),
        "mean_prompt_len": sum(req.prompt_len for req in requests) / max(1, len(requests)),
        "min_prompt_len": min((req.prompt_len for req in requests), default=0),
        "max_prompt_len": max((req.prompt_len for req in requests), default=0),
        "mean_batch_service_ms": sum(row["service_ms"] for row in batch_rows) / max(1, len(batch_rows)),
    }
    return rows, summary


def write_outputs(tag: str, rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    rows_out = RESULTS / f"{tag}_rows.csv"
    summary_out = RESULTS / f"{tag}_summary.csv"
    if rows:
        with rows_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if summaries:
        with summary_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    if rows or summaries:
        print(f"saved {rows_out}")
        print(f"saved {summary_out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "model_cache" / "LLaDA-8B-Instruct"))
    p.add_argument("--model_type", default="llada")
    p.add_argument("--trace", type=Path, default=RESULTS / "serving_workloads_v1_rows.csv")
    p.add_argument("--source", default="workload_sharegpt_multiturn")
    p.add_argument("--n_requests", type=int, default=8)
    p.add_argument("--min_total_len", type=int, default=0)
    p.add_argument("--max_total_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--max_batch", type=int, default=4)
    p.add_argument("--gen_length", type=int, default=128)
    p.add_argument("--block_length", type=int, default=128)
    p.add_argument("--arrival_gap_ms", type=float, default=0.0)
    p.add_argument("--shape_slo_ms", type=float, default=5000.0)
    p.add_argument("--policies", default="fifo,length_bucket,aging_bucket")
    p.add_argument("--tag", default="dinfer_online_smoke")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--skip_warmup", action="store_true")
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; dInfer online wrapper requires GPU")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    lengths = load_lengths(
        args.trace,
        source=args.source,
        n_requests=args.n_requests,
        min_total_len=args.min_total_len,
        max_total_len=args.max_total_len,
        seed=args.seed,
    )
    requests = [
        Request(
            index=i,
            prompt_len=len(make_prompt_ids(tokenizer, max(1, target))),
            prompt_ids=make_prompt_ids(tokenizer, max(1, target)),
            arrival_ms=i * args.arrival_gap_ms,
        )
        for i, target in enumerate(lengths)
    ]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    sample_params = SamplingParams(
        threshold=0.95,
        cache="",
        temperature=0.0,
        early_stop=True,
        cont_weight=0.0,
        prefix_look=0,
        after_look=0,
        warmup_steps=0,
        enable_torch_compile=not args.no_compile,
        mask_id=126336,
        eos_id=126081,
        parallel_decoding="threshold",
        use_credit=False,
        use_bd=False,
        max_length=args.max_total_len + args.gen_length,
        ep_size=1,
        prefilling_limit=256,
        mini_batch_size=args.max_batch,
        batch_size=args.max_batch,
        use_naive_batching=False,
    )

    serving = DiffusionLLMServing(
        args.model,
        model_type=args.model_type,
        sample_params=sample_params,
        num_gpus=1,
        dp_size=1,
        tpep_size=1,
        backend="torch",
        timeout=args.timeout,
    )

    if not args.skip_warmup:
        warmup_batch = pad_batch(requests[: min(args.max_batch, len(requests))], pad_id)
        print(f"warming dInfer serving with batch={warmup_batch.shape[0]} len={warmup_batch.shape[1]}")
        _ = serving.generate(
            warmup_batch,
            gen_length=min(args.gen_length, args.block_length),
            block_length=min(args.gen_length, args.block_length),
        )
        torch.cuda.empty_cache()

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    try:
        for policy in [x for x in args.policies.split(",") if x]:
            print(f"running policy={policy}")
            rows, summary = run_policy(
                serving=serving,
                policy=policy,
                base_requests=requests,
                max_batch=args.max_batch,
                gen_length=args.gen_length,
                block_length=args.block_length,
                arrival_gap_ms=args.arrival_gap_ms,
                shape_slo_ms=args.shape_slo_ms,
                pad_id=pad_id,
            )
            for row in rows:
                row.update(
                    {
                        "source": args.source,
                        "seed": args.seed,
                        "max_batch": args.max_batch,
                        "backend": "dinfer_serving",
                    }
                )
            summary.update({"source": args.source, "seed": args.seed, "backend": "dinfer_serving"})
            all_rows.extend(rows)
            summaries.append(summary)
            write_outputs(args.tag, all_rows, summaries)
            print(
                f"{policy}: tok/s={summary['tok_per_s']:.2f}, "
                f"p95={summary['p95_latency_ms']/1000:.3f}s, failed={summary['n_failed']}"
            )
    finally:
        serving.stop_serving()

    fifo = next((row for row in summaries if row["policy"] == "fifo"), None)
    if fifo is not None:
        for row in summaries:
            row["speedup_vs_fifo_req_s"] = row["req_per_s"] / max(1e-9, fifo["req_per_s"])
            row["p95_ratio_vs_fifo"] = row["p95_latency_ms"] / max(1e-9, fifo["p95_latency_ms"])

    write_outputs(args.tag, all_rows, summaries)


if __name__ == "__main__":
    main()
