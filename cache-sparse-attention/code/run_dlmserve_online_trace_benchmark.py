"""Direct online-arrival dlmserve benchmark.

This script drives real ``dlmserve`` through ``Engine.generate_async`` with one
request admitted per arrival event (controlled by ``--arrival_gap_ms``), while a
single background ``Engine._engine_loop`` task keeps the scheduler flowing.

It supports FIFO / length-bucket (aka shape_only) / aging-bucket
(aka shape_aging_old) policies, token_budget (ShapeShift full), and the legacy
length-sorted admission diagnostic.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import math
import random
import sys
import time
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any



ROOT = Path(__file__).resolve().parents[2]
CSA = ROOT / "cache-sparse-attention"
RESULTS = CSA / "results"
TRACE = RESULTS / "real_length_trace_sharegpt5k_v1_rows.csv"
DLMSERVE = CSA / "external" / "dlmserve"

sys.path.insert(0, str(DLMSERVE))
sys.path.insert(0, str(ROOT))



POLICIES = (
    "fifo",
    "length_bucket",
    "shape_only",
    "aging_bucket",
    "shape_aging_old",
    "token_budget",
    "shapeshift",
    "length_sorted_admission",
)
BLOCK_BUCKETS = (32, 64, 128, 256, 512, 1024)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def _length_bucket(prompt_len: int) -> int:
    for bucket in BLOCK_BUCKETS:
        if prompt_len <= bucket:
            return bucket
    return BLOCK_BUCKETS[-1]


def _normalize_scheduler_policy(policy: str) -> str:
    """Map public policy aliases to scheduler policy names.

    - shape_only -> length_bucket (shape-only batching only)
    - shape_aging_old -> aging_bucket (age-based fairness only)
    """
    alias = {
        "shape_only": "length_bucket",
        "shape_aging_old": "aging_bucket",
    }
    key = policy.strip().lower()
    return alias.get(key, key)


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


def make_prompt(tokenizer, target_tokens: int) -> tuple[str, int]:
    # Stable word choice keeps the probe input compact and easy to scale.
    word = "the"
    text = " ".join([word] * max(1, target_tokens))
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= target_tokens:
        return text, len(ids)

    lo = 1
    hi = max(1, target_tokens)
    best_text = text
    best_len = len(ids)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = " ".join([word] * mid)
        cand_len = len(tokenizer(candidate, add_special_tokens=False)["input_ids"])
        if cand_len <= target_tokens:
            best_text = candidate
            best_len = cand_len
            lo = mid + 1
        else:
            hi = mid - 1

    return best_text, best_len


async def benchmark_policy(
    *,
    engine: Engine,
    prompts: list[str],
    prompt_lens: list[int],
    params: SamplingParams,
    max_batch: int,
    policy: str,
    n_requests: int,
    arrival_gap_ms: float,
    seed: int,
    shape_slo_ms: float,
    active_pool_factor: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy={policy!r}")

    if policy == "length_sorted_admission":
        order = sorted(range(len(prompt_lens)), key=lambda i: (prompt_lens[i], i))
    else:
        order = list(range(len(prompt_lens)))

    ordered_prompts = [prompts[i] for i in order]
    ordered_prompt_lens = [prompt_lens[i] for i in order]
    n_requests = min(n_requests, len(ordered_prompts))

    by_index: list[dict[str, Any]] = []
    for idx in range(n_requests):
        by_index.append(
            {
                "policy": policy,
                "seed": seed,
                "prompt_len": ordered_prompt_lens[idx],
                "length_bucket": _length_bucket(ordered_prompt_lens[idx]),
                "arrival_ms": float("nan"),
                "start_ms": float("nan"),
                "end_ms": float("nan"),
                "output_tokens": 0,
                "status": "queued",
                "error": "",
            }
        )

    scheduler = engine._scheduler
    scheduler_policy = (
        "fifo" if policy == "length_sorted_admission" else _normalize_scheduler_policy(policy)
    )
    engine._scheduler.shape_policy = scheduler_policy
    engine._scheduler.shape_slo_ms = shape_slo_ms
    engine._scheduler.active_pool_factor = active_pool_factor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    arrival_queue: deque[tuple[int, float]] = deque()
    req_id_to_index: dict[int, int] = {}
    tracked_req_ids: set[int] = set()
    original_admit = scheduler.admit

    def _tracked_admit(
        state: torch.Tensor,
        params_in: SamplingParams,
        generator: torch.Generator | None = None,
        future: asyncio.Future[Any] | None = None,
    ):
        req = original_admit(state, params_in, generator=generator, future=future)
        arrival = arrival_queue.popleft() if arrival_queue else None
        if req is None:
            if arrival is not None:
                index, arrival_ms = arrival
                by_index[index]["arrival_ms"] = arrival_ms
                by_index[index]["status"] = "failed"
                by_index[index]["error"] = "queue backpressure"
            return None
        req_id_to_index[req.req_id] = -1
        if arrival is not None:
            index, arrival_ms = arrival
            req_id_to_index[req.req_id] = index
            by_index[index]["arrival_ms"] = arrival_ms
            by_index[index]["status"] = "admitted"

        if req._future is not None and req_id_to_index.get(req.req_id, -1) >= 0:
            rid = req.req_id
            tracked_req_ids.add(rid)

            def _on_done(fut: asyncio.Future[Any]) -> None:
                index = req_id_to_index.get(rid, -1)
                if index < 0 or index >= len(by_index):
                    return
                by_index[index]["end_ms"] = time.perf_counter() * 1000.0
                try:
                    out = fut.result()
                except Exception as exc:
                    by_index[index]["status"] = "failed"
                    by_index[index]["error"] = repr(exc)
                    by_index[index]["output_tokens"] = 0
                else:
                    by_index[index]["status"] = "done"
                    by_index[index]["output_tokens"] = int(out.output_ids.shape[0])

            req._future.add_done_callback(_on_done)
        return req

    scheduler.admit = _tracked_admit

    # Keep one engine loop running and submit requests according to the arrival
    # gap. Requests are always submitted one-at-a-time by generate_async.
    loop = asyncio.get_running_loop()
    engine_task = loop.create_task(engine._engine_loop())
    all_done = asyncio.Event()
    submit_results: list[tuple[int, dict[str, Any]]] = []

    async def submit_request(index: int, prompt: str) -> None:
        try:
            outputs = await engine.generate_async([prompt], params=params, prerendered=True)
            submit_results.append((index, {"output_tokens": len(outputs[0].output_ids)}))
        except Exception as exc:
            submit_results.append((index, {"error": repr(exc)}))

    start_monotonic_ms = time.perf_counter() * 1000.0

    async def monitor_start() -> None:
        while not all_done.is_set():
            active_ids = {req.req_id for req in getattr(scheduler, "_active", [])}
            now_ms = time.perf_counter() * 1000.0
            for req_id in list(tracked_req_ids):
                idx = req_id_to_index.get(req_id, -1)
                if idx < 0 or idx >= len(by_index):
                    continue
                if not math.isnan(by_index[idx]["start_ms"]) or by_index[idx]["status"] == "done":
                    continue
                if req_id in active_ids:
                    by_index[idx]["start_ms"] = now_ms
            await asyncio.sleep(0)

    monitor_task = loop.create_task(monitor_start())

    submit_tasks: list[asyncio.Task[None]] = []
    try:
        for idx, prompt in enumerate(ordered_prompts[:n_requests]):
            if idx > 0 and arrival_gap_ms > 0:
                await asyncio.sleep(arrival_gap_ms / 1000.0)
            arrival_ms = time.perf_counter() * 1000.0
            arrival_queue.append((idx, arrival_ms))
            submit_tasks.append(loop.create_task(submit_request(idx, prompt)))

        await asyncio.gather(*submit_tasks)
    finally:
        all_done.set()
        await monitor_task
        engine._shutdown = True
        engine_task.cancel()
        with suppress(asyncio.CancelledError):
            await engine_task
        engine._shutdown = False

    # Restore the original admit path before continuing.
    scheduler.admit = original_admit

    # Fill in any uncaught errors from direct submission exceptions.
    for idx, result in submit_results:
        if by_index[idx]["status"] == "failed":
            by_index[idx]["error"] = result.get("error", by_index[idx]["error"])
            by_index[idx]["status"] = "failed"
            continue
        if "error" in result:
            by_index[idx]["status"] = "failed"
            by_index[idx]["error"] = result["error"]
        elif "output_tokens" in result:
            by_index[idx]["output_tokens"] = int(result["output_tokens"])

    for item in by_index:
        if item["status"] == "queued":
            item["status"] = "failed"
            item["error"] = "request not admitted (possible queue backpressure)"

    # Normalize absolute timestamps into latencies:
    # arrival_ms/start_ms/end_ms are absolute wall-clock milliseconds.
    # arrival/end latency are retained as offsets from t0 for makespan recovery;
    # response latency is the per-request completion-after-arrival metric.
    for item in by_index:
        if math.isnan(item["arrival_ms"]):
            continue
        item["arrival_latency_ms"] = item["arrival_ms"] - start_monotonic_ms
        if not math.isnan(item["start_ms"]):
            item["start_latency_ms"] = item["start_ms"] - item["arrival_ms"]
        else:
            item["start_latency_ms"] = float("nan")
        if not math.isnan(item["end_ms"]):
            item["end_latency_ms"] = item["end_ms"] - start_monotonic_ms
            item["response_latency_ms"] = item["end_ms"] - item["arrival_ms"]
        else:
            item["end_latency_ms"] = float("nan")
            item["response_latency_ms"] = float("nan")

    done_latencies_ms = [
        item["response_latency_ms"]
        for item in by_index
        if item["status"] == "done" and not math.isnan(item.get("response_latency_ms", float("nan")))
    ]
    done_tokens = [item["output_tokens"] for item in by_index if item["status"] == "done"]
    completed_end_ms = [
        item["end_ms"] for item in by_index if not math.isnan(item.get("end_ms", float("nan")))
    ]
    admitted_arrival_ms = [
        item["arrival_ms"]
        for item in by_index
        if not math.isnan(item.get("arrival_ms", float("nan")))
    ]
    if completed_end_ms and admitted_arrival_ms:
        wall_ms = max(completed_end_ms) - min(admitted_arrival_ms)
    else:
        wall_ms = float("nan")
    req_s = len(done_latencies_ms) / max(1e-9, wall_ms / 1000.0)
    tok_s = sum(done_tokens) / max(1e-9, wall_ms / 1000.0)
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if torch.cuda.is_available() else float("nan")

    summary = {
        "policy": policy,
        "max_batch": max_batch,
        "n_requests": n_requests,
        "steps": params.num_denoising_steps,
        "gen_length": params.gen_length,
        "arrival_gap_ms": arrival_gap_ms,
        "shape_slo_ms": shape_slo_ms,
        "active_pool_factor": active_pool_factor,
        "seed": seed,
        "wall_ms": wall_ms,
        "req_per_s": req_s,
        "tok_per_s": tok_s,
        "p50_latency_ms": percentile(done_latencies_ms, 0.50),
        "mean_latency_ms": sum(done_latencies_ms) / max(1, len(done_latencies_ms))
        if done_latencies_ms
        else float("nan"),
        "p95_latency_ms": percentile(done_latencies_ms, 0.95),
        "p99_latency_ms": percentile(done_latencies_ms, 0.99),
        "max_latency_ms": max(done_latencies_ms) if done_latencies_ms else float("nan"),
        "peak_gb": peak_gb,
        "n_completed": len(done_latencies_ms),
        "n_failed": n_requests - len(done_latencies_ms),
        "mean_prompt_len": sum(item["prompt_len"] for item in by_index) / max(1, len(by_index)),
        "min_prompt_len": min(item["prompt_len"] for item in by_index) if by_index else 0,
        "max_prompt_len": max(item["prompt_len"] for item in by_index) if by_index else 0,
    }
    return by_index, summary, wall_ms


def write_outputs(tag: str, rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    rows_out = RESULTS / f"{tag}_rows.csv"
    summary_out = RESULTS / f"{tag}_summary.csv"
    if rows:
        with rows_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {rows_out}")
    if summaries:
        with summary_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"saved {summary_out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "model_cache" / "LLaDA-8B-Instruct"))
    p.add_argument("--dtype", default="bf16", choices=["bf16", "int4"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--source", default="sharegpt_multiturn_accum")
    p.add_argument("--trace", type=Path, default=TRACE)
    p.add_argument("--n_requests", type=int, default=24)
    p.add_argument("--min_total_len", type=int, default=0)
    p.add_argument("--max_total_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--max_batch", type=int, default=4)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--gen_length", type=int, default=128)
    p.add_argument("--arrival_gap_ms", type=float, default=0.0)
    p.add_argument("--shape_slo_ms", type=float, default=5000.0)
    p.add_argument("--active_pool_factor", type=int, default=4)
    p.add_argument("--policies", default="fifo,length_bucket,aging_bucket,shapeshift")
    p.add_argument("--padding_ratio", type=float, default=1.25)
    p.add_argument("--tag", default="dlmserve_online_smoke")
    args = p.parse_args()
    if min(args.n_requests, args.max_batch, args.steps, args.gen_length, args.active_pool_factor) < 1:
        p.error("request, batch, step, generation and active-pool counts must be positive")
    if not math.isfinite(args.arrival_gap_ms) or args.arrival_gap_ms < 0:
        p.error("arrival_gap_ms must be finite and nonnegative")
    if Path(args.tag).name != args.tag or args.tag in {".", ".."}:
        p.error("tag must be a plain filename prefix")

    global torch, Engine, SamplingParams
    try:
        import torch
        from shapeshift.engine import Engine
        from dlmserve.sampler import SamplingParams
    except ImportError as exc:
        p.error(f"Missing runtime dependency: {exc}. See docs/REPRODUCIBILITY.md; "
                "install this repository with its pinned runtime extra.")

    RESULTS.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        p.error("CUDA unavailable; a real GPU benchmark cannot run on this machine.")

    raw_lens = load_lengths(
        args.trace,
        source=args.source,
        n_requests=args.n_requests,
        min_total_len=args.min_total_len,
        max_total_len=args.max_total_len,
        seed=args.seed,
    )

    temp_engine = Engine(
        model_id=args.model,
        dtype=args.dtype,
        device=args.device,
        max_batch=args.max_batch,
    )
    prompts: list[str] = []
    prompt_lens: list[int] = []
    for target in raw_lens:
        text, true_len = make_prompt(temp_engine.tokenizer, max(1, target))
        prompts.append(text)
        prompt_lens.append(true_len)
    del temp_engine

    params = SamplingParams(
        num_denoising_steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.gen_length,
        temperature=0.0,
        seed=args.seed,
    )

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    requested_policies = [x for x in args.policies.split(",") if x]
    for policy in requested_policies:
        policy = policy.strip().lower()
        if policy not in POLICIES:
            raise ValueError(f"unsupported policy: {policy}")
        print(f"running policy={policy}")
        scheduler_trace_out = RESULTS / f"{args.tag}_{policy}_scheduler_trace.csv"
        # Reuse one engine per policy to keep memory clean and scheduler state fresh.
        engine = Engine(
            model_id=args.model,
            dtype=args.dtype,
            device=args.device,
            max_batch=args.max_batch,
            scheduler_policy="fifo",
            scheduler_slo_ms=args.shape_slo_ms,
            scheduler_active_pool_factor=args.active_pool_factor,
            scheduler_trace_csv=str(scheduler_trace_out),
            scheduler_padding_ratio=args.padding_ratio,
        )
        try:
            per_rows, summary, _ = asyncio.run(
                benchmark_policy(
                    engine=engine,
                    prompts=prompts,
                    prompt_lens=prompt_lens,
                    params=params,
                    max_batch=args.max_batch,
                    policy=policy,
                    n_requests=args.n_requests,
                    arrival_gap_ms=args.arrival_gap_ms,
                    seed=args.seed,
                    shape_slo_ms=args.shape_slo_ms,
                    active_pool_factor=args.active_pool_factor,
                    device=torch.device(args.device),
                )
            )
        finally:
            engine._shutdown = True
            del engine
            torch.cuda.empty_cache()
        for row in per_rows:
            row.update(
                {
                    "source": args.source,
                    "min_total_len": args.min_total_len,
                    "max_total_len": args.max_total_len,
                    "arrival_gap_ms": args.arrival_gap_ms,
                    "max_batch": args.max_batch,
                    "dtype": args.dtype,
                    "device": args.device,
                    "shape_slo_ms": args.shape_slo_ms,
                    "active_pool_factor": args.active_pool_factor,
                }
            )
        rows.extend(per_rows)
        summary["scheduler_trace_csv"] = str(scheduler_trace_out)
        summary["implementation"] = "reconstructed-v0.1.0"
        summary["padding_ratio_budget"] = args.padding_ratio
        summaries.append(summary)
        write_outputs(args.tag, rows, summaries)
        print(
            f"{policy}: req/s={summary['req_per_s']:.3f}, tok/s={summary['tok_per_s']:.1f}, "
            f"mean={summary['mean_latency_ms']/1000:.3f}s, p95={summary['p95_latency_ms']/1000:.3f}s, "
            f"peak={summary['peak_gb']:.2f}GB, failed={summary['n_failed']}"
        )

    if rows:
        write_outputs(args.tag, rows, summaries)
    else:
        print("no rows emitted")

    # Policy-relative speedups (against FIFO when present).
    fifo = next((row for row in summaries if row["policy"] == "fifo"), None)
    if fifo is not None:
        for row in summaries:
            row["speedup_vs_fifo_req_s"] = row["req_per_s"] / max(1e-9, fifo["req_per_s"])
            row["speedup_vs_fifo_p95_latency"] = fifo["p95_latency_ms"] / max(
                1e-9, row["p95_latency_ms"]
            )
    if rows:
        write_outputs(args.tag, rows, summaries)
    else:
        print("no rows emitted")


if __name__ == "__main__":
    main()
