"""Fast-dLLM body-side throughput on serving-workload length traces.

This runner intentionally measures offline/body-side generation throughput, not
online request scheduling. It reuses the same workload length traces as the
dlmserve online experiments, pads each batch to the batch maximum with ordinary
tokens, and runs the official Fast-dLLM v1 LLaDA generation functions.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
CSA = ROOT / "cache-sparse-attention"
FAST_LLADA = CSA / "external" / "Fast-dLLM" / "v1" / "llada"
RESULTS = CSA / "results"

sys.path.insert(0, str(FAST_LLADA))

from generate import generate, generate_with_dual_cache, generate_with_prefix_cache  # noqa: E402
from model.modeling_llada import LLaDAModelLM  # noqa: E402


METHODS = {
    "vanilla": ("vanilla", generate, 128, None),
    "prefix_cache": ("prefix_cache", generate_with_prefix_cache, 128, None),
    "dual_cache": ("dual_cache", generate_with_dual_cache, 128, None),
    "parallel_t": ("parallel_t", generate, 32, 0.90),
    "prefix_cache_parallel_t": ("prefix_cache_parallel_t", generate_with_prefix_cache, 32, 0.90),
    "dual_cache_parallel_t": ("dual_cache_parallel_t", generate_with_dual_cache, 32, 0.90),
}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def read_lengths(path: Path, source: str, n_requests: int, seed: int) -> list[int]:
    rows: list[int] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("source") != source:
                continue
            rows.append(int(float(row["prompt_len"])))

    if not rows:
        raise ValueError(f"no rows for source={source!r} in {path}")

    # Match the online harness convention: deterministic shuffle by seed, then
    # take the requested prefix.
    import random

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n_requests]


def make_batches(lengths: list[int], batch_size: int, order: str) -> list[list[int]]:
    if order == "length_sorted":
        lengths = sorted(lengths)
    elif order != "fifo":
        raise ValueError(f"unsupported order: {order}")
    return [lengths[i : i + batch_size] for i in range(0, len(lengths), batch_size)]


def make_prompt(batch_lengths: list[int], token_id: int, device: torch.device) -> torch.Tensor:
    max_len = max(batch_lengths)
    # Pad with a normal lexical token rather than the mask token. This emulates
    # body-side batch padding without accidentally denoising prompt padding.
    return torch.full((len(batch_lengths), max_len), token_id, dtype=torch.long, device=device)


@torch.no_grad()
def run_method(
    *,
    model,
    method_name: str,
    fn,
    steps: int,
    threshold: float | None,
    batches: list[list[int]],
    token_id: int,
    device: torch.device,
    gen_length: int,
    block_length: int,
    warmup: bool,
) -> tuple[list[dict], dict]:
    if warmup and batches:
        prompt = make_prompt(batches[0], token_id, device)
        _ = fn(
            model,
            prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=0.0,
            remasking="low_confidence",
            threshold=threshold,
        )
        sync(device)

    rows: list[dict] = []
    total_sec = 0.0
    total_nfe = 0
    total_tokens = 0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for batch_idx, batch_lengths in enumerate(batches):
        prompt = make_prompt(batch_lengths, token_id, device)
        sync(device)
        start = time.perf_counter()
        _, nfe = fn(
            model,
            prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=0.0,
            remasking="low_confidence",
            threshold=threshold,
        )
        sync(device)
        elapsed = time.perf_counter() - start
        batch_tokens = len(batch_lengths) * gen_length
        total_sec += elapsed
        total_nfe += int(nfe)
        total_tokens += batch_tokens
        rows.append(
            {
                "method": method_name,
                "batch_idx": batch_idx,
                "batch_size": len(batch_lengths),
                "max_prompt_len": max(batch_lengths),
                "min_prompt_len": min(batch_lengths),
                "mean_prompt_len": sum(batch_lengths) / len(batch_lengths),
                "elapsed_sec": elapsed,
                "nfe": int(nfe),
                "output_tokens": batch_tokens,
                "tok_per_s": batch_tokens / elapsed,
                "fps": int(nfe) / elapsed,
                "tpf": batch_tokens / int(nfe) if nfe else float("nan"),
            }
        )

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else float("nan")
    summary = {
        "method": method_name,
        "n_batches": len(batches),
        "n_requests": sum(len(batch) for batch in batches),
        "gen_length": gen_length,
        "block_length": block_length,
        "steps_arg": steps,
        "threshold": threshold,
        "total_sec": total_sec,
        "total_nfe": total_nfe,
        "total_tokens": total_tokens,
        "tok_per_s": total_tokens / total_sec,
        "fps": total_nfe / total_sec,
        "tpf": total_tokens / total_nfe if total_nfe else float("nan"),
        "mean_batch_sec": pd.DataFrame(rows)["elapsed_sec"].mean() if rows else float("nan"),
        "p95_batch_sec": pd.DataFrame(rows)["elapsed_sec"].quantile(0.95) if rows else float("nan"),
        "peak_gb": peak_gb,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--workload_name", required=True)
    parser.add_argument("--n_requests", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--order", choices=["fifo", "length_sorted"], default="fifo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_id", default=str(ROOT / "model_cache" / "LLaDA-8B-Instruct"))
    parser.add_argument("--methods", default="parallel_t,prefix_cache_parallel_t,dual_cache_parallel_t")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--tag", default="fastdllm_offline_trace")
    args = parser.parse_args()

    device = torch.device(args.device)
    RESULTS.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenized = tokenizer("the", add_special_tokens=False)["input_ids"]
    token_id = int(tokenized[0] if tokenized else 1)

    lengths = read_lengths(args.trace, args.source, args.n_requests, args.seed)
    batches = make_batches(lengths, args.batch_size, args.order)

    print(f"loading model {args.model_id} on {device}")
    model = LLaDAModelLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    rows: list[dict] = []
    summaries: list[dict] = []
    for method_key in [item.strip() for item in args.methods.split(",") if item.strip()]:
        if method_key not in METHODS:
            raise ValueError(f"unknown method {method_key}; choices={sorted(METHODS)}")
        method_name, fn, steps, threshold = METHODS[method_key]
        print(f"running {method_name} workload={args.workload_name} batches={len(batches)} order={args.order}")
        method_rows, summary = run_method(
            model=model,
            method_name=method_name,
            fn=fn,
            steps=steps,
            threshold=threshold,
            batches=batches,
            token_id=token_id,
            device=device,
            gen_length=args.gen_length,
            block_length=args.block_length,
            warmup=args.warmup,
        )
        for row in method_rows:
            row.update(
                {
                    "workload": args.workload_name,
                    "source": args.source,
                    "order": args.order,
                    "seed": args.seed,
                }
            )
        summary.update(
            {
                "workload": args.workload_name,
                "source": args.source,
                "order": args.order,
                "seed": args.seed,
                "backend": "Fast-dLLM-v1",
            }
        )
        rows.extend(method_rows)
        summaries.append(summary)
        print(
            f"{method_name}: {summary['tok_per_s']:.2f} tok/s, "
            f"{summary['fps']:.2f} fps, {summary['tpf']:.2f} tpf, "
            f"total={summary['total_sec']:.1f}s"
        )

    rows_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summaries)
    out_base = RESULTS / args.tag
    rows_path = out_base.with_name(out_base.name + "_rows.csv")
    summary_path = out_base.with_name(out_base.name + "_summary.csv")
    rows_df.to_csv(rows_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"saved {rows_path}")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
