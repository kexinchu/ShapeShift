"""Minimal same-machine comparison against Fast-dLLM v1 LLaDA generation.

This is not an integration of active-set output into Fast-dLLM.  It runs the
official Fast-dLLM v1 LLaDA generation functions on the same local checkpoint
and prompts, then compares vanilla, prefix-cache, and dual-cache decoding.

The goal is to establish a concrete SOTA baseline surface:

* how much body-side cache acceleration we can reproduce locally;
* whether the cache paths preserve final outputs under the same schedule;
* how many forward evaluations remain, motivating output-side composition.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
CSA = ROOT / "cache-sparse-attention"
FAST_LLADA = CSA / "external" / "Fast-dLLM" / "v1" / "llada"
OUT_DIR = CSA / "results"

sys.path.insert(0, str(FAST_LLADA))
sys.path.insert(0, str(ROOT / "early-stop" / "code"))
sys.path.insert(0, str(CSA / "code"))

from generate import (  # noqa: E402
    generate,
    generate_with_dual_cache,
    generate_with_prefix_cache,
    get_num_transfer_tokens,
)
from model.modeling_llada import LLaDAModelLM  # noqa: E402
from run_E5 import BLOCK_LENGTH, GEN_LENGTH, GEN_STEPS, MODEL_ID  # noqa: E402
from early_stop import MASK_ID  # noqa: E402
from run_selective_lm_head_e2e import build_prompt, forward_selective_lm_head, load_task, score  # noqa: E402


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def generate_active_output(
    model,
    prompt: torch.Tensor,
    *,
    steps: int = GEN_STEPS,
    gen_length: int = GEN_LENGTH,
    block_length: int = BLOCK_LENGTH,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    threshold: float | None = None,
) -> tuple[torch.Tensor, int]:
    if temperature != 0.0:
        raise ValueError("active-output Fast-DLLM probe supports temperature=0 only")
    if remasking != "low_confidence":
        raise ValueError("active-output Fast-DLLM probe supports low_confidence only")

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        MASK_ID,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == MASK_ID
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        i = 0
        while True:
            nfe += 1
            mask_index = x == MASK_ID
            mask_index[:, block_end:] = False
            positions = mask_index.any(dim=0).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                break

            logits = forward_selective_lm_head(model, x, positions)
            x0_local = torch.argmax(logits, dim=-1)
            prob = F.softmax(logits.to(torch.float64), dim=-1)
            conf = torch.gather(prob, dim=-1, index=x0_local.unsqueeze(-1)).squeeze(-1)
            valid = mask_index.index_select(1, positions)
            neg_inf = torch.tensor(torch.finfo(conf.dtype).min, device=conf.device, dtype=conf.dtype)
            conf = torch.where(valid, conf, neg_inf)

            transfer_local = torch.zeros_like(valid)
            if threshold is not None:
                transfer_local = valid & (conf >= threshold)
                max_conf_indices = torch.argmax(conf, dim=1, keepdim=True)
                force_mask = torch.zeros_like(transfer_local).scatter_(1, max_conf_indices, True)
                transfer_local = (transfer_local | force_mask) & valid
            else:
                quota = num_transfer_tokens[:, i].to(dtype=torch.long, device=conf.device)
                values, idx = torch.sort(conf, dim=1, descending=True)
                del values
                bsz, local_len = conf.shape
                cols = torch.arange(local_len, device=conf.device).unsqueeze(0).expand(bsz, local_len)
                select_sorted = cols < quota.unsqueeze(1).expand(bsz, local_len)
                transfer_int = torch.zeros(bsz, local_len, device=conf.device, dtype=torch.int8)
                transfer_local = transfer_int.scatter(1, idx, select_sorted.to(torch.int8)).bool() & valid

            for row in range(x.shape[0]):
                local = transfer_local[row].nonzero(as_tuple=True)[0]
                if local.numel() == 0:
                    continue
                selected_positions = positions[local]
                x[row, selected_positions] = x0_local[row, local]

            i += 1
            if (x[:, block_start:block_end] == MASK_ID).sum() == 0:
                break
    return x, nfe


def run_one(
    fn,
    model,
    prompt_ids: torch.Tensor,
    *,
    device: torch.device,
    steps: int,
    threshold: float | None,
) -> tuple[torch.Tensor, float, int, float]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    t0 = time.perf_counter()
    out, nfe = fn(
        model,
        prompt_ids,
        steps=steps,
        gen_length=GEN_LENGTH,
        block_length=BLOCK_LENGTH,
        temperature=0.0,
        remasking="low_confidence",
        threshold=threshold,
    )
    sync(device)
    elapsed = time.perf_counter() - t0
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else float("nan")
    )
    return out[:, prompt_ids.shape[1] : prompt_ids.shape[1] + GEN_LENGTH].detach().cpu(), elapsed, int(nfe), peak_gb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="gsm8k", choices=["mmlu", "gsm8k", "math500"])
    parser.add_argument("--n_prompts", type=int, default=64)
    parser.add_argument("--max_rows", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--include_parallel", action="store_true")
    parser.add_argument("--parallel_threshold", type=float, default=0.9)
    parser.add_argument("--tag", default="fastdllm_llada_compare")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    examples = load_task(args.task, args.n_prompts)[: args.max_rows]
    steps_per_block = GEN_STEPS // (GEN_LENGTH // BLOCK_LENGTH)
    methods: list[tuple[str, object, int, float | None]] = [
        ("vanilla", generate, GEN_STEPS, None),
        ("active_output", generate_active_output, GEN_STEPS, None),
        ("prefix_cache", generate_with_prefix_cache, GEN_STEPS, None),
        ("dual_cache", generate_with_dual_cache, GEN_STEPS, None),
    ]
    if args.include_parallel:
        methods.extend(
            [
                ("parallel_t", generate, steps_per_block, args.parallel_threshold),
                ("active_output_parallel_t", generate_active_output, steps_per_block, args.parallel_threshold),
                ("prefix_cache_parallel_t", generate_with_prefix_cache, steps_per_block, args.parallel_threshold),
                ("dual_cache_parallel_t", generate_with_dual_cache, steps_per_block, args.parallel_threshold),
            ]
        )

    rows = []
    for idx, ex in enumerate(examples):
        prompt = build_prompt(tokenizer, args.task, ex).to(device)
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)

        outputs: dict[str, torch.Tensor] = {}
        texts: dict[str, str] = {}
        for method, fn, method_steps, threshold in methods:
            try:
                gen_ids, elapsed, nfe, peak_gb = run_one(
                    fn,
                    model,
                    prompt,
                    device=device,
                    steps=method_steps,
                    threshold=threshold,
                )
                text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
                outputs[method] = gen_ids
                texts[method] = text
                rows.append(
                    {
                        "task": args.task,
                        "prompt_idx": idx,
                        "method": method,
                        "steps_arg": method_steps,
                        "threshold": threshold,
                        "elapsed_sec": elapsed,
                        "nfe": nfe,
                        "peak_gb": peak_gb,
                        "text": text,
                        "correct": score(args.task, text, ex),
                        "error": "",
                    }
                )
                print(
                    f"idx={idx} method={method} sec={elapsed:.3f} nfe={nfe} "
                    f"peak={peak_gb:.2f} correct={rows[-1]['correct']}"
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "task": args.task,
                        "prompt_idx": idx,
                        "method": method,
                        "elapsed_sec": float("nan"),
                        "nfe": -1,
                        "peak_gb": float("nan"),
                        "text": "",
                        "correct": False,
                        "error": repr(exc),
                    }
                )
                print(f"idx={idx} method={method} error={exc!r}")

        base = outputs.get("vanilla")
        for method, _, _, _ in methods:
            if method == "vanilla":
                continue
            if base is None or method not in outputs:
                continue
            same_ids = bool(torch.equal(base, outputs[method]))
            same_text = texts.get("vanilla", "") == texts.get(method, "")
            rows.append(
                {
                    "task": args.task,
                    "prompt_idx": idx,
                    "method": f"compare_vanilla_vs_{method}",
                    "elapsed_sec": float("nan"),
                    "nfe": -1,
                    "peak_gb": float("nan"),
                    "text": "",
                    "correct": same_ids and same_text,
                    "same_ids": same_ids,
                    "same_text": same_text,
                    "error": "",
                }
            )
            print(f"idx={idx} compare {method}: same_ids={same_ids} same_text={same_text}")

    df = pd.DataFrame(rows)
    rows_path = OUT_DIR / f"{args.tag}_rows.csv"
    summary_path = OUT_DIR / f"{args.tag}_summary.csv"
    df.to_csv(rows_path, index=False)

    method_names = [m[0] for m in methods]
    gen = df[df["method"].isin(method_names)].copy()
    ok = gen[gen["error"].fillna("") == ""]
    summary = (
        ok.groupby("method", as_index=False)
        .agg(
            n_rows=("prompt_idx", "count"),
            mean_sec=("elapsed_sec", "mean"),
            mean_nfe=("nfe", "mean"),
            mean_peak_gb=("peak_gb", "mean"),
            acc=("correct", "mean"),
        )
        .sort_values("method")
    )
    if not summary.empty and (summary["method"] == "vanilla").any():
        base_sec = float(summary.loc[summary["method"] == "vanilla", "mean_sec"].iloc[0])
        summary["speedup_vs_vanilla"] = base_sec / summary["mean_sec"]

    cmp_rows = df[df["method"].str.startswith("compare_", na=False)]
    if not cmp_rows.empty:
        cmp_summary = (
            cmp_rows.groupby("method", as_index=False)
            .agg(n_rows=("prompt_idx", "count"), same_ids=("same_ids", "mean"), same_text=("same_text", "mean"))
        )
        for _, row in cmp_summary.iterrows():
            print(row.to_dict())

    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"saved {rows_path}")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
