"""Build serving-oriented length traces from local datasets.

The paper's main experiments need workloads with realistic request-length
heterogeneity. This script converts local ShareGPT, Wikipedia-style, HotpotQA,
and article data into compact CSV traces consumed by the dlmserve benchmark.
It is intentionally offline-first: if LongBench/L-Eval are later downloaded,
their json/jsonl files can be passed through the generic text-file path.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable

import ijson
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "cache-sparse-attention" / "results"
DEFAULT_TOKENIZER = ROOT / "model_cache" / "LLaDA-8B-Instruct"
DEFAULT_SHAREGPT = ROOT / "cache-sparse-attention" / "data" / "ShareGPT_V3_unfiltered_cleaned_split.json"
DEFAULT_WIKITEXT = ROOT / "cache-sparse-attention" / "data" / "wikitext2_128x2048.jsonl"
DEFAULT_HOTPOT = ROOT / "cache-sparse-attention" / "data" / "hotpotqa_100.jsonl"
DEFAULT_ARTICLES = ROOT / "cache-sparse-attention" / "data" / "articles.jsonl"


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def row(source: str, uid: str, prompt_len: int, gen_len: int, origin: str) -> dict[str, object]:
    return {
        "source": source,
        "request_uid": uid,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "total_len": prompt_len + gen_len,
        "origin_source": origin,
    }


def iter_sharegpt_prompts(path: Path, max_conversations: int) -> Iterable[tuple[str, str, str]]:
    with path.open("rb") as f:
        for conv_idx, obj in enumerate(ijson.items(f, "item")):
            if conv_idx >= max_conversations:
                break
            context: list[str] = []
            turn_idx = 0
            for msg in obj.get("conversations") or []:
                role = msg.get("from", "")
                value = msg.get("value", "")
                if not isinstance(value, str) or not value.strip():
                    continue
                if role == "human":
                    single = value
                    multiturn = "\n".join(context + [f"Human: {value}\nAssistant:"])
                    yield f"sg_{conv_idx}_{turn_idx}_single", "sharegpt_single_turn", single
                    yield f"sg_{conv_idx}_{turn_idx}_multi", "sharegpt_multiturn_accum", multiturn
                    context.append(f"Human: {value}")
                    turn_idx += 1
                elif role == "gpt":
                    context.append(f"Assistant: {value}")


def iter_jsonl_text(path: Path, *, source: str, max_rows: int) -> Iterable[tuple[str, str, str]]:
    with path.open() as f:
        for idx, line in enumerate(f):
            if idx >= max_rows:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            texts: list[str] = []
            if isinstance(obj, str):
                texts.append(obj)
            elif isinstance(obj, dict):
                for key in ("prompt", "text", "article", "input", "question", "content", "context"):
                    val = obj.get(key)
                    if isinstance(val, str):
                        texts.append(val)
                    elif isinstance(val, list):
                        texts.extend(str(x) for x in val if isinstance(x, str))
                if not texts and len(obj) == 1:
                    only = next(iter(obj.values()))
                    if isinstance(only, list):
                        for item in only:
                            if isinstance(item, dict):
                                obs = item.get("observation")
                                question = item.get("question")
                                if isinstance(question, str):
                                    texts.append(question)
                                if isinstance(obs, str):
                                    texts.append(obs)
                            elif isinstance(item, str):
                                texts.append(item)
            text = "\n".join(t for t in texts if t.strip())
            if text.strip():
                yield f"{source}_{idx}", source, text


def build_source_rows(tokenizer, items: Iterable[tuple[str, str, str]], gen_len: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for uid, source, text in items:
        rows.append(row(source, uid, token_len(tokenizer, text), gen_len, source))
    return rows


def choose(
    rows: list[dict[str, object]],
    *,
    source: str,
    n: int,
    seed: int,
    min_total: int = 0,
    max_total: int = 100000,
) -> list[dict[str, object]]:
    candidates = [
        r
        for r in rows
        if r["source"] == source and min_total <= int(r["total_len"]) <= max_total
    ]
    if len(candidates) < n:
        raise ValueError(f"not enough {source} rows: {len(candidates)} < {n}")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


def relabel(rows: list[dict[str, object]], source: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for idx, item in enumerate(rows):
        copied = dict(item)
        copied["source"] = source
        copied["request_uid"] = f"{source}_{idx:04d}_{item['request_uid']}"
        copied["origin_source"] = item["source"]
        out.append(copied)
    return out


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_source: dict[str, list[int]] = {}
    for item in rows:
        by_source.setdefault(str(item["source"]), []).append(int(item["total_len"]))
    summaries: list[dict[str, object]] = []
    for source, vals in sorted(by_source.items()):
        vals = sorted(vals)
        def pct(q: float) -> int:
            return vals[min(len(vals) - 1, int(q * (len(vals) - 1)))]
        summaries.append(
            {
                "source": source,
                "n": len(vals),
                "min": vals[0],
                "p50": pct(0.50),
                "p90": pct(0.90),
                "p95": pct(0.95),
                "p99": pct(0.99),
                "max": vals[-1],
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--sharegpt", type=Path, default=DEFAULT_SHAREGPT)
    p.add_argument("--wikitext", type=Path, default=DEFAULT_WIKITEXT)
    p.add_argument("--hotpot", type=Path, default=DEFAULT_HOTPOT)
    p.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    p.add_argument("--max_conversations", type=int, default=5000)
    p.add_argument("--max_jsonl_rows", type=int, default=1000)
    p.add_argument("--gen_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out_prefix", default="serving_workloads_v1")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    base_rows: list[dict[str, object]] = []
    if args.sharegpt.exists():
        base_rows.extend(
            build_source_rows(
                tokenizer,
                iter_sharegpt_prompts(args.sharegpt, args.max_conversations),
                args.gen_len,
            )
        )
    for path, source in (
        (args.wikitext, "wikitext2_2048"),
        (args.hotpot, "hotpotqa_wikipedia"),
        (args.articles, "llm_judge_articles"),
    ):
        if path.exists():
            base_rows.extend(
                build_source_rows(
                    tokenizer,
                    iter_jsonl_text(path, source=source, max_rows=args.max_jsonl_rows),
                    args.gen_len,
                )
            )

    workloads: list[dict[str, object]] = []
    workloads.extend(relabel(choose(base_rows, source="sharegpt_multiturn_accum", n=96, seed=args.seed, max_total=2048), "workload_sharegpt_multiturn"))
    workloads.extend(relabel(choose(base_rows, source="hotpotqa_wikipedia", n=96, seed=args.seed, max_total=2048), "workload_hotpotqa_wikipedia"))
    workloads.extend(relabel(choose(base_rows, source="wikitext2_2048", n=96, seed=args.seed, max_total=4096), "workload_wikitext2"))

    short_chat = choose(base_rows, source="sharegpt_single_turn", n=32, seed=args.seed, max_total=512)
    long_chat = choose(base_rows, source="sharegpt_multiturn_accum", n=32, seed=args.seed + 1, min_total=1024, max_total=2048)
    hotpot = choose(base_rows, source="hotpotqa_wikipedia", n=32, seed=args.seed + 2, max_total=2048)
    multitask = short_chat + long_chat + hotpot
    random.Random(args.seed).shuffle(multitask)
    workloads.extend(relabel(multitask, "workload_multitask_mix"))

    rows_out = RESULTS / f"{args.out_prefix}_rows.csv"
    summary_out = RESULTS / f"{args.out_prefix}_summary.csv"
    write_csv(rows_out, workloads)
    write_csv(summary_out, summarize(workloads))
    print(f"wrote {rows_out} ({len(workloads)} rows)")
    print(f"wrote {summary_out}")


if __name__ == "__main__":
    main()
