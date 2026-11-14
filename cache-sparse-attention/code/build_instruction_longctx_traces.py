"""Build instruction and long-context serving traces.

The benchmark runner consumes compact CSV rows with tokenized prompt lengths.
This builder keeps dataset handling separate from the dlmserve harness, so a
new workload can be added without changing the serving code.

The script is offline-first. HuggingFace datasets are loaded with
``reuse_dataset_if_exists`` when a local cache is available; long-context
benchmarks can also be supplied as local JSON/JSONL exports.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "cache-sparse-attention" / "results"
DEFAULT_TOKENIZER = ROOT / "model_cache" / "LLaDA-8B-Instruct"
DEFAULT_ALPACA_ARROW = ROOT / "cache-sparse-attention" / "data" / "alpaca-train.arrow"


def token_len(tokenizer: Any, text: str) -> int:
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


def normalize_dataset(obj: Dataset | DatasetDict, split: str) -> Dataset:
    if isinstance(obj, DatasetDict):
        if split in obj:
            return obj[split]
        return obj[next(iter(obj.keys()))]
    return obj


def format_instruction(example: dict[str, Any]) -> str:
    instruction = str(example.get("instruction") or "").strip()
    inp = str(example.get("input") or "").strip()
    if inp:
        return f"Instruction:\n{instruction}\n\nInput:\n{inp}\n\nResponse:"
    return f"Instruction:\n{instruction}\n\nResponse:"


def iter_alpaca(max_rows: int, arrow_path: Path) -> Iterable[tuple[str, str, str]]:
    if arrow_path.exists():
        ds = Dataset.from_file(str(arrow_path))
    else:
        ds = normalize_dataset(
            load_dataset("tatsu-lab/alpaca", split="train", download_mode="reuse_dataset_if_exists"),
            "train",
        )
    for idx, example in enumerate(ds):
        if idx >= max_rows:
            break
        text = format_instruction(dict(example))
        if text.strip():
            yield f"alpaca_{idx}", "alpaca_instruction", text


def format_openorca(example: dict[str, Any]) -> str:
    system = str(example.get("system_prompt") or example.get("system") or "").strip()
    question = str(example.get("question") or example.get("prompt") or "").strip()
    if system:
        return f"System:\n{system}\n\nUser:\n{question}\n\nAssistant:"
    return f"User:\n{question}\n\nAssistant:"


def iter_openorca(max_rows: int) -> Iterable[tuple[str, str, str]]:
    ds = normalize_dataset(
        load_dataset("Open-Orca/OpenOrca", split="train", download_mode="reuse_dataset_if_exists"),
        "train",
    )
    for idx, example in enumerate(ds):
        if idx >= max_rows:
            break
        text = format_openorca(dict(example))
        if text.strip():
            yield f"openorca_{idx}", "openorca_instruction", text


def format_flan(example: dict[str, Any]) -> str:
    inputs = str(example.get("inputs") or example.get("input") or example.get("source") or "").strip()
    task = str(example.get("task_name") or example.get("template_type") or "").strip()
    if task:
        return f"Task: {task}\n\nInput:\n{inputs}\n\nResponse:"
    return f"Input:\n{inputs}\n\nResponse:"


def iter_flan(max_rows: int) -> Iterable[tuple[str, str, str]]:
    ds = normalize_dataset(
        load_dataset("Muennighoff/flan", split="train", download_mode="reuse_dataset_if_exists"),
        "train",
    )
    for idx, example in enumerate(ds):
        if idx >= max_rows:
            break
        text = format_flan(dict(example))
        if text.strip():
            yield f"flan_{idx}", "flan_instruction", text


def flatten_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(flatten_text(item))
        return texts
    if isinstance(value, dict):
        texts = []
        for key in (
            "input",
            "inputs",
            "context",
            "question",
            "article",
            "document",
            "text",
            "content",
            "prompt",
        ):
            if key in value:
                texts.extend(flatten_text(value[key]))
        return texts
    return []


def iter_jsonl(path: Path, source: str, max_rows: int) -> Iterable[tuple[str, str, str]]:
    with path.open() as f:
        for idx, line in enumerate(f):
            if idx >= max_rows:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = "\n".join(t for t in flatten_text(obj) if t.strip())
            if text.strip():
                yield f"{source}_{idx}", source, text


def iter_json(path: Path, source: str, max_rows: int) -> Iterable[tuple[str, str, str]]:
    with path.open() as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            records = obj["data"]
        elif "examples" in obj and isinstance(obj["examples"], list):
            records = obj["examples"]
        else:
            records = [obj]
    elif isinstance(obj, list):
        records = obj
    else:
        records = []
    for idx, record in enumerate(records[:max_rows]):
        text = "\n".join(t for t in flatten_text(record) if t.strip())
        if text.strip():
            yield f"{source}_{idx}", source, text


def iter_zip(path: Path, source: str, max_rows: int) -> Iterable[tuple[str, str, str]]:
    emitted = 0
    with zipfile.ZipFile(path) as zf:
        def sort_key(name: str) -> tuple[int, str]:
            match = re.search(r"_(\d+)k\.", name)
            return (int(match.group(1)) if match else 10**9, name)

        names = sorted(
            (
                name for name in zf.namelist()
                if name.endswith(".jsonl") or name.endswith(".json")
            ),
            key=sort_key,
        )
        for name in names:
            with zf.open(name) as raw:
                if name.endswith(".jsonl"):
                    for line_idx, raw_line in enumerate(raw):
                        if emitted >= max_rows:
                            return
                        try:
                            obj = json.loads(raw_line.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        text = "\n".join(t for t in flatten_text(obj) if t.strip())
                        if text.strip():
                            yield f"{source}_{Path(name).stem}_{line_idx}", source, text
                            emitted += 1
                else:
                    try:
                        obj = json.loads(raw.read().decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    for uid, _, text in iter_records(obj, source, max_rows - emitted):
                        if emitted >= max_rows:
                            return
                        yield f"{source}_{Path(name).stem}_{uid}", source, text
                        emitted += 1


def iter_records(obj: Any, source: str, max_rows: int) -> Iterable[tuple[str, str, str]]:
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            records = obj["data"]
        elif "examples" in obj and isinstance(obj["examples"], list):
            records = obj["examples"]
        else:
            records = [obj]
    elif isinstance(obj, list):
        records = obj
    else:
        records = []
    for idx, record in enumerate(records[:max_rows]):
        text = "\n".join(t for t in flatten_text(record) if t.strip())
        if text.strip():
            yield f"{idx}", source, text


def iter_parquet(path: Path, source: str, max_rows: int) -> Iterable[tuple[str, str, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for local parquet inputs") from exc

    emitted = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=min(1024, max_rows)):
        columns = batch.to_pydict()
        names = list(columns)
        for row_idx in range(batch.num_rows):
            if emitted >= max_rows:
                return
            record = {name: columns[name][row_idx] for name in names}
            if source.startswith("openorca") or "openorca" in source:
                text = format_openorca(record)
            else:
                text = "\n".join(t for t in flatten_text(record) if t.strip())
            if text.strip():
                yield f"{source}_{emitted}", source, text
                emitted += 1


def parse_source_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SOURCE=PATH")
    source, path = value.split("=", 1)
    source = source.strip()
    if not source:
        raise argparse.ArgumentTypeError("source must be non-empty")
    return source, Path(path)


def build_or_skip(
    tokenizer: Any,
    items: Iterable[tuple[str, str, str]],
    *,
    out_source: str,
    n: int,
    seed: int,
    gen_len: int,
    min_total: int,
    max_total: int,
    allow_partial: bool,
) -> list[dict[str, object]]:
    try:
        return build_rows(
            tokenizer,
            items,
            out_source=out_source,
            n=n,
            seed=seed,
            gen_len=gen_len,
            min_total=min_total,
            max_total=max_total,
        )
    except Exception as exc:
        if not allow_partial:
            raise
        print(f"skip {out_source}: {exc}")
        return []


def build_rows(
    tokenizer: Any,
    items: Iterable[tuple[str, str, str]],
    *,
    out_source: str,
    n: int,
    seed: int,
    gen_len: int,
    min_total: int,
    max_total: int,
) -> list[dict[str, object]]:
    rows = []
    for uid, origin, text in items:
        prompt_len = token_len(tokenizer, text)
        total = prompt_len + gen_len
        if min_total <= total <= max_total:
            rows.append(row(out_source, uid, prompt_len, gen_len, origin))
    if len(rows) < n:
        raise ValueError(f"not enough rows for {out_source}: {len(rows)} < {n}")
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_source: dict[str, list[int]] = {}
    for item in rows:
        by_source.setdefault(str(item["source"]), []).append(int(item["total_len"]))
    out = []
    for source, vals in sorted(by_source.items()):
        vals = sorted(vals)

        def pct(q: float) -> int:
            return vals[min(len(vals) - 1, int(q * (len(vals) - 1)))]

        out.append(
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
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--max_rows", type=int, default=5000)
    parser.add_argument("--gen_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min_total", type=int, default=0)
    parser.add_argument("--max_total", type=int, default=4096)
    parser.add_argument("--include_alpaca", action="store_true")
    parser.add_argument("--include_openorca", action="store_true")
    parser.add_argument("--include_flan", action="store_true")
    parser.add_argument("--alpaca_arrow", type=Path, default=DEFAULT_ALPACA_ARROW)
    parser.add_argument("--alpaca_out_source", default="workload_alpaca_instruction")
    parser.add_argument("--longbench_jsonl", type=Path)
    parser.add_argument("--leval_jsonl", type=Path)
    parser.add_argument("--lveval_jsonl", type=Path)
    parser.add_argument("--jsonl_input", action="append", type=parse_source_path, default=[])
    parser.add_argument("--json_input", action="append", type=parse_source_path, default=[])
    parser.add_argument("--zip_input", action="append", type=parse_source_path, default=[])
    parser.add_argument("--parquet_input", action="append", type=parse_source_path, default=[])
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--out_prefix", default="instruction_longctx_workloads_v1")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows: list[dict[str, object]] = []
    if args.include_alpaca:
        rows.extend(
            build_or_skip(
                tokenizer,
                iter_alpaca(args.max_rows, args.alpaca_arrow),
                out_source=args.alpaca_out_source,
                n=args.n,
                seed=args.seed,
                gen_len=args.gen_len,
                min_total=args.min_total,
                max_total=args.max_total,
                allow_partial=args.allow_partial,
            )
        )
    if args.include_openorca:
        rows.extend(
            build_or_skip(
                tokenizer,
                iter_openorca(args.max_rows),
                out_source="workload_openorca_instruction",
                n=args.n,
                seed=args.seed,
                gen_len=args.gen_len,
                min_total=args.min_total,
                max_total=args.max_total,
                allow_partial=args.allow_partial,
            )
        )
    if args.include_flan:
        rows.extend(
            build_or_skip(
                tokenizer,
                iter_flan(args.max_rows),
                out_source="workload_flan_instruction",
                n=args.n,
                seed=args.seed,
                gen_len=args.gen_len,
                min_total=args.min_total,
                max_total=args.max_total,
                allow_partial=args.allow_partial,
            )
        )
    for path, source in (
        (args.longbench_jsonl, "workload_longbench_subset"),
        (args.leval_jsonl, "workload_leval_subset"),
        (args.lveval_jsonl, "workload_lveval_subset"),
    ):
        if path is None:
            continue
        rows.extend(
            build_or_skip(
                tokenizer,
                iter_jsonl(path, source, args.max_rows),
                out_source=source,
                n=args.n,
                seed=args.seed,
                gen_len=args.gen_len,
                min_total=args.min_total,
                max_total=args.max_total,
                allow_partial=args.allow_partial,
            )
        )

    for source, path in args.jsonl_input:
        rows.extend(
            build_or_skip(
                tokenizer,
                iter_jsonl(path, source, args.max_rows),
                out_source=source,
                n=args.n,
                seed=args.seed,
                gen_len=args.gen_len,
                min_total=args.min_total,
                max_total=args.max_total,
                allow_partial=args.allow_partial,
            )
        )
    for source, path in args.json_input:
        rows.extend(
            build_or_skip(
                tokenizer,
                iter_json(path, source, args.max_rows),
                out_source=source,
                n=args.n,
                seed=args.seed,
                gen_len=args.gen_len,
                min_total=args.min_total,
                max_total=args.max_total,
                allow_partial=args.allow_partial,
            )
        )
    for source, path in args.zip_input:
        rows.extend(
            build_or_skip(
                tokenizer,
                iter_zip(path, source, args.max_rows),
                out_source=source,
                n=args.n,
                seed=args.seed,
                gen_len=args.gen_len,
                min_total=args.min_total,
                max_total=args.max_total,
                allow_partial=args.allow_partial,
            )
        )
    for source, path in args.parquet_input:
        rows.extend(
            build_or_skip(
                tokenizer,
                iter_parquet(path, source, args.max_rows),
                out_source=source,
                n=args.n,
                seed=args.seed,
                gen_len=args.gen_len,
                min_total=args.min_total,
                max_total=args.max_total,
                allow_partial=args.allow_partial,
            )
        )

    if not rows:
        raise SystemExit("no workload requested")
    rows_out = RESULTS / f"{args.out_prefix}_rows.csv"
    summary_out = RESULTS / f"{args.out_prefix}_summary.csv"
    write_csv(rows_out, rows)
    write_csv(summary_out, summarize(rows))
    print(f"wrote {rows_out} ({len(rows)} rows)")
    print(f"wrote {summary_out}")


if __name__ == "__main__":
    main()
