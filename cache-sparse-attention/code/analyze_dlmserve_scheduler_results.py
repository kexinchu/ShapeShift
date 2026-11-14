"""Analyze dlmserve scheduler benchmark outputs.

This utility supports three kinds of logs:

1. all-at-once summaries, where each row is one policy/mode result.
2. per-row/online traces, where each row is one request.
3. scheduler traces, where each row is one scheduling step.

The script emits:
- compact summary CSV (throughput, latency quantiles, fairness, SLO, wait),
- per-length-bucket slowdown CSV (FIFO-normalized),
- seed-stat CSV (mean/std/95% CI over seeds),
- compact LaTeX snippets suitable for tables.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "cache-sparse-attention" / "results"


DEFAULT_INPUTS = [
    RESULTS / "dlmserve_scheduler_aging_multiturn_n24_b16_s8_g128_v1.csv",
    RESULTS / "dlmserve_scheduler_aging_multiturn_n24_b8_s8_g128_v1.csv",
    RESULTS / "dlmserve_scheduler_aging_multiturn_n24_b4_s8_g128_v1.csv",
]

SUMMARY_OUT_DEFAULT = RESULTS / "dlmserve_scheduler_fairness_summary_v1.csv"
BUCKET_OUT_DEFAULT = RESULTS / "dlmserve_scheduler_fairness_bucket_slowdown_v1.csv"
LATEX_OUT_DEFAULT = RESULTS / "dlmserve_scheduler_latex_tables_v1.tex"
SEED_STATS_OUT_DEFAULT = RESULTS / "dlmserve_scheduler_seed_stats_v1.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_csv",
        action="append",
        default=None,
        help="One or more explicit input CSV paths.",
    )
    p.add_argument(
        "--glob",
        default=None,
        help="Glob under cache-sparse-attention/results, e.g. 'dlmserve_*_rows.csv'.",
    )
    p.add_argument("--summary_out", default=str(SUMMARY_OUT_DEFAULT))
    p.add_argument("--bucket_out", default=str(BUCKET_OUT_DEFAULT))
    p.add_argument("--latex_out", default=str(LATEX_OUT_DEFAULT))
    p.add_argument("--seed_stats_out", default=str(SEED_STATS_OUT_DEFAULT))
    p.add_argument(
        "--fallback_wall_latency",
        action="store_true",
        help="For summary rows without per-request latency, approximate latency metrics by wall_s / n_requests.",
    )
    p.add_argument(
        "--bucket_bounds",
        default="0,256,512,1024,2048,4096,10000",
        help="Comma-separated prompt length bucket bounds (right-open).",
    )
    return p.parse_args()


def canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.lower().strip() for c in df.columns})


def first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def first_value(df: pd.DataFrame, names: list[str]) -> Any | None:
    for name in names:
        if name not in df.columns:
            continue
        s = df[name]
        if s.empty:
            continue
        first = s.dropna().iloc[0] if hasattr(s, "dropna") else None
        if first is None or (isinstance(first, float) and np.isnan(first)):
            continue
        return first
    return None


def to_numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def first_scalar(df: pd.DataFrame, names: list[str]) -> float | None:
    col = first_present(df, names)
    if col is None:
        return None
    s = to_numeric(df, col).dropna()
    if s.empty:
        return None
    return _finite_float(s.iloc[0])


def _finite_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _clean_numeric_series(df: pd.DataFrame, names: list[str], default: pd.Series | None = None) -> pd.Series:
    col = first_present(df, names)
    if not col:
        return pd.Series(dtype=float) if default is None else default
    return to_numeric(df, col)


def _wait_quantiles(s: pd.Series) -> dict[str, float | None]:
    s = s.dropna()
    if s.empty:
        return {
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


def infer_from_filename(input_file: str, key: str) -> float | None:
    stem = Path(input_file).name
    patterns = {
        "seed": [r"_seed(\d+)(?:_|\.|$)"],
        "max_batch": [r"_b(\d+)(?:_|\.|$)"],
        "steps": [r"_s(\d+)(?:_|\.|$)", r"steps(\d+)"],
        "gen_length": [r"_g(\d+)(?:_|\.|$)", r"gen(?:eration)?_?length[_-]?(\d+)"],
        "arrival_gap_ms": [r"_gap(\d+)(?:_|\.|$)"],
    }
    for pattern in patterns.get(key, []):
        m = re.search(pattern, stem)
        if m:
            return float(m.group(1))
    return None


def parse_bucket_bounds(raw: str) -> list[float]:
    edges = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not edges:
        return [0.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 10000.0]
    edges = sorted(set(edges))
    if edges[0] != 0.0:
        edges = [0.0] + edges
    if edges[-1] <= 0:
        edges.append(10000.0)
    return edges


def make_bucket_labels(edges: list[float]) -> list[str]:
    return [f"{int(lo)}-{int(hi)}" for lo, hi in zip(edges[:-1], edges[1:])]


def detect_dataset_type(df: pd.DataFrame, policy_col: str | None) -> str:
    if policy_col is None:
        return "summary"

    has_step_cols = any(col in df.columns for col in ["step_idx", "batch_size", "seed_len"])
    has_trace_wait_cols = any(col in df.columns for col in [
        "oldest_start_wait_ms",
        "oldest_interservice_wait_ms",
        "oldest_wait_ms",
    ])
    has_request_level = any(
        col in df.columns
        for col in [
            "latency_ms",
            "latency_s",
            "start_ms",
            "end_ms",
            "arrival_ms",
            "request_id",
            "request_uid",
            "req_id",
            "idx",
            "arrival_latency_ms",
        ]
    )
    has_summary_cols = any(
        c in df.columns
        for c in [
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "max_latency_ms",
            "req_per_s",
            "tok_per_s",
            "speedup_vs_fifo_req_s",
        ]
    )
    if has_step_cols and has_trace_wait_cols and has_summary_cols is False:
        return "scheduler_trace"
    if not has_request_level and has_summary_cols:
        return "summary"
    if has_request_level and len(df) > 1:
        return "per_row"
    # Ambiguous small files (typically 4-mode summary), prefer summary.
    return "summary"


def infer_latency_ms(df: pd.DataFrame) -> pd.Series:
    arrival_offset = first_present(df, ["arrival_latency_ms", "scheduled_arrival_offset_ms"])
    end_offset = first_present(df, ["end_latency_ms", "completion_latency_ms"])
    if arrival_offset and end_offset:
        return to_numeric(df, end_offset) - to_numeric(df, arrival_offset)

    col = first_present(
        df,
        [
            "latency_ms",
            "response_latency_ms",
            "completion_latency_ms",
            "response_latency_ms",
            "response_ms",
            "completion_ms",
            "elapsed_ms",
            "duration_ms",
        ],
    )
    if col:
        return to_numeric(df, col)

    col = first_present(
        df,
        [
            "latency_s",
            "response_latency_s",
            "response_s",
            "completion_s",
            "elapsed_s",
            "duration_s",
            "end_s",
        ],
    )
    if col:
        return to_numeric(df, col) * 1000.0

    start = first_present(df, ["start_ms", "start_time_ms", "dispatch_ms", "begin_ms"])
    end = first_present(df, ["end_ms", "end_time_ms", "finish_ms", "response_end_ms"])
    if start and end:
        return to_numeric(df, end) - to_numeric(df, start)

    start = first_present(df, ["start_s", "start_time_s", "dispatch_s", "begin_s"])
    end = first_present(df, ["end_s", "end_time_s", "finish_s", "response_end_s"])
    if start and end:
        return (to_numeric(df, end) - to_numeric(df, start)) * 1000.0

    arrival = first_present(df, ["arrival_ms", "arrival_time_ms", "arrival_latency_ms"])
    completion = first_present(df, ["completion_ms", "completion_time_ms", "finish_ms"])
    if arrival and completion:
        return to_numeric(df, completion) - to_numeric(df, arrival)

    return pd.Series(dtype=float)


def infer_start_wait_ms(df: pd.DataFrame) -> pd.Series:
    col = first_present(
        df,
        [
            "oldest_start_wait_ms",
            "start_latency_ms",
            "queue_wait_ms",
            "wait_ms",
            "ttft_ms",
            "time_to_first_token_ms",
            "oldest_wait_ms",
        ],
    )
    if col:
        return to_numeric(df, col)

    start_offset = first_present(df, ["start_offset_ms", "start_time_offset_ms"])
    arrival_offset = first_present(df, ["arrival_latency_ms", "scheduled_arrival_offset_ms"])
    if start_offset and arrival_offset:
        return to_numeric(df, start_offset) - to_numeric(df, arrival_offset)

    start_abs = first_present(df, ["start_ms", "start_time_ms", "dispatch_ms", "begin_ms"])
    arrival_abs = first_present(df, ["arrival_ms", "arrival_time_ms"])
    if start_abs and arrival_abs:
        return to_numeric(df, start_abs) - to_numeric(df, arrival_abs)

    return pd.Series(dtype=float)


def infer_interservice_wait_ms(df: pd.DataFrame) -> pd.Series:
    return _clean_numeric_series(
        df,
        [
            "oldest_interservice_wait_ms",
            "interservice_wait_ms",
            "inter_step_wait_ms",
        ],
    )


def infer_wall_s(df: pd.DataFrame, lat_ms: pd.Series) -> float | None:
    arrival_latency = first_present(df, ["arrival_latency_ms", "scheduled_arrival_offset_ms"])
    end_latency = first_present(df, ["end_latency_ms", "latency_ms", "completion_latency_ms"])
    if arrival_latency and end_latency:
        arrivals = to_numeric(df, arrival_latency).dropna()
        ends = to_numeric(df, end_latency).dropna()
        if not arrivals.empty and not ends.empty:
            return (float(ends.max()) - float(arrivals.min())) / 1000.0

    col = first_present(
        df,
        ["wall_s", "makespan_s", "elapsed_s", "duration_s", "runtime_s"],
    )
    if col is None:
        col = first_present(df, ["wall_ms", "makespan_ms", "elapsed_ms", "duration_ms", "runtime_ms"])
        if col is not None:
            val = first_scalar(df, [col])
            if val is not None and val > 0:
                return val / 1000.0
    else:
        val = first_scalar(df, [col])
        if val is not None and val > 0:
            return val

    arrival = first_present(df, ["arrival_ms", "arrival_time_ms", "arrival_latency_ms"])
    completion = first_present(df, ["completion_ms", "completion_time_ms", "finish_ms", "completed_ms"])
    if arrival and completion:
        a = to_numeric(df, arrival)
        c = to_numeric(df, completion)
        if not a.empty and not c.empty:
            wall = c.max() - a.min()
            if wall > 0:
                return float(wall / 1000.0)

    if not lat_ms.empty:
        lat = lat_ms.dropna()
        if not lat.empty and lat.max() > 0:
            return float(lat.max() / 1000.0)
    return None


def add_bucket(df: pd.DataFrame, bucket_edges: list[float]) -> pd.Series:
    prompt = first_present(
        df,
        ["prompt_len", "prompt_length", "prompt_tokens", "total_len", "total_length", "seq_len"],
    )
    if not prompt:
        return pd.Series(["unknown"] * len(df), index=df.index)
    vals = to_numeric(df, prompt)
    if vals.dropna().empty:
        return pd.Series(["unknown"] * len(df), index=df.index)
    if vals.dropna().nunique(dropna=True) <= 1:
        return pd.Series(["single"] * len(df), index=df.index)
    try:
        return pd.cut(
            vals.fillna(vals.median()),
            bins=bucket_edges,
            labels=make_bucket_labels(bucket_edges),
            right=False,
            include_lowest=True,
        ).astype(str)
    except Exception:
        n = min(4, len(df))
        return pd.qcut(vals.rank(method="first"), q=n, labels=[f"q{i+1}" for i in range(n)]).astype(str)


def summarize_policy_from_per_row(
    file_df: pd.DataFrame,
    policy: str,
    policy_df: pd.DataFrame,
    input_file: str,
    fallback_wall_latency: bool,
) -> dict[str, Any]:
    lat_ms = infer_latency_ms(policy_df)
    lat = lat_ms.dropna()
    wall_s = infer_wall_s(policy_df, lat_ms)
    n_rows = len(policy_df)
    n_requests = first_scalar(policy_df, ["n_requests"]) or n_rows

    req_per_s = first_scalar(policy_df, ["req_per_s", "throughput"])
    if req_per_s is None and wall_s and n_rows > 0:
        req_per_s = n_rows / wall_s

    tok_per_s = first_scalar(policy_df, ["tok_per_s", "token_per_s", "throughput_tok"])
    if tok_per_s is None and wall_s and wall_s > 0:
        tokens = to_numeric(policy_df, first_present(policy_df, ["output_tokens", "completion_tokens", "generated_tokens", "n_output_tokens"]))
        if not tokens.empty and not tokens.dropna().empty:
            tok_per_s = float(tokens.fillna(0.0).sum() / wall_s)

    p50 = float(lat.quantile(0.50)) if len(lat) else None
    p95 = float(lat.quantile(0.95)) if len(lat) else None
    p99 = float(lat.quantile(0.99)) if len(lat) else None
    max_lat = float(lat.max()) if len(lat) else None
    latency_estimated = False
    if (
        fallback_wall_latency
        and wall_s is not None
        and wall_s > 0
        and n_rows > 0
        and (p50 is None or p95 is None or p99 is None or max_lat is None)
    ):
        est = wall_s * 1000.0 / n_rows
        p50 = p50 if p50 is not None else est
        p95 = p95 if p95 is not None else est
        p99 = p99 if p99 is not None else est
        max_lat = max_lat if max_lat is not None else est
        latency_estimated = True

    prompt_col = first_present(
        policy_df,
        ["prompt_len", "prompt_length", "prompt_tokens", "total_len", "total_length", "seq_len"],
    )
    prompt_vals = to_numeric(policy_df, prompt_col)
    mean_prompt_len = float(prompt_vals.mean()) if not prompt_vals.empty else None
    min_prompt_len = float(prompt_vals.min()) if not prompt_vals.empty else None
    max_prompt_len = float(prompt_vals.max()) if not prompt_vals.empty else None

    slo_col = first_present(policy_df, ["slo_ms", "shape_slo_ms", "slo"])
    slo_ms = first_scalar(policy_df, [slo_col] if slo_col else [])
    slo_violation = None
    if slo_ms is not None and not lat.empty:
        slo_violation = float((lat > slo_ms).mean())
    wait_ms = infer_start_wait_ms(policy_df)
    wait_ms_clean = wait_ms.dropna()
    inter_wait_ms = infer_interservice_wait_ms(policy_df)
    wait_stats = _wait_quantiles(wait_ms_clean)
    inter_wait_stats = _wait_quantiles(inter_wait_ms.dropna())
    wait_violation = None
    starvation_count = None
    if slo_ms is not None and len(wait_ms_clean):
        wait_violation = float((wait_ms_clean > slo_ms).mean())
        starvation_count = int((wait_ms_clean > 2.0 * slo_ms).sum())

    rec: dict[str, Any] = {
        "policy": str(policy),
        "input_file": input_file,
        "policy_source": "per_row",
        "n_requests": int(n_requests) if n_requests is not None else None,
        "n_rows": n_rows,
        "wall_s": wall_s,
        "req_per_s": req_per_s,
        "tok_per_s": tok_per_s,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "max_latency_ms": max_lat,
        "mean_prompt_len": mean_prompt_len,
        "min_prompt_len": min_prompt_len,
        "max_prompt_len": max_prompt_len,
        "latency_estimated": latency_estimated,
        "slo_ms": slo_ms,
        "slo_violation_rate": slo_violation,
        "p50_start_wait_ms": wait_stats["p50"],
        "p95_start_wait_ms": wait_stats["p95"],
        "p99_start_wait_ms": wait_stats["p99"],
        "max_start_wait_ms": wait_stats["max"],
        "p50_interservice_wait_ms": inter_wait_stats["p50"],
        "p95_interservice_wait_ms": inter_wait_stats["p95"],
        "p99_interservice_wait_ms": inter_wait_stats["p99"],
        "max_interservice_wait_ms": inter_wait_stats["max"],
        "wait_slo_violation_rate": wait_violation,
        "starvation_count_2x_slo": starvation_count,
        "jain_fairness": np.nan,
        "mean_slowdown_vs_fifo": np.nan,
        "speedup_vs_fifo_req_s": np.nan,
        "speedup_vs_fifo_tok_s": np.nan,
    }

    for key in [
        "source",
        "seed",
        "max_batch",
        "steps",
        "gen_length",
        "shape_slo_ms",
        "active_pool_factor",
        "arrival_gap_ms",
        "n_outputs",
    ]:
        val = first_scalar(policy_df, [key])
        if val is None:
            val = infer_from_filename(input_file, key)
        if val is not None:
            rec[key] = val if key in {"max_batch", "steps", "gen_length", "seed"} else val

    return rec


def summarize_all_at_once_row(file_row: pd.Series, input_file: str) -> dict[str, Any]:
    row = file_row.to_frame().T
    policy_val = first_value(row, ["policy", "mode"])
    policy = str(policy_val) if policy_val is not None else "unknown"

    wall_s = first_scalar(row, ["wall_s", "makespan_s", "elapsed_s", "duration_s", "runtime_s"])
    if wall_s is None:
        wall_ms = first_scalar(row, ["wall_ms", "makespan_ms", "elapsed_ms", "duration_ms", "runtime_ms"])
        wall_s = wall_ms / 1000.0 if wall_ms else None

    n_requests = first_scalar(row, ["n_requests", "n_outputs"])
    n_rows = int(first_scalar(row, ["n_outputs"]) or n_requests or 0)
    req_per_s = first_scalar(row, ["req_per_s"])
    if req_per_s is None and wall_s is not None and n_rows > 0:
        req_per_s = n_rows / wall_s

    tok_per_s = first_scalar(row, ["tok_per_s", "token_per_s"])

    p50 = first_scalar(row, ["p50_latency_ms"])
    p95 = first_scalar(row, ["p95_latency_ms"])
    p99 = first_scalar(row, ["p99_latency_ms", "p99_latency"])
    max_lat = first_scalar(row, ["max_latency_ms", "max_latency", "max_lat_ms"])

    latency_estimated = False
    if wall_s is not None and n_requests and (p50 is None or p95 is None or p99 is None or max_lat is None):
        est = wall_s * 1000.0 / max(1.0, n_requests)
        if p50 is None:
            p50 = est
        if p95 is None:
            p95 = est
        if p99 is None:
            p99 = est
        if max_lat is None:
            max_lat = est
        latency_estimated = True

    rec: dict[str, Any] = {
        "policy": policy,
        "input_file": input_file,
        "policy_source": "summary",
        "n_requests": int(n_requests) if n_requests is not None else None,
        "n_rows": n_rows,
        "wall_s": wall_s,
        "req_per_s": req_per_s,
        "tok_per_s": tok_per_s,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "max_latency_ms": max_lat,
        "mean_prompt_len": first_scalar(row, ["mean_prompt_len"]),
        "min_prompt_len": first_scalar(row, ["min_prompt_len"]),
        "max_prompt_len": first_scalar(row, ["max_prompt_len"]),
        "latency_estimated": latency_estimated,
        "slo_ms": first_scalar(row, ["shape_slo_ms", "slo_ms", "slo"]),
        "slo_violation_rate": np.nan,
        "p50_start_wait_ms": first_scalar(
            row,
            [
                "p50_start_wait_ms",
                "p50_wait_ms",
                "p50_queue_wait_ms",
            ],
        ),
        "p95_start_wait_ms": first_scalar(
            row,
            [
                "p95_start_wait_ms",
                "p95_queue_wait_ms",
                "wait_p95_ms",
            ],
        ),
        "p99_start_wait_ms": first_scalar(
            row,
            [
                "p99_start_wait_ms",
                "p99_wait_ms",
                "wait_p99_ms",
            ],
        ),
        "max_start_wait_ms": first_scalar(row, ["max_start_wait_ms", "max_queue_wait_ms"]),
        "p50_interservice_wait_ms": first_scalar(
            row,
            [
                "p50_interservice_wait_ms",
                "p50_wait_interservice_ms",
                "oldest_p50_interservice_wait_ms",
            ],
        ),
        "p95_interservice_wait_ms": first_scalar(
            row,
            [
                "p95_interservice_wait_ms",
                "p95_wait_interservice_ms",
                "oldest_p95_interservice_wait_ms",
            ],
        ),
        "p99_interservice_wait_ms": first_scalar(
            row,
            [
                "p99_interservice_wait_ms",
                "p99_wait_interservice_ms",
                "oldest_p99_interservice_wait_ms",
            ],
        ),
        "max_interservice_wait_ms": first_scalar(
            row,
            [
                "max_interservice_wait_ms",
                "max_wait_interservice_ms",
                "oldest_max_interservice_wait_ms",
            ],
        ),
        "wait_slo_violation_rate": np.nan,
        "starvation_count_2x_slo": np.nan,
        "jain_fairness": np.nan,
        "mean_slowdown_vs_fifo": np.nan,
        "speedup_vs_fifo_req_s": first_scalar(row, ["speedup_vs_fifo_req_s"]),
        "speedup_vs_fifo_tok_s": first_scalar(row, ["speedup_vs_fifo_tok_s", "speedup_vs_fifo_tok"]),
    }

    for key in ["source", "seed", "max_batch", "steps", "gen_length", "shape_slo_ms", "active_pool_factor", "arrival_gap_ms"]:
        val = first_scalar(row, [key])
        if val is not None:
            rec[key] = val
    return rec


def summarize_scheduler_trace(policy_df: pd.DataFrame, input_file: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if policy_df.empty:
        return out

    for policy_name, p_df in policy_df.groupby("policy", dropna=False):
        policy_name = str(policy_name)
        start_wait = infer_start_wait_ms(p_df).dropna()
        inter_wait = infer_interservice_wait_ms(p_df).dropna()
        start_stats = _wait_quantiles(start_wait)
        inter_stats = _wait_quantiles(inter_wait)
        rec: dict[str, Any] = {
            "policy": policy_name,
            "input_file": input_file,
            "policy_source": "scheduler_trace",
            "n_requests": int(len(p_df)),
            "n_rows": int(len(p_df)),
            "wall_s": None,
            "req_per_s": np.nan,
            "tok_per_s": np.nan,
            "p50_latency_ms": np.nan,
            "p95_latency_ms": np.nan,
            "p99_latency_ms": np.nan,
            "max_latency_ms": np.nan,
            "mean_prompt_len": np.nan,
            "min_prompt_len": np.nan,
            "max_prompt_len": np.nan,
            "latency_estimated": False,
            "slo_ms": first_scalar(p_df, ["shape_slo_ms", "slo_ms", "slo"]),
            "slo_violation_rate": np.nan,
            "p50_start_wait_ms": start_stats["p50"],
            "p95_start_wait_ms": start_stats["p95"],
            "p99_start_wait_ms": start_stats["p99"],
            "max_start_wait_ms": start_stats["max"],
            "p50_interservice_wait_ms": inter_stats["p50"],
            "p95_interservice_wait_ms": inter_stats["p95"],
            "p99_interservice_wait_ms": inter_stats["p99"],
            "max_interservice_wait_ms": inter_stats["max"],
            "wait_slo_violation_rate": np.nan,
            "starvation_count_2x_slo": np.nan,
            "jain_fairness": np.nan,
            "mean_slowdown_vs_fifo": np.nan,
            "speedup_vs_fifo_req_s": np.nan,
            "speedup_vs_fifo_tok_s": np.nan,
        }

        for key in [
            "source",
            "seed",
            "max_batch",
            "steps",
            "gen_length",
            "shape_slo_ms",
            "active_pool_factor",
            "arrival_gap_ms",
        ]:
            val = first_scalar(p_df, [key])
            if val is None:
                val = infer_from_filename(input_file, key)
            if val is not None:
                rec[key] = val
        out.append(rec)

    return out


def summarize_policy_rows(df: pd.DataFrame, bucket_edges: list[float]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["latency_ms"] = infer_latency_ms(out)
    out = out[out["latency_ms"].notna()].copy()
    if out.empty:
        return out
    out["policy"] = out.get("policy", out.get("mode", pd.Series(dtype=str)))
    if "policy" in out.columns and out["policy"].isna().any():
        out["policy"] = out["policy"].fillna("unknown")
    out["length_bucket"] = add_bucket(out, bucket_edges)
    return out


def jain_fairness(x: pd.Series | np.ndarray | list[float]) -> float | None:
    vals = pd.Series(x, dtype=float).to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    total = float(np.sum(vals))
    if total <= 0.0:
        return None
    sq = float(np.sum(vals * vals))
    if sq <= 0.0:
        return None
    return float((total * total) / (len(vals) * sq))


def compare_bucket_fairness(
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    bucket_edges: list[float],
) -> pd.DataFrame:
    output_cols = [
        "input_file",
        "policy",
        "source",
        "seed",
        "max_batch",
        "steps",
        "gen_length",
        "arrival_gap_ms",
        "shape_slo_ms",
        "length_bucket",
        "n_rows",
        "p50_latency_ms",
        "p95_latency_ms",
        "mean_latency_ms",
        "slowdown_mean",
        "slowdown_p95",
    ]
    if raw_df.empty:
        return pd.DataFrame(columns=output_cols)

    raw_df = summarize_policy_rows(raw_df, bucket_edges)
    if raw_df.empty:
        return pd.DataFrame(columns=output_cols)
    raw_df = raw_df.copy()

    group_keys = [
        "input_file",
        "source",
        "seed",
        "max_batch",
        "steps",
        "gen_length",
        "arrival_gap_ms",
        "shape_slo_ms",
        "active_pool_factor",
    ]
    group_keys = [k for k in group_keys if k in raw_df.columns]

    rows: list[dict[str, Any]] = []

    for _, group in raw_df.groupby(group_keys, dropna=False):
        if group.empty:
            continue
        if "length_bucket" not in group.columns:
            continue
        fifo_group = group[group["policy"].astype(str).str.lower() == "fifo"]
        if fifo_group.empty:
            continue
        fifo_base = fifo_group.groupby("length_bucket")["latency_ms"].mean()
        fifo_fallback = float(fifo_group["latency_ms"].mean()) if not fifo_group.empty else None
        if pd.isna(fifo_fallback):
            fifo_fallback = None

        for policy, p_df in group.groupby("policy", dropna=False):
            p_df = p_df.copy()
            lat = p_df["latency_ms"].dropna()
            if lat.empty:
                continue
            policy_name = str(policy)
            slowdown_vals: list[float] = []

            for bucket, b_df in p_df.groupby("length_bucket"):
                base = fifo_base.get(bucket)
                if base is None or base <= 0:
                    if fifo_fallback is None or not np.isfinite(fifo_fallback):
                        continue
                    base = fifo_fallback
                mean_b = float(b_df["latency_ms"].mean())
                slowdown = mean_b / float(base)
                slowdown_vals.append(slowdown)
                rows.append(
                    {
                        "input_file": p_df["input_file"].iloc[0],
                        "policy": policy_name,
                        "source": first_scalar(p_df, ["source"]),
                        "seed": first_scalar(p_df, ["seed"]),
                        "max_batch": first_scalar(p_df, ["max_batch"]),
                        "steps": first_scalar(p_df, ["steps"]),
                        "gen_length": first_scalar(p_df, ["gen_length"]),
                        "arrival_gap_ms": first_scalar(p_df, ["arrival_gap_ms"]),
                        "shape_slo_ms": first_scalar(p_df, ["shape_slo_ms"]),
                        "length_bucket": bucket,
                        "n_rows": int(len(b_df)),
                        "p50_latency_ms": float(b_df["latency_ms"].quantile(0.50)),
                        "p95_latency_ms": float(b_df["latency_ms"].quantile(0.95)),
                        "mean_latency_ms": float(b_df["latency_ms"].mean()),
                        "slowdown_mean": slowdown,
                        "slowdown_p95": float((b_df["latency_ms"] / float(base)).quantile(0.95)),
                    }
                )

            norm = []
            for bucket, b_df in p_df.groupby("length_bucket"):
                base = fifo_base.get(bucket)
                if base is None or base <= 0:
                    base = fifo_fallback
                if base is None or base <= 0:
                    continue
                norm.extend((b_df["latency_ms"] / float(base)).tolist())

            if norm:
                j = jain_fairness(norm)
                mask = (summary_df["input_file"] == p_df["input_file"].iloc[0]) & (
                    summary_df["policy"] == policy_name
                )
                if not mask.empty:
                    summary_df.loc[mask, "jain_fairness"] = j
            if slowdown_vals and policy_name.lower() != "fifo":
                mask = (summary_df["input_file"] == p_df["input_file"].iloc[0]) & (
                    summary_df["policy"] == policy_name
                )
                if not mask.empty:
                    summary_df.loc[mask, "mean_slowdown_vs_fifo"] = float(np.mean(slowdown_vals))

            slo_col = first_present(p_df, ["slo_ms", "shape_slo_ms", "slo"])
            if slo_col:
                slo_vals = to_numeric(p_df, slo_col).dropna()
                if not slo_vals.empty:
                    slo = float(slo_vals.iloc[0])
                    mask = (summary_df["input_file"] == p_df["input_file"].iloc[0]) & (
                        summary_df["policy"] == policy_name
                    )
                    summary_df.loc[mask, "slo_ms"] = slo
                    summary_df.loc[mask, "slo_violation_rate"] = float((lat > slo).mean())

    return pd.DataFrame(rows, columns=output_cols)


def apply_speedup(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    group_keys = [
        "source",
        "seed",
        "max_batch",
        "steps",
        "gen_length",
        "shape_slo_ms",
        "active_pool_factor",
        "arrival_gap_ms",
    ]
    group_keys = [k for k in group_keys if k in summary_df.columns]
    if not group_keys:
        return summary_df

    summary_df["speedup_vs_fifo_req_s"] = np.nan
    summary_df["speedup_vs_fifo_tok_s"] = np.nan

    for _, group in summary_df.groupby(group_keys, dropna=False):
        if group.empty:
            continue
        fifo = group[group["policy"].astype(str).str.lower() == "fifo"]
        if fifo.empty:
            continue
        fifo_req = pd.to_numeric(fifo["req_per_s"], errors="coerce").dropna()
        fifo_tok = pd.to_numeric(fifo["tok_per_s"], errors="coerce").dropna()
        fifo_req_val = float(fifo_req.iloc[0]) if not fifo_req.empty else None
        fifo_tok_val = float(fifo_tok.iloc[0]) if not fifo_tok.empty else None

        for idx in group.index:
            if fifo_req_val is not None:
                summary_df.loc[idx, "speedup_vs_fifo_req_s"] = (
                    pd.to_numeric(summary_df.loc[idx, "req_per_s"], errors="coerce") / fifo_req_val
                    if pd.notna(summary_df.loc[idx, "req_per_s"])
                    else np.nan
                )
            if fifo_tok_val is not None:
                summary_df.loc[idx, "speedup_vs_fifo_tok_s"] = (
                    pd.to_numeric(summary_df.loc[idx, "tok_per_s"], errors="coerce") / fifo_tok_val
                    if pd.notna(summary_df.loc[idx, "tok_per_s"])
                    else np.nan
                )

        fifo_idx = group[group["policy"].astype(str).str.lower() == "fifo"].index
        summary_df.loc[fifo_idx, "speedup_vs_fifo_req_s"] = 1.0
        summary_df.loc[fifo_idx, "speedup_vs_fifo_tok_s"] = 1.0
    return summary_df


def _summary_stats_for_metric(series: pd.Series) -> tuple[float | None, float | None, float | None]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return np.nan, np.nan, np.nan
    mean = float(vals.mean())
    if len(vals) == 1:
        return mean, 0.0, 0.0
    std = float(vals.std(ddof=1))
    ci95 = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 0 else np.nan
    return mean, std, ci95


def summarize_seed_stats(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    required = [
        "source",
        "policy",
        "max_batch",
        "steps",
        "gen_length",
        "shape_slo_ms",
        "active_pool_factor",
        "arrival_gap_ms",
    ]
    group_cols = [c for c in required if c in summary_df.columns]
    if "seed" not in summary_df.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, group in summary_df.groupby(group_cols, dropna=False):
        if group.empty:
            continue
        if "seed" not in group.columns:
            continue
        seed_vals = pd.to_numeric(group["seed"], errors="coerce").dropna()
        seed_count = int(seed_vals.nunique(dropna=True))
        if seed_count == 0:
            continue
        rec: dict[str, Any] = {}
        for key in group_cols:
            if key in {"policy", "source"}:
                rec[key] = first_value(group, [key])
            else:
                rec[key] = first_scalar(group, [key])
        rec["n_seeds"] = seed_count

        for metric in [
            "req_per_s",
            "tok_per_s",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "max_latency_ms",
            "p50_start_wait_ms",
            "p95_start_wait_ms",
            "p99_start_wait_ms",
            "max_start_wait_ms",
            "p50_interservice_wait_ms",
            "p95_interservice_wait_ms",
            "p99_interservice_wait_ms",
            "max_interservice_wait_ms",
            "slo_violation_rate",
            "wait_slo_violation_rate",
            "starvation_count_2x_slo",
            "mean_slowdown_vs_fifo",
            "speedup_vs_fifo_req_s",
            "speedup_vs_fifo_tok_s",
            "jain_fairness",
        ]:
            mean, std, ci95 = _summary_stats_for_metric(group[metric])
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_std"] = std
            rec[f"{metric}_ci95"] = ci95
        rows.append(rec)

    return pd.DataFrame(rows)


def make_raw_request_rows(df: pd.DataFrame, input_file: str, bucket_edges: list[float]) -> pd.DataFrame:
    out = df.copy()
    out["input_file"] = input_file
    if "policy" not in out.columns:
        if "mode" in out.columns:
            out["policy"] = out["mode"]
        else:
            out["policy"] = "unknown"
    out["latency_ms"] = infer_latency_ms(out)
    out = out[out["latency_ms"].notna()].copy()
    if out.empty:
        return out

    out["length_bucket"] = add_bucket(out, bucket_edges)
    if "slo_ms" not in out.columns and "shape_slo_ms" in out.columns:
        out["slo_ms"] = to_numeric(out, "shape_slo_ms")

    for col in ["source", "seed", "max_batch", "steps", "gen_length", "shape_slo_ms", "active_pool_factor", "arrival_gap_ms"]:
        if col not in out.columns and col in df.columns:
            out[col] = to_numeric(df, col).iloc[0] if not to_numeric(df, col).empty else np.nan
    return out.reset_index(drop=True)


def format_num(val: Any, precision: int = 4) -> str:
    if pd.isna(val):
        return "N/A"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if np.isnan(v) or np.isinf(v):
        return "N/A"
    return f"{v:.{precision}g}"


def write_latex(summary_df: pd.DataFrame, bucket_df: pd.DataFrame, out: Path) -> None:
    lines: list[str] = []
    lines.append("% Auto-generated by analyze_dlmserve_scheduler_results.py")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lrrrrrrrrrrrr}")
    lines.append(
        "Policy & Req/s & Tok/s & SLO~viol & p95 lat & p99 lat & P50 start & P95 start & P99 start & "
        "P95 inter & P99 inter & Wait viol & Jain & Slowdown"
    )
    lines.append("\\\\")
    lines.append("\\hline")
    if summary_df.empty:
        lines.append("No data\\\\")
    else:
        for _, row in summary_df.sort_values(["input_file", "policy"]).iterrows():
            policy = str(row.get("policy", ""))
            batch = row.get("max_batch")
            batch_suffix = f"@b{int(batch)}" if pd.notna(batch) else ""
            lines.append(
                f"{policy}{batch_suffix} & {format_num(row.get('req_per_s'))} & "
                f"{format_num(row.get('tok_per_s'))} & {format_num(row.get('slo_violation_rate'))} & "
                f"{format_num(row.get('p95_latency_ms'))} & {format_num(row.get('p99_latency_ms'))} & "
                f"{format_num(row.get('p50_start_wait_ms'))} & {format_num(row.get('p95_start_wait_ms'))} & "
                f"{format_num(row.get('p99_start_wait_ms'))} & {format_num(row.get('p95_interservice_wait_ms'))} & "
                f"{format_num(row.get('p99_interservice_wait_ms'))} & {format_num(row.get('wait_slo_violation_rate'))} & "
                f"{format_num(row.get('jain_fairness'))} & {format_num(row.get('mean_slowdown_vs_fifo'))}\\\\"
            )
    lines.append("\\end{tabular}")
    lines.append("\\caption{dlmserve scheduler latency, throughput, fairness, and slowdown summary}")
    lines.append("\\label{tab:dlmserve-scheduler-summary}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{llrrrr}")
    lines.append("Policy & Bucket & Count & p50 & Slowdown & p95 slow\\")
    lines.append("\\hline")
    if bucket_df.empty:
        lines.append("No per-row request data available\\\\")
    else:
        for _, row in bucket_df.iterrows():
            lines.append(
                f"{row.get('policy','')} & {row.get('length_bucket','')} & {int(row.get('n_rows') or 0)} & "
                f"{format_num(row.get('mean_latency_ms'))} & {format_num(row.get('slowdown_mean'))} & "
                f"{format_num(row.get('slowdown_p95'))}\\\\"
            )
    lines.append("\\end{tabular}")
    lines.append("\\caption{per-length-bucket latency slowdown against FIFO}")
    lines.append("\\label{tab:dlmserve-scheduler-bucket}")
    lines.append("\\end{table}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def gather_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input_csv:
        inputs = [Path(x) for x in args.input_csv]
    else:
        inputs = list(DEFAULT_INPUTS)
    if args.glob:
        inputs.extend(sorted(RESULTS.glob(args.glob)))
    # dedupe and keep deterministic order
    out: list[Path] = []
    seen: set[Path] = set()
    for p in inputs:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def main() -> None:
    args = parse_args()
    bucket_edges = parse_bucket_bounds(args.bucket_bounds)
    inputs = gather_inputs(args)

    summary_rows: list[dict[str, Any]] = []
    raw_rows: list[pd.DataFrame] = []

    for path in inputs:
        if not path.exists():
            print(f"skip missing file: {path}")
            continue

        df = canonical_columns(pd.read_csv(path))
        if df.empty:
            print(f"skip empty file: {path}")
            continue

        policy_col = first_present(df, ["mode", "policy"])
        if policy_col is None:
            print(f"skip (no mode/policy column): {path}")
            continue
        if "mode" in df.columns and "policy" not in df.columns:
            df = df.rename(columns={"mode": "policy"})
            policy_col = "policy"

        dtype = detect_dataset_type(df, policy_col)
        if dtype == "per_row":
            for policy_name, policy_df in df.groupby(df[policy_col], dropna=False):
                summary_rows.append(
                    summarize_policy_from_per_row(
                        file_df=df,
                        policy=str(policy_name),
                        policy_df=policy_df,
                        input_file=path.name,
                        fallback_wall_latency=args.fallback_wall_latency,
                    )
                )
            raw_rows.append(make_raw_request_rows(df, path.name, bucket_edges))
        elif dtype == "scheduler_trace":
            summary_rows.extend(
                summarize_scheduler_trace(df, path.name)
            )
        else:
            for _, row in df.iterrows():
                summary_rows.append(summarize_all_at_once_row(row, path.name))

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        print("No parsed rows. Nothing to emit.")
        return

    summary_df = apply_speedup(summary_df)
    seed_stats_df = summarize_seed_stats(summary_df)
    raw_df = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()
    bucket_df = compare_bucket_fairness(raw_df, summary_df, bucket_edges)

    summary_path = Path(args.summary_out)
    bucket_path = Path(args.bucket_out)
    latex_path = Path(args.latex_out)
    seed_stats_path = Path(args.seed_stats_out)
    summary_df.to_csv(summary_path, index=False)
    bucket_df.to_csv(bucket_path, index=False)
    seed_stats_df.to_csv(seed_stats_path, index=False)
    write_latex(summary_df, bucket_df, latex_path)

    print(f"saved {summary_path}")
    print(f"saved {bucket_path}")
    print(f"saved {latex_path}")
    print(f"saved {seed_stats_path}")


if __name__ == "__main__":
    main()
