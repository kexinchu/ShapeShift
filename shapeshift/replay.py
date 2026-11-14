"""Absolute-arrival, step-boundary replay with explicit admission delay.

No model dependencies. Input timestamps never depend on service duration.
All event records are buffered and written after the timed region.
"""
from __future__ import annotations

import csv
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class TraceRequest:
    uid: str
    source: str
    arrival_ms: float
    prompt_len: int
    gen_len: int
    prompt: str | None = None


def load_trace(path: Path, *, source: str | None = None, gen_length: int = 128,
               arrival_mode: str = "timestamps", arrival_gap_ms: float = 0,
               limit: int | None = None, spacing: float = 1.0) -> list[TraceRequest]:
    """Preserve input order. Reject malformed traces instead of silently filtering."""
    if arrival_mode not in {"timestamps", "constant", "all_ready"}:
        raise ValueError("unknown arrival mode")
    if (not math.isfinite(spacing) or spacing <= 0 or
            not math.isfinite(arrival_gap_ms) or arrival_gap_ms < 0 or gen_length < 1):
        raise ValueError("invalid arrival/generation settings")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    result, seen = [], set()
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            src = row.get("source", row.get("task", ""))
            if source is not None and src != source:
                continue
            uid = row.get("request_uid", "")
            if not uid or uid in seen:
                raise ValueError("request_uid must be nonempty and unique in selected trace")
            seen.add(uid)
            length = int(row["prompt_len"])
            g = int(row["gen_len"])
            if length < 1 or g != gen_length:
                raise ValueError("positive prompt_len and matching gen_len required")
            if row.get("total_len") and int(row["total_len"]) != length + g:
                raise ValueError("total_len does not match prompt_len + gen_len")
            if arrival_mode == "timestamps":
                arrival = float(row["arrival_ms"]) * spacing
            elif arrival_mode == "constant":
                arrival = len(result) * arrival_gap_ms * spacing
            else:
                arrival = 0.0
            if not math.isfinite(arrival) or arrival < 0 or (result and arrival < result[-1].arrival_ms):
                raise ValueError("arrivals must be finite, nonnegative and nondecreasing")
            result.append(TraceRequest(uid, src, arrival, length, g, row.get("prompt")))
    if not result:
        raise ValueError("empty selected trace")
    if limit is not None:
        if limit > len(result):
            raise ValueError("limit exceeds selected trace length")
        result = result[:limit]
    # Preserve time gaps, but define measurement origin at the first arrival.
    origin = result[0].arrival_ms
    return [TraceRequest(r.uid, r.source, r.arrival_ms-origin, r.prompt_len, r.gen_len, r.prompt)
            for r in result]


def percentile(values, q):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(q*len(values))-1)]


def replay(engine, trace: list[TraceRequest], params, prepare: Callable, *,
           clock=time.perf_counter, sleep=time.sleep, synchronize=lambda: None,
           timeout_s=3600.0):
    """Drive real engine steps, admitting arrivals at each boundary.

    prepare(entry) builds a state on admission from pre-tokenized CPU input.
    End-to-end latency starts at the immutable scheduled arrival, not admission.
    Dispatch/finish are host timestamps around a synchronized device step.
    """
    if not trace or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("nonempty trace and finite positive timeout required")
    if engine._scheduler.has_work():
        raise ValueError("replay needs an empty scheduler")
    synchronize()
    origin = clock()
    now = lambda: (clock()-origin)*1000
    rows = [dict(request_uid=r.uid, source=r.source, scheduled_arrival_ms=r.arrival_ms,
                 actual_arrival_ms=None, admitted_ms=None, first_dispatch_ms=None,
                 completion_ms=None, prompt_len=r.prompt_len, gen_len=r.gen_len,
                 status="pending", error_type="", output_tokens=0, output_sha256=None,
                 dispatch_count=0) for r in trace]
    by_req, previous_dispatch, gaps, steps = {}, {}, [], []
    scheduler = engine._scheduler
    cursor = 0
    while cursor < len(trace) or scheduler.has_work():
        if now() >= timeout_s*1000:
            for row in rows:
                if row['status'] in {'pending', 'admitted'}:
                    row.update(status='timeout', completion_ms=now())
            # Release request canvases; no futures are created by this driver.
            for req in list(scheduler._active) + list(scheduler._pending):
                scheduler.complete(req)
            break
        # Ingress is polled at boundaries; processing cost may make more arrivals due.
        while cursor < len(trace) and trace[cursor].arrival_ms <= now():
            if now() >= timeout_s*1000:
                break
            entry, row = trace[cursor], rows[cursor]
            row['actual_arrival_ms'] = now()
            try:
                state = prepare(entry)
                actual_length = int(state.prompt_len)
                if actual_length != entry.prompt_len or int(state.gen_length) != entry.gen_len:
                    raise ValueError("prepared canvas differs from trace")
                req = scheduler.admit(state, params)
                if req is None:
                    row.update(status='rejected', completion_ms=now())
                else:
                    row.update(status='admitted', admitted_ms=now())
                    by_req[req.req_id] = cursor
            except Exception as exc:
                row.update(status='failed', error_type=type(exc).__name__, completion_ms=now())
            cursor += 1
        if not scheduler.has_work():
            if cursor < len(trace):
                delay = min(trace[cursor].arrival_ms-now(), timeout_s*1000-now())
                if delay > 0:
                    sleep(delay/1000)
            continue
        start_schedule = now()
        batch = scheduler.schedule()
        dispatch = now()
        if not batch:
            raise RuntimeError("scheduler has work but made no progress")
        ids = [r.req_id for r in batch]
        lengths = [int(r.state.prompt_len+r.state.gen_length) for r in batch]
        # Actual upstream left-padding rectangle (also valid for diagnostic baseline).
        width = max(r.state.prompt_len for r in batch) + max(r.state.gen_length for r in batch)
        for req in batch:
            row = rows[by_req[req.req_id]]
            if row['first_dispatch_ms'] is None:
                row['first_dispatch_ms'] = dispatch
            if req.req_id in previous_dispatch:
                gaps.append(dispatch-previous_dispatch[req.req_id])
            previous_dispatch[req.req_id] = dispatch
            row['dispatch_count'] += 1
        progress_before = [(r.state.block_idx, r.state.step_in_block) for r in batch]
        error = None
        try:
            engine._step_batch(batch)
            synchronize()
        except Exception as exc:
            error = type(exc).__name__
        finish = now()
        steps.append(dict(tick=len(steps), request_ids=ids,
            request_uids=[trace[by_req[rid]].uid for rid in ids],
            dispatch_ms=dispatch, finish_ms=finish, scheduler_ms=dispatch-start_schedule,
            progress_before=progress_before,
            progress_after=[(r.state.block_idx, r.state.step_in_block) for r in batch],
            synchronized_step_ms=finish-dispatch, batch_size=len(batch),
            max_total_len=width, useful_tokens=sum(lengths), padded_tokens=len(batch)*width,
            padding_ratio=len(batch)*width/sum(lengths), error_type=error,
            active_count=scheduler.active_count, pending_count=scheduler.queue_depth))
        for req in batch:
            row = rows[by_req[req.req_id]]
            if error:
                row.update(status='failed', error_type=error, completion_ms=now())
                scheduler.complete(req)
            elif req.state.done:
                try:
                    output = engine._make_output(req)
                    ids_out = output.output_ids.tolist()
                    mask_id = getattr(getattr(engine, '_loaded', None), 'mask_id', None)
                    if len(ids_out) != row['gen_len'] or (mask_id is not None and mask_id in ids_out):
                        raise RuntimeError('incomplete output canvas')
                    digest = hashlib.sha256(','.join(map(str, ids_out)).encode()).hexdigest()
                    row.update(status='done', output_tokens=len(ids_out), output_sha256=digest,
                               completion_ms=now())
                except Exception as exc:
                    row.update(status='failed', error_type=type(exc).__name__, completion_ms=now())
                scheduler.complete(req)
    for row in rows:
        row['response_latency_ms'] = (row['completion_ms']-row['scheduled_arrival_ms']
                                      if row['completion_ms'] is not None and row['completion_ms'] >= row['scheduled_arrival_ms'] else None)
        row['admission_delay_ms'] = (row['admitted_ms']-row['scheduled_arrival_ms']
                                     if row['admitted_ms'] is not None else None)
        row['first_service_wait_ms'] = (row['first_dispatch_ms']-row['scheduled_arrival_ms']
                                        if row['first_dispatch_ms'] is not None else None)
    done = [r for r in rows if r['status']=='done']
    lat = [r['response_latency_ms'] for r in done]
    end = max(r['completion_ms'] for r in rows)
    wall = end-min(r['scheduled_arrival_ms'] for r in rows)
    useful = sum(s['useful_tokens'] for s in steps)
    summary = dict(n_requests=len(rows), n_completed=len(done), n_failed=len(rows)-len(done),
        n_rejected=sum(r['status']=='rejected' for r in rows),
        n_timeout=sum(r['status']=='timeout' for r in rows),
        valid_for_comparison=len(done)==len(rows), wall_ms=wall,
        tok_per_s=sum(r['output_tokens'] for r in done)/(wall/1000) if wall>0 else None,
        req_per_s=len(done)/(wall/1000) if wall>0 else None,
        mean_latency_ms=sum(lat)/len(lat) if lat else None,
        p95_latency_ms=percentile(lat,.95), p99_latency_ms=percentile(lat,.99),
        p95_inter_dispatch_ms=percentile(gaps,.95), inter_dispatch_samples=len(gaps),
        p95_first_service_wait_ms=percentile([r['first_service_wait_ms'] for r in done],.95),
        p95_admission_delay_ms=percentile([r['admission_delay_ms'] for r in done],.95),
        mean_step_padding_ratio=sum(s['padding_ratio'] for s in steps)/len(steps) if steps else None,
        work_weighted_padding_ratio=sum(s['padded_tokens'] for s in steps)/useful if useful else None,
        scheduler_total_ms=sum(s['scheduler_ms'] for s in steps),
        synchronized_step_total_ms=sum(s['synchronized_step_ms'] for s in steps),
        replay_mode='absolute-arrival-step-boundary',
        fairness_definition='dispatch-to-dispatch intervals; first/terminal waits excluded')
    return rows, steps, summary
