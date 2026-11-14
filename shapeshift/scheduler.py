"""Paper-algorithm reconstruction; not the implementation behind archived results.

Scheduling is single-threaded, as in dlmserve's asyncio engine. Selection marks
service at dispatch. A service-gap target is a priority rule, not a hard SLO.
"""
from __future__ import annotations

import csv
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(eq=False)
class Request:
    req_id: int
    state: Any
    params: Any
    admitted_ms: float
    last_service_ms: float
    generator: Any = field(default=None, repr=False)
    result: Any = field(default=None, repr=False)
    _future: Any = field(default=None, repr=False)

    @property
    def total_len(self) -> int:
        return self.state.prompt_len + self.state.gen_length

    @property
    def compatibility_key(self) -> tuple:
        # Upstream step_batch uses the first row's sampling configuration.
        # Equal generation lengths also make max(P + G) the actual padded width.
        s, p = self.state, self.params
        return (s.block_idx, s.step_in_block, s.block_length, s.steps_per_block,
                s.gen_length, p.temperature, p.use_local_leap,
                p.local_leap_anchor_threshold, p.local_leap_neighbor_threshold,
                p.local_leap_radius)


class ShapeShiftScheduler:
    """Bounded active pool, step-level nearest-length fill, aging and compactness."""

    POLICIES = {"fifo", "length_bucket", "aging_bucket", "shapeshift"}
    ALIASES = {"shape_only": "length_bucket", "shape_aging_old": "aging_bucket",
               "token_budget": "shapeshift"}

    def __init__(self, max_batch: int = 8, max_waiting: int = 256, *,
                 shape_policy: str = "shapeshift", shape_slo_ms: float = 5000,
                 active_pool_factor: int = 4, padding_ratio: float = 1.25,
                 trace_csv: str | Path | None = None,
                 clock: Callable[[], float] | None = None):
        self.max_batch = max_batch
        self.max_waiting = max_waiting
        self.shape_policy = shape_policy
        self.shape_slo_ms = shape_slo_ms
        self.active_pool_factor = active_pool_factor
        self.padding_ratio = padding_ratio
        self._clock = clock or (lambda: time.monotonic() * 1000)
        self._pending: deque[Request] = deque()
        self._active: list[Request] = []
        self._next_id = 0
        self._tick = 0
        self.trace_csv = Path(trace_csv) if trace_csv else None
        self._validate()
        if self.trace_csv:
            self.trace_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_csv.open("w", newline="") as f:
                csv.writer(f).writerow(["tick", "time_ms", "policy", "request_ids",
                                       "batch_size", "max_total_len", "useful_tokens",
                                       "padded_tokens", "padding_ratio", "anchor_gap_ms"])

    def _validate(self) -> str:
        for name in ("max_batch", "active_pool_factor", "max_waiting"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.shape_slo_ms) or self.shape_slo_ms < 0:
            raise ValueError("shape_slo_ms must be finite and nonnegative")
        if not math.isfinite(self.padding_ratio) or self.padding_ratio < 1:
            raise ValueError("padding_ratio must be finite and >= 1")
        policy = self.ALIASES.get(self.shape_policy, self.shape_policy)
        if policy not in self.POLICIES:
            raise ValueError(f"unknown policy: {self.shape_policy}")
        return policy

    def admit(self, state, params, generator=None, future=None) -> Request | None:
        if params.temperature != 0:
            raise ValueError("Reconstructed integration supports temperature=0 only; "
                             "upstream does not thread per-request RNG through step_batch")
        if state.prompt_len < 0 or state.gen_length <= 0:
            raise ValueError("invalid canvas length")
        self._discard_cancelled()
        if len(self._pending) >= self.max_waiting:
            return None
        now = self._clock()
        req = Request(self._next_id, state, params, now, now, generator, _future=future)
        self._next_id += 1
        self._pending.append(req)
        return req

    def _discard_cancelled(self):
        def alive(r):
            return r._future is None or not r._future.cancelled()
        self._pending = deque(r for r in self._pending if alive(r))
        self._active[:] = [r for r in self._active if alive(r)]

    def schedule(self) -> list[Request]:
        policy = self._validate()
        self._discard_cancelled()
        capacity = self.max_batch * self.active_pool_factor
        while self._pending and len(self._active) < capacity:
            self._active.append(self._pending.popleft())
        if not self._active:
            return []
        now = self._clock()
        overdue = [r for r in self._active if now - r.last_service_ms >= self.shape_slo_ms]
        if policy == "fifo":
            anchor = min(self._active, key=lambda r: r.req_id)
        elif policy in {"aging_bucket", "shapeshift"} and overdue:
            anchor = min(overdue, key=lambda r: (r.last_service_ms, r.req_id))
        else:
            anchor = min(self._active, key=lambda r: (r.total_len, r.last_service_ms, r.req_id))
        candidates = [r for r in self._active if r is not anchor
                      and r.compatibility_key == anchor.compatibility_key]
        if policy == "fifo":
            candidates.sort(key=lambda r: r.req_id)
        else:
            candidates.sort(key=lambda r: (abs(r.total_len - anchor.total_len),
                                            r.last_service_ms, r.req_id))
        chosen = [anchor]
        useful, width = anchor.total_len, anchor.total_len
        anchor_gap = now - anchor.last_service_ms
        for req in candidates:
            if len(chosen) == self.max_batch:
                break
            new_width = max(width, req.total_len)
            ratio = (len(chosen) + 1) * new_width / (useful + req.total_len)
            if policy == "shapeshift" and ratio > self.padding_ratio:
                continue
            chosen.append(req)
            useful += req.total_len
            width = new_width
        for req in chosen:
            req.last_service_ms = now
        if self.trace_csv:
            with self.trace_csv.open("a", newline="") as f:
                csv.writer(f).writerow([self._tick, now, policy,
                    ";".join(str(r.req_id) for r in chosen), len(chosen), width,
                    useful, len(chosen) * width, len(chosen) * width / useful, anchor_gap])
        self._tick += 1
        return chosen

    def complete(self, req: Request) -> None:
        if req in self._active:
            self._active.remove(req)
        elif req in self._pending:
            self._pending.remove(req)
        if req._future is not None and not req._future.done():
            req._future.set_result(req.result)

    def has_work(self) -> bool:
        self._discard_cancelled()
        return bool(self._pending or self._active)

    @property
    def queue_depth(self) -> int:
        return len(self._pending)

    @property
    def active_count(self) -> int:
        return len(self._active)
