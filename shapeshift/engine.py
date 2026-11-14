"""Adapter for the pinned upstream dlmserve engine (LLaDA, deterministic decoding)."""
from dlmserve.engine import Engine as UpstreamEngine
from .scheduler import ShapeShiftScheduler


class Engine(UpstreamEngine):
    def __init__(self, model_id="gsai-ml/LLaDA-8B-Instruct", dtype="bf16",
                 device="cuda", max_batch=8, *, scheduler_policy="shapeshift",
                 scheduler_slo_ms=5000, scheduler_active_pool_factor=4,
                 scheduler_padding_ratio=1.25, scheduler_trace_csv=None,
                 max_waiting=256):
        scheduler = ShapeShiftScheduler(
            max_batch=max_batch, max_waiting=max_waiting, shape_policy=scheduler_policy,
            shape_slo_ms=scheduler_slo_ms, active_pool_factor=scheduler_active_pool_factor,
            padding_ratio=scheduler_padding_ratio, trace_csv=scheduler_trace_csv)
        super().__init__(model_id=model_id, dtype=dtype, device=device, max_batch=max_batch)
        self._scheduler = scheduler
