"""Adapter for the pinned upstream dlmserve engine (LLaDA, deterministic decoding)."""
from dlmserve.engine import Engine as UpstreamEngine
from dlmserve.scheduler import DiffusionScheduler
from .scheduler import ShapeShiftScheduler


class Engine(UpstreamEngine):
    def __init__(self, model_id="gsai-ml/LLaDA-8B-Instruct", dtype="bf16",
                 device="cuda", max_batch=8, *, scheduler_policy="shapeshift",
                 scheduler_slo_ms=5000, scheduler_active_pool_factor=4,
                 scheduler_padding_ratio=1.25, scheduler_trace_csv=None,
                 max_waiting=256, bucket_width=256, compatibility_mode="progress_independent"):
        self.scheduler_options = dict(max_batch=max_batch, max_waiting=max_waiting,
            shape_slo_ms=scheduler_slo_ms, active_pool_factor=scheduler_active_pool_factor,
            padding_ratio=scheduler_padding_ratio, bucket_width=bucket_width,
            compatibility_mode=compatibility_mode)
        scheduler = self.make_scheduler(scheduler_policy, trace_csv=scheduler_trace_csv)
        super().__init__(model_id=model_id, dtype=dtype, device=device, max_batch=max_batch)
        self._scheduler = scheduler

    def make_scheduler(self, policy, trace_csv=None):
        if policy == "upstream_default":
            # Preserve the pinned upstream policy and its 2B active pool.
            return DiffusionScheduler(max_batch=self.scheduler_options["max_batch"],
                                      max_waiting=self.scheduler_options["max_waiting"])
        return ShapeShiftScheduler(shape_policy=policy, trace_csv=trace_csv,
                                   **self.scheduler_options)

    def reset_scheduler(self, policy):
        if self._scheduler.has_work():
            raise RuntimeError("cannot reset a nonempty scheduler")
        self._scheduler = self.make_scheduler(policy)
