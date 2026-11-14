"""Audit witnesses, not acceptance of documented paper/implementation differences."""
import csv
import importlib.util
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from shapeshift import ShapeShiftScheduler


def make_state(length, step=0):
    params = NS(temperature=0, use_local_leap=False,
                local_leap_anchor_threshold=.9, local_leap_neighbor_threshold=.75,
                local_leap_radius=4)
    return NS(prompt_len=length-8, gen_length=8, block_idx=0,
              step_in_block=step, block_length=8, steps_per_block=8), params


def paper_select(requests, now, batch_size, tau, beta):
    """Literal Algorithm 1 reference, with stable request-ID tie resolution."""
    if not requests:
        return []
    overdue = [r for r in requests if now-r.last_service_ms >= tau]
    anchor = (min(overdue, key=lambda r: (r.last_service_ms, r.req_id)) if overdue
              else min(requests, key=lambda r: (r.total_len, r.last_service_ms, r.req_id)))
    chosen = [anchor]
    for req in sorted((r for r in requests if r is not anchor),
                      key=lambda r: (abs(r.total_len-anchor.total_len),
                                     r.last_service_ms, r.req_id)):
        trial = chosen+[req]
        if len(chosen) < batch_size and len(trial)*max(r.total_len for r in trial)/sum(r.total_len for r in trial) <= beta:
            chosen.append(req)
    return [r.req_id for r in chosen]


class PaperAuditTests(unittest.TestCase):
    def test_2000_homogeneous_ready_pools_match_algorithm_one(self):
        rng = random.Random(13)
        for _ in range(2000):
            n, b = rng.randint(1, 32), rng.randint(1, 8)
            tau, beta = rng.choice([0, 10, 5000]), rng.choice([1, 1.1, 1.25, 2])
            now = 10000
            scheduler = ShapeShiftScheduler(max_batch=b, active_pool_factor=n,
                shape_slo_ms=tau, padding_ratio=beta, clock=lambda: now)
            reqs = [scheduler.admit(*make_state(rng.randint(8, 4096))) for _ in range(n)]
            for req in reqs:
                req.last_service_ms = rng.randint(0, now)
            expected = paper_select(reqs, now, b, tau, beta)
            self.assertEqual([r.req_id for r in scheduler.schedule()], expected)

    def test_mixed_steps_now_match_paper_pool(self):
        scheduler = ShapeShiftScheduler(max_batch=2, shape_slo_ms=5000, clock=lambda: 0)
        reqs = [scheduler.admit(*make_state(length, step))
                for length, step in [(100, 0), (101, 1), (110, 0)]]
        reference = paper_select(reqs, 0, 2, 5000, 1.25)
        actual = [r.req_id for r in scheduler.schedule()]
        self.assertEqual(reference, [0, 1])
        self.assertEqual(actual, [0, 1])
        self.assertEqual(actual, reference)

    @unittest.skipUnless(importlib.util.find_spec('dlmserve') is not None,
                         'optional pinned upstream not installed')
    def test_witness_fifo_is_not_pinned_upstream_default(self):
        import torch
        from unittest.mock import patch
        from dlmserve.denoise_loop import init_state
        from dlmserve.sampler import SamplingParams
        from dlmserve.scheduler import DiffusionScheduler
        params = SamplingParams(gen_length=8, num_denoising_steps=8)
        upstream = DiffusionScheduler(max_batch=2)
        rebuilt = ShapeShiftScheduler(max_batch=2, shape_policy='fifo', clock=lambda: 0)
        with patch('dlmserve.scheduler.time.monotonic', return_value=0):
            for step in [0, 1, 1]:
                for scheduler in [upstream, rebuilt]:
                    state = init_state(torch.ones((1, 16), dtype=torch.long), params, mask_id=15)
                    state.step_in_block = step
                    scheduler.admit(state, params)
            self.assertEqual([r.req_id for r in upstream.schedule()], [1, 2])
            self.assertEqual([r.req_id for r in rebuilt.schedule()], [0, 1])

    def test_trace_reader_preserves_order_and_arrival_times(self):
        from shapeshift.replay import load_trace
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'rows.csv'
            with path.open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['source', 'request_uid', 'prompt_len', 'gen_len', 'total_len', 'arrival_ms'])
                w.writerows([['test', str(n), n, 128, n+128, n*n] for n in range(10,20)])
            actual = load_trace(path)
            self.assertEqual([r.prompt_len for r in actual], list(range(10,20)))
            self.assertEqual([r.arrival_ms for r in actual], [n*n-100 for n in range(10,20)])


if __name__ == '__main__':
    unittest.main()
