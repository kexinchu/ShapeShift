import asyncio
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from shapeshift import ShapeShiftScheduler


def state(length, *, step=0, leap=False, gen=8):
    p = NS(temperature=0, use_local_leap=leap, local_leap_anchor_threshold=.9,
           local_leap_neighbor_threshold=.75, local_leap_radius=4)
    s = NS(prompt_len=length-gen, gen_length=gen, block_idx=0, step_in_block=step,
           block_length=gen, steps_per_block=4, params=p)
    return s, p


class SchedulerTests(unittest.TestCase):
    def make(self, lengths, **kwargs):
        self.now = 0
        scheduler = ShapeShiftScheduler(clock=lambda: self.now, **kwargs)
        reqs = [scheduler.admit(*state(n)) for n in lengths]
        return scheduler, reqs

    def test_compactness_and_state_preservation(self):
        s, r = self.make([1000, 100, 110], max_batch=3, padding_ratio=1.1)
        self.assertEqual(s.schedule(), [r[1], r[2]])
        self.assertEqual(s.active_count, 3)
        self.assertEqual(r[0].state.prompt_len, 992)
        self.now = 5000
        self.assertEqual(s.schedule(), [r[0]])  # old outlier becomes anchor

    def test_fifo_and_shape_ablation(self):
        s, r = self.make([1000, 100, 110], max_batch=2, shape_policy="fifo")
        self.assertEqual(s.schedule(), r[:2])
        s.shape_policy = "shape_only"
        self.assertEqual(s.schedule(), [r[1], r[2]])

    def test_aging_prevents_persistent_outlier_starvation(self):
        s, r = self.make([1000, 100, 100], max_batch=1, shape_slo_ms=10)
        seen = set()
        for tick in range(20):
            self.now = tick * 10
            seen.update(x.req_id for x in s.schedule())
        self.assertEqual(seen, {0, 1, 2})

    def test_arrivals_reconsidered_each_step(self):
        s, r = self.make([1000, 100], max_batch=2)
        self.assertEqual(s.schedule(), [r[1]])
        new = s.admit(*state(105))
        self.now = 1
        self.assertEqual(s.schedule(), [r[1], new])

    def test_compatibility(self):
        s, r = self.make([100], max_batch=8, compatibility_mode="strict")
        for spec in (dict(step=1), dict(leap=True), dict(gen=16)):
            s.admit(*state(100, **spec))
        self.assertEqual(s.schedule(), r)

    def test_bounded_pool_and_backpressure(self):
        s, r = self.make([100, 200], max_batch=1, active_pool_factor=1, max_waiting=2)
        self.assertIsNone(s.admit(*state(300)))
        self.assertEqual(s.schedule(), [r[0]])
        self.assertEqual((s.active_count, s.queue_depth), (1, 1))
        s.complete(r[0])
        self.assertEqual(s.schedule(), [r[1]])
        s.complete(r[1])
        self.assertFalse(s.has_work())

    def test_future_completion_and_cancellation(self):
        loop = asyncio.new_event_loop()
        try:
            s = ShapeShiftScheduler()
            future = loop.create_future()
            r = s.admit(*state(100), future=future)
            s.schedule()
            r.result = "done"
            s.complete(r)
            s.complete(r)
            self.assertEqual(future.result(), "done")
            f = loop.create_future()
            s.admit(*state(100), future=f)
            f.cancel()
            self.assertEqual(s.schedule(), [])
            self.assertFalse(s.has_work())
        finally:
            loop.close()

    def test_invalid_configuration_and_stochastic_requests(self):
        for kwargs in ({"max_batch": 0}, {"active_pool_factor": 0},
                       {"padding_ratio": .9}, {"shape_slo_ms": float("nan")},
                       {"shape_policy": "typo"}):
            with self.assertRaises(ValueError):
                ShapeShiftScheduler(**kwargs)
        s, p = state(100)
        p.temperature = 1
        with self.assertRaises(ValueError):
            ShapeShiftScheduler().admit(s, p)

    def test_trace_matches_selected_rectangle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.csv"
            s, _ = self.make([100, 110], max_batch=2, trace_csv=path)
            s.schedule()
            s.flush_trace()
            with path.open() as f:
                row = list(csv.DictReader(f))[0]
            self.assertEqual(int(row["padded_tokens"]), 220)
            self.assertEqual(int(row["useful_tokens"]), 210)
            self.assertAlmostEqual(float(row["padding_ratio"]), 220/210)


if __name__ == "__main__":
    unittest.main()
