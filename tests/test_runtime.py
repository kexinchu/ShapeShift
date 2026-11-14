"""CPU integration against real upstream code; no model downloads."""
import importlib.util
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ("torch", "dlmserve"))

if AVAILABLE:
    import torch
    from dlmserve.denoise_loop import init_state, step_batch
    from dlmserve.sampler import SamplingParams
    from shapeshift.engine import Engine
    from shapeshift import ShapeShiftScheduler

    class ToyModel(torch.nn.Module):
        config = NS()

        def forward(self, input_ids, attention_mask=None, position_ids=None):
            # Position-dependent token and confidence, independent of padding.
            positions = position_ids if position_ids is not None else torch.arange(input_ids.shape[1])[None, :]
            logits = torch.zeros((*input_ids.shape, 16))
            tokens = 1 + positions % 10
            logits.scatter_(2, tokens[..., None].expand(input_ids.shape[0], -1, -1),
                            (2 + positions.float() / 10)[..., None].expand(input_ids.shape[0], -1, -1))
            return NS(logits=logits)

    class ToyTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, prompts, **kwargs):
            ids = torch.tensor([[1] * int(p) for p in prompts])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def batch_decode(self, ids, **kwargs):
            return [" ".join(map(str, row.tolist())) for row in ids]


@unittest.skipUnless(AVAILABLE, "optional runtime dependencies not installed")
class RuntimeTests(unittest.TestCase):
    def test_real_step_loop_matches_individual_execution(self):
        params = SamplingParams(gen_length=8, block_length=4, num_denoising_steps=8)
        model = ToyModel()
        reference = []
        for length in (3, 20, 5, 30):
            s = init_state(torch.ones((1, length), dtype=torch.long), params, mask_id=15)
            while not s.done:
                step_batch(model, [s], mask_id=15, pad_token_id=0)
            reference.append(s.seq.clone())
        for policy in ("fifo", "length_bucket", "aging_bucket", "shapeshift"):
            with self.subTest(policy=policy):
                scheduler = ShapeShiftScheduler(max_batch=3, shape_policy=policy, shape_slo_ms=0)
                requests = [scheduler.admit(init_state(torch.ones((1, n), dtype=torch.long),
                            params, mask_id=15), params) for n in (3, 20, 5, 30)]
                for _ in range(100):
                    if not scheduler.has_work():
                        break
                    batch = scheduler.schedule()
                    step_batch(model, [r.state for r in batch], mask_id=15, pad_token_id=0)
                    for r in batch:
                        if r.state.done:
                            scheduler.complete(r)
                self.assertFalse(scheduler.has_work())
                for r, expected in zip(requests, reference):
                    self.assertTrue(torch.equal(r.state.seq, expected))

    def test_benchmark_does_not_count_failed_requests_as_completions(self):
        import asyncio
        import importlib.util
        from pathlib import Path
        runner_path = Path(__file__).resolve().parents[1] / "cache-sparse-attention/code/run_dlmserve_online_trace_benchmark.py"
        spec = importlib.util.spec_from_file_location("benchmark_runner", runner_path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        runner.torch = torch
        loaded = NS(model=ToyModel(), tokenizer=ToyTokenizer(),
                    device=torch.device("cpu"), mask_id=15)
        with patch("dlmserve.engine.load_model", return_value=loaded):
            engine = Engine(device="cpu", max_batch=1)
            step = engine._step_batch
            def fail_long(batch):
                if batch[0].state.prompt_len == 20:
                    raise RuntimeError("injected model failure")
                return step(batch)
            engine._step_batch = fail_long
            with self.assertLogs("dlmserve.engine", level="ERROR"):
                rows, summary, _ = asyncio.run(runner.benchmark_policy(
                    engine=engine, prompts=["3", "20"], prompt_lens=[3, 20],
                    params=SamplingParams(gen_length=8, num_denoising_steps=4),
                    max_batch=1, policy="fifo", n_requests=2, arrival_gap_ms=0,
                    seed=13, shape_slo_ms=5000, active_pool_factor=4,
                    device=torch.device("cpu")))
            self.assertEqual(summary["n_completed"], 1)
            self.assertEqual(summary["n_failed"], 1)
            self.assertEqual([r["status"] for r in rows], ["done", "failed"])

    def test_engine_generate_uses_adapter_and_completes_futures(self):
        loaded = NS(model=ToyModel(), tokenizer=ToyTokenizer(),
                    device=torch.device("cpu"), mask_id=15)
        with patch("dlmserve.engine.load_model", return_value=loaded):
            engine = Engine(device="cpu", max_batch=3, scheduler_slo_ms=0)
            self.assertIsInstance(engine._scheduler, ShapeShiftScheduler)
            result = engine.generate(["3", "20", "5"],
                SamplingParams(gen_length=8, block_length=4, num_denoising_steps=8),
                prerendered=True)
            self.assertEqual(len(result), 3)
            self.assertTrue(all(len(r.output_ids) == 8 for r in result))
            self.assertFalse(engine._scheduler.has_work())


if __name__ == "__main__":
    unittest.main()
