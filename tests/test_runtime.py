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

    def test_replay_counts_model_failures_and_completes_other_requests(self):
        from shapeshift.replay import TraceRequest, replay
        loaded = NS(model=ToyModel(), tokenizer=ToyTokenizer(), device=torch.device("cpu"), mask_id=15)
        params = SamplingParams(gen_length=8, num_denoising_steps=4)
        with patch("dlmserve.engine.load_model", return_value=loaded):
            engine = Engine(device="cpu", max_batch=1)
            step = engine._step_batch
            def fail_long(batch):
                if batch[0].state.prompt_len == 20:
                    raise RuntimeError("injected model failure")
                return step(batch)
            engine._step_batch = fail_long
            rows, events, summary = replay(engine,
                [TraceRequest("a", "test", 0, 3, 8), TraceRequest("b", "test", 0, 20, 8)],
                params, lambda r: init_state(torch.ones((1,r.prompt_len),dtype=torch.long),params,15))
            self.assertEqual(summary["n_completed"],1)
            self.assertEqual(summary["n_failed"],1)
            self.assertFalse(summary["valid_for_comparison"])
            self.assertEqual([r['status'] for r in rows], ['done','failed'])
            self.assertTrue(events)

    def test_context_dependent_model_mixed_progress_matches_individual_steps(self):
        import copy
        class ContextModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None, position_ids=None):
                context = (input_ids*attention_mask).sum(dim=1,keepdim=True)
                target = 1+(position_ids+context)%10
                logits = torch.zeros((*input_ids.shape,16))
                logits.scatter_(2,target[...,None],(2+position_ids.float()/10)[...,None])
                return NS(logits=logits)
        model = ContextModel()
        params = SamplingParams(gen_length=8, block_length=4, num_denoising_steps=8)
        originals = [init_state(torch.ones((1,n),dtype=torch.long),params,15) for n in [3,7,20]]
        for state, advance in zip(originals,[0,1,4]):
            for _ in range(advance):
                step_batch(model,[state],mask_id=15,pad_token_id=0)
        expected = copy.deepcopy(originals)
        for state in expected:
            step_batch(model,[state],mask_id=15,pad_token_id=0)
        step_batch(model,originals,mask_id=15,pad_token_id=0)
        for actual, reference in zip(originals,expected):
            self.assertTrue(torch.equal(actual.seq,reference.seq))
            self.assertEqual((actual.block_idx,actual.step_in_block),
                             (reference.block_idx,reference.step_in_block))

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
