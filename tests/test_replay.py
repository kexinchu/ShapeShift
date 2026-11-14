import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from shapeshift.replay import TraceRequest, load_trace, replay
from shapeshift.scheduler import ShapeShiftScheduler


class Clock:
    def __init__(self): self.t=0.0
    def __call__(self): return self.t
    def sleep(self,seconds): self.t+=seconds


class Values(list):
    def tolist(self): return list(self)


class FakeEngine:
    def __init__(self,clock,**options):
        self.clock=clock
        self._scheduler=ShapeShiftScheduler(clock=lambda:clock()*1000,**options)
    def _step_batch(self,batch):
        self.clock.sleep(.01)
        for r in batch:
            r.state.remaining-=1
            r.state.step_in_block+=1
            r.state.done=r.state.remaining==0
    def _make_output(self,req): return NS(output_ids=Values([1]*req.state.gen_length))


def params():
    return NS(temperature=0,use_local_leap=False,local_leap_anchor_threshold=.9,
              local_leap_neighbor_threshold=.75,local_leap_radius=4)


def prepare(r):
    return NS(prompt_len=r.prompt_len,gen_length=r.gen_len,block_idx=0,
              step_in_block=0,block_length=r.gen_len,steps_per_block=2,remaining=2,done=False)


class ReplayTests(unittest.TestCase):
    def test_late_admission_does_not_shift_future_arrivals(self):
        clock=Clock(); engine=FakeEngine(clock,max_batch=2)
        trace=[TraceRequest('a','x',0,100,8),TraceRequest('b','x',3,110,8),TraceRequest('c','x',7,120,8)]
        rows,steps,summary=replay(engine,trace,params(),prepare,clock=clock,sleep=clock.sleep)
        self.assertEqual([r['scheduled_arrival_ms'] for r in rows],[0,3,7])
        self.assertEqual([r['actual_arrival_ms'] for r in rows],[0,10,10])
        self.assertEqual(rows[1]['first_dispatch_ms'],10)
        self.assertEqual(rows[1]['response_latency_ms'],27)
        self.assertEqual(rows[1]['admission_delay_ms'],7)
        self.assertTrue(summary['valid_for_comparison'])
        self.assertEqual(summary['inter_dispatch_samples'],3)
        self.assertEqual(summary['n_completed'],3)
        self.assertEqual(summary['synchronized_step_total_ms'],40)

    def test_idle_time_counts_in_makespan(self):
        clock=Clock(); engine=FakeEngine(clock,max_batch=2)
        rows,_,summary=replay(engine,[TraceRequest('a','x',0,10,8),TraceRequest('b','x',100,10,8)],
            params(),prepare,clock=clock,sleep=clock.sleep)
        self.assertAlmostEqual(summary['wall_ms'],120)
        self.assertAlmostEqual(summary['tok_per_s'],16/.12)
        self.assertEqual(rows[1]['first_dispatch_ms'],100)

    def test_preparation_failures_preserve_identity(self):
        clock=Clock(); engine=FakeEngine(clock,max_batch=2)
        def fail(r):
            if r.uid=='bad':raise RuntimeError('injected')
            return prepare(r)
        rows,_,summary=replay(engine,[TraceRequest('bad','x',0,10,8),TraceRequest('ok','x',0,10,8)],
            params(),fail,clock=clock,sleep=clock.sleep)
        self.assertEqual([(r['request_uid'],r['status']) for r in rows],[('bad','failed'),('ok','done')])
        self.assertFalse(summary['valid_for_comparison'])

    def test_incomplete_output_is_not_a_success(self):
        clock=Clock();engine=FakeEngine(clock)
        engine._make_output=lambda req:NS(output_ids=Values([]))
        rows,_,summary=replay(engine,[TraceRequest('a','x',0,10,8)],params(),prepare,
            clock=clock,sleep=clock.sleep)
        self.assertFalse(summary['valid_for_comparison'])
        self.assertEqual(rows[0]['status'],'failed')

    def test_rejection_and_timeout_are_invalid_and_release_canvases(self):
        for waiting,timeout in [(1,10),(10,.005)]:
            clock=Clock();engine=FakeEngine(clock,max_batch=1,max_waiting=waiting)
            rows,_,summary=replay(engine,[TraceRequest(str(i),'x',0,10,8) for i in range(3)],
                params(),prepare,clock=clock,sleep=clock.sleep,timeout_s=timeout)
            self.assertFalse(summary['valid_for_comparison'])
            self.assertEqual(len(rows),3)
            self.assertFalse(engine._scheduler.has_work())
            self.assertGreater(summary['n_rejected']+summary['n_timeout'],0)

    def test_bands_and_sjf(self):
        def requests(policy):
            s=ShapeShiftScheduler(max_batch=3,shape_policy=policy,clock=lambda:0,bucket_width=256)
            for n in [800,100,810]:s.admit(prepare(TraceRequest(str(n),'x',0,n-8,8)),params())
            return [r.req_id for r in s.schedule()]
        self.assertEqual(requests('static_bucket_fifo'),[0,2])
        self.assertEqual(requests('sjf_step'),[1,0,2])
        self.assertEqual(requests('sjf_step'),requests('length_bucket'))

    def test_invalid_trace_and_explicit_constant_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.csv'
            fields=['source','request_uid','prompt_len','gen_len','arrival_ms']
            def write(rows):
                with path.open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(fields);w.writerows(rows)
            write([['x','a',10,128,3],['x','b',20,128,2]])
            with self.assertRaises(ValueError):load_trace(path)
            fixed=load_trace(path,arrival_mode='constant',arrival_gap_ms=7)
            self.assertEqual([r.arrival_ms for r in fixed],[0,7])
            write([['x','a',10,128,0],['x','a',20,128,1]])
            with self.assertRaises(ValueError):load_trace(path)
            write([['x','a',10,64,0]])
            with self.assertRaises(ValueError):load_trace(path)
            write([['x','a',10,128,'nan']])
            with self.assertRaises(ValueError):load_trace(path)

    def test_matrix_has_exactly_36_unique_paired_runs(self):
        root=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('matrix',root/'tools/run_small_matrix.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        config=json.loads((root/'configs/small_matrix.json').read_text())
        plan=module.build_plan(config,'checkpoint','revision',Path('/tmp/pilot'))
        self.assertEqual(len(plan),36)
        self.assertEqual(len({r['run_id'] for r in plan}),36)
        for workload in ['alpaca','sharegpt','openorca']:
            for seed in [13,17,23]:
                runs=[r for r in plan if r['workload']==workload and r['seed']==seed]
                self.assertEqual({r['policy'] for r in runs},set(config['policies']))
                self.assertTrue(all('--arrival_mode' in r['argv'] and 'timestamps' in r['argv'] for r in runs))
        with self.assertRaises((ValueError,FileNotFoundError)):module.preflight(plan)


if __name__=='__main__':unittest.main()
