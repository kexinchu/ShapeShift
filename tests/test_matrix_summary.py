import csv
import json
import tempfile
import unittest
from pathlib import Path

from shapeshift.matrix_summary import summarize


class MatrixSummaryTests(unittest.TestCase):
    def fixture(self, root, *, mismatch=False, shifted=False):
        plan=[]
        for policy,rate in [('fifo',10),('shapeshift',20)]:
            run_id='work-'+policy+'-s13'
            plan.append(dict(run_id=run_id,workload='work',policy=policy,seed=13,
                             trace_sha256='tracehash',git_commit='commit',selected_requests=1))
            out=root/run_id;out.mkdir()
            (out/'manifest.json').write_text(json.dumps(dict(status='complete',trace_sha256='tracehash',
                git_commit='commit',input_token_sha256='tokens',upstream_source_sha256={'loop':'hash'},
                config=dict(policy=policy,output_dir=run_id,seed=13,steps=128))))
            (out/'summary.json').write_text(json.dumps(dict(valid_for_comparison=True,tok_per_s=rate,p95_latency_ms=100/rate)))
            with (out/'requests.csv').open('w',newline='') as f:
                row=dict(request_uid='a',source='work',scheduled_arrival_ms='1' if shifted and policy!='fifo' else '0',
                    prompt_len='10',gen_len='128',status='done',
                    output_sha256='different' if mismatch and policy!='fifo' else 'samehash')
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        (root/'plan.json').write_text(json.dumps(plan))

    def test_paired_ratios_require_output_equivalence(self):
        for mismatch in [False,True]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);self.fixture(root,mismatch=mismatch)
                report=summarize(root)
                full=next(r for r in report['comparisons'] if r['policy']=='shapeshift')
                self.assertEqual(report['valid_for_comparison'],not mismatch)
                self.assertEqual(full['throughput_speedup'],None if mismatch else 2)

    def test_rejects_different_arrival_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root,shifted=True)
            with self.assertRaises(ValueError):summarize(root)


if __name__=='__main__':unittest.main()
