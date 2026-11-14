"""Validate paired run identities and output hashes before computing ratios."""
import csv
import json
import statistics
from pathlib import Path


def summarize(root: Path):
    plan=json.loads((root/'plan.json').read_text())
    pairs={}
    for run in plan:
        directory=root/run['run_id']
        manifest=json.loads((directory/'manifest.json').read_text())
        summary=json.loads((directory/'summary.json').read_text())
        if manifest['status']!='complete' or not summary['valid_for_comparison']:
            raise ValueError('incomplete or failed run in matrix')
        if manifest['trace_sha256']!=run['trace_sha256'] or manifest['git_commit']!=run['git_commit']:
            raise ValueError('run identity mismatch')
        with (directory/'requests.csv').open(newline='') as f:
            rows=list(csv.DictReader(f))
        if len(rows)!=run['selected_requests'] or any(r['status']!='done' for r in rows):
            raise ValueError('request count/status mismatch')
        key=(run['workload'],run['seed'])
        if run['policy'] in pairs.setdefault(key,{}):
            raise ValueError('duplicate policy in pair')
        pairs[key][run['policy']] = (manifest,summary,rows)
    comparisons=[]
    for (workload,seed),policies in pairs.items():
        if 'fifo' not in policies:raise ValueError('missing FIFO pair')
        bm,bs,br=policies['fifo']
        identity=lambda rows:[(r['request_uid'],r['source'],r['scheduled_arrival_ms'],r['prompt_len'],r['gen_len']) for r in rows]
        hashes=lambda rows:[r['output_sha256'] for r in rows]
        def config(m):
            return {k:v for k,v in m['config'].items() if k not in {'policy','output_dir'}}
        for policy,(m,s,r) in policies.items():
            if (identity(r)!=identity(br) or m['input_token_sha256']!=bm['input_token_sha256'] or
                    config(m)!=config(bm) or m['upstream_source_sha256']!=bm['upstream_source_sha256'] or
                    m.get('model_metadata_sha256')!=bm.get('model_metadata_sha256') or
                    m.get('versions')!=bm.get('versions') or m.get('gpu')!=bm.get('gpu')):
                raise ValueError('paired inputs/configuration/dependency differ')
            equivalent=hashes(r)==hashes(br) and all(hashes(r))
            comparisons.append(dict(workload=workload,seed=seed,policy=policy,
                output_hashes_equal=bool(equivalent),
                throughput_speedup=s['tok_per_s']/bs['tok_per_s'] if equivalent else None,
                p95_ratio=s['p95_latency_ms']/bs['p95_latency_ms'] if equivalent else None))
    groups={}
    for row in comparisons:groups.setdefault((row['workload'],row['policy']),[]).append(row)
    aggregates=[]
    for (workload,policy),rows in groups.items():
        valid=all(r['output_hashes_equal'] for r in rows)
        values=[r['throughput_speedup'] for r in rows] if valid else []
        aggregates.append(dict(workload=workload,policy=policy,n_seeds=len(rows),
            valid_for_comparison=valid,mean_paired_speedup=statistics.mean(values) if values else None,
            std_paired_speedup=statistics.stdev(values) if len(values)>1 else None,
            mean_paired_p95_ratio=statistics.mean(r['p95_ratio'] for r in rows) if valid else None))
    return dict(comparisons=comparisons,aggregates=aggregates,
        valid_for_comparison=all(r['output_hashes_equal'] for r in comparisons),
        scope='Output hashes check deterministic agreement only; length-proxy inputs do not establish task quality.')
