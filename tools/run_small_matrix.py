"""Plan, preflight and sequentially execute the 36-run reconstruction pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shapeshift.benchmark import parser as benchmark_parser
from shapeshift.replay import load_trace


def build_plan(config, model, revision, output_root, *, data_root=ROOT):
    names = [w['name'] for w in config['workloads']]
    if len(set(names)) != len(names) or any(not n or not all(c.isalnum() or c in '_-' for c in n) for n in names):
        raise ValueError('unique, filename-safe workload names required')
    policies, seeds = config['policies'], config['seeds']
    if not policies or not seeds or len(set(policies)) != len(policies) or len(set(seeds)) != len(seeds):
        raise ValueError('unique policies/seeds required')
    reserved = {'trace','source','model','model_revision','policy','output_dir','seed','validate_only'}
    if reserved & set(config['parameters']):
        raise ValueError('parameters override reserved run fields')
    plan = []
    for w in config['workloads']:
        for seed in seeds:
            # Fixed randomized order, independent of trace order, reduces order/thermal bias.
            order = list(policies)
            random.Random(seed).shuffle(order)
            for policy in order:
                run_id = f"{w['name']}-{policy}-s{seed}"
                argv = ['--trace', str((data_root/w['trace']).resolve()), '--model', model,
                        '--model_revision', revision, '--policy', policy, '--seed', str(seed),
                        '--output_dir', str((output_root/run_id).resolve())]
                if w.get('source') is not None:
                    argv += ['--source', w['source']]
                for key,value in config['parameters'].items():
                    argv += ['--'+key, str(value)]
                # Catch unsupported policy/option names even without GPU/data.
                benchmark_parser().parse_args(argv)
                plan.append(dict(run_id=run_id, workload=w['name'], policy=policy,
                    seed=seed, argv=argv, command=[sys.executable,'-m','shapeshift.benchmark',*argv]))
    return plan


def preflight(plan):
    for run in plan:
        args = benchmark_parser().parse_args(run['argv'])
        if not Path(args.model).is_dir():
            raise ValueError('local model directory missing')
        selected = load_trace(args.trace, source=args.source, gen_length=args.gen_length,
            arrival_mode=args.arrival_mode, arrival_gap_ms=args.arrival_gap_ms,
            spacing=args.spacing, limit=args.limit)
        if args.input_mode == 'prompt' and any(not r.prompt for r in selected):
            raise ValueError('prompt mode requires prompt text')
        run['trace_sha256'] = hashlib.sha256(args.trace.read_bytes()).hexdigest()
        run['selected_requests'] = len(selected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/small_matrix.json')
    p.add_argument('--model', required=True)
    p.add_argument('--model-revision', required=True)
    p.add_argument('--data-root', type=Path, default=ROOT)
    p.add_argument('--output-root', type=Path, default=ROOT/'runs/small-matrix')
    p.add_argument('--execute', action='store_true', help='Without this flag, only print the plan')
    p.add_argument('--preflight', action='store_true', help='Check all input traces and checkpoint directory without loading GPU')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    plan = build_plan(config, args.model, args.model_revision, args.output_root, data_root=args.data_root)
    if args.preflight or args.execute:
        preflight(plan)
    if not args.execute:
        print(json.dumps(dict(run_count=len(plan), input_validation='passed' if args.preflight else 'not_run', runs=plan),indent=2))
        return 0
    # Freeze the exact software identity before using an expensive GPU.
    sha = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    if dirty:
        raise ValueError('commit source changes before executing a matrix')
    args.output_root.mkdir(parents=True,exist_ok=True)
    (args.output_root/'logs').mkdir(exist_ok=True)
    for run in plan:
        run['git_commit'] = sha
        run['fingerprint'] = hashlib.sha256(json.dumps(
            {k:run[k] for k in ['argv','trace_sha256','git_commit']},sort_keys=True).encode()).hexdigest()
    plan_path = args.output_root/'plan.json'
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        if not args.resume or previous != plan:
            raise ValueError('existing matrix requires --resume with the identical frozen plan')
    else:
        plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    for run in plan:
        out = Path(benchmark_parser().parse_args(run['argv']).output_dir)
        marker = out/'matrix-record.json'
        if out.exists():
            if args.resume and marker.exists():
                old = json.loads(marker.read_text())
                summary = json.loads((out/'summary.json').read_text())
                manifest = json.loads((out/'manifest.json').read_text())
                if (old.get('fingerprint')==run['fingerprint'] and
                        summary.get('valid_for_comparison') and manifest.get('status')=='complete' and
                        manifest.get('trace_sha256')==run['trace_sha256']):
                    print('verified; skipping',run['run_id'],flush=True)
                    continue
            raise ValueError(f"incomplete or mismatched run; preserve it and choose a new output root: {run['run_id']}")
        print('running',run['run_id'],flush=True)
        with (args.output_root/'logs'/f"{run['run_id']}.log").open('w') as log:
            result = subprocess.run(run['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            print(f"stopped: {run['run_id']} failed (exit {result.returncode}); inspect local log",file=sys.stderr)
            return result.returncode
        summary = json.loads((out/'summary.json').read_text())
        if not summary['valid_for_comparison']:
            raise ValueError('invalid run cannot be included in the matrix')
        marker.write_text(json.dumps({'fingerprint':run['fingerprint']},indent=2)+'\n')
    from shapeshift.matrix_summary import summarize
    report=summarize(args.output_root)
    (args.output_root/'paired-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f"completed {len(plan)} runs; output equivalence: {report['valid_for_comparison']}")
    return 0 if report['valid_for_comparison'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
