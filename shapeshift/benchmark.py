"""Reproducible LLaDA step-boundary benchmark (one policy per process)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
import time
from pathlib import Path

from .replay import load_trace, replay
from .scheduler import ShapeShiftScheduler


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--source')
    p.add_argument('--model', required=True, help='Local LLaDA checkpoint/snapshot directory')
    p.add_argument('--model_revision', required=True, help='Recorded checkpoint revision; never inferred')
    p.add_argument('--policy', choices=sorted(ShapeShiftScheduler.POLICIES | {'upstream_default'}), default='shapeshift')
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--input_mode', choices=['length_proxy', 'prompt'], default='length_proxy')
    p.add_argument('--arrival_mode', choices=['timestamps', 'constant', 'all_ready'], default='timestamps')
    p.add_argument('--arrival_gap_ms', type=float, default=0)
    p.add_argument('--spacing', type=float, default=1)
    p.add_argument('--limit', type=int)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--gen_length', type=int, default=128)
    p.add_argument('--max_batch', type=int, default=8)
    p.add_argument('--max_waiting', type=int, default=256)
    p.add_argument('--shape_slo_ms', type=float, default=5000)
    p.add_argument('--active_pool_factor', type=int, default=4)
    p.add_argument('--padding_ratio', type=float, default=1.25)
    p.add_argument('--bucket_width', type=int, default=256)
    p.add_argument('--compatibility_mode', choices=['strict', 'progress_independent'], default='progress_independent')
    p.add_argument('--dtype', choices=['bf16', 'int4'], default='bf16')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--warmup_batches', type=int, default=2)
    p.add_argument('--timeout_s', type=float, default=3600)
    p.add_argument('--validate_only', action='store_true', help='Validate trace and options without importing GPU dependencies')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if min(args.steps, args.gen_length) < 1 or args.warmup_batches < 1:
        p.error('positive steps/gen_length and at least one warmup batch required')
    import math
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        p.error('timeout must be finite and positive')
    ShapeShiftScheduler(max_batch=args.max_batch, max_waiting=args.max_waiting,
        shape_slo_ms=args.shape_slo_ms, active_pool_factor=args.active_pool_factor,
        padding_ratio=args.padding_ratio, bucket_width=args.bucket_width,
        compatibility_mode=args.compatibility_mode)
    trace = load_trace(args.trace, source=args.source, gen_length=args.gen_length,
        arrival_mode=args.arrival_mode, arrival_gap_ms=args.arrival_gap_ms,
        limit=args.limit, spacing=args.spacing)
    if args.input_mode == 'prompt' and any(not r.prompt for r in trace):
        p.error('prompt mode needs prompt text on every selected row')
    if args.validate_only:
        print(json.dumps(dict(requests=len(trace), policy=args.policy,
                              last_arrival_ms=trace[-1].arrival_ms)))
        return 0
    if not Path(args.model).is_dir():
        p.error('use a local pinned LLaDA checkpoint directory')
    import torch
    from dlmserve.denoise_loop import init_state
    from dlmserve.sampler import SamplingParams
    from .engine import Engine
    if not torch.cuda.is_available() or not args.device.startswith('cuda'):
        p.error('CUDA GPU required; no synthetic performance result will be emitted')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Exclusive run directories prevent mixing seeds or overwriting evidence.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update(trace=args.trace.name, model=Path(args.model).name, output_dir=args.output_dir.name)
    config.pop('validate_only')
    code_root = Path(__file__).resolve().parents[1]
    def git(*command):
        try:
            return subprocess.check_output(['git', '-C', str(code_root), *command], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return 'unavailable'
    versions = {}
    for name in ['torch', 'transformers', 'accelerate', 'dlmserve']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not-installed-as-distribution'
    manifest = dict(protocol='shapeshift-replay-v2', status='running', config=config,
        git_commit=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain')),
        trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        selected_trace=[dict(uid=r.uid, source=r.source, arrival_ms=r.arrival_ms,
                             prompt_len=r.prompt_len, gen_len=r.gen_len) for r in trace],
        versions=versions, python=platform.python_version(), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device),
        gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
        model_revision_source='user supplied; checkpoint metadata hashes recorded below',
        model_metadata_sha256={x.name: hashlib.sha256(x.read_bytes()).hexdigest()
            for x in Path(args.model).iterdir() if x.is_file() and
            (x.name in {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'}
             or x.name.endswith('.index.json'))},
        execution_scope='LLaDA deterministic decoding; length_proxy does not measure task quality',
        upstream_source_sha256={})
    import dlmserve.denoise_loop, dlmserve.engine, dlmserve.scheduler
    for module in [dlmserve.denoise_loop, dlmserve.engine, dlmserve.scheduler]:
        manifest['upstream_source_sha256'][module.__name__] = hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
    write_json(args.output_dir/'manifest.json', manifest)
    try:
        engine = Engine(model_id=args.model, dtype=args.dtype, device=args.device,
            max_batch=args.max_batch, scheduler_policy=args.policy,
            scheduler_slo_ms=args.shape_slo_ms, scheduler_active_pool_factor=args.active_pool_factor,
            scheduler_padding_ratio=args.padding_ratio, max_waiting=args.max_waiting,
            bucket_width=args.bucket_width, compatibility_mode=args.compatibility_mode)
        params = SamplingParams(num_denoising_steps=args.steps, gen_length=args.gen_length,
            block_length=args.gen_length, temperature=0, seed=args.seed)
        token_ids = {}
        for r in trace:
            if args.input_mode == 'prompt':
                ids = engine.tokenizer(r.prompt, add_special_tokens=False)['input_ids']
            else:
                word_ids = engine.tokenizer(' the', add_special_tokens=False)['input_ids']
                special = set(getattr(engine.tokenizer, 'all_special_ids', [])) | {engine._loaded.mask_id}
                token = next((t for t in word_ids if t not in special), None)
                if token is None:
                    raise ValueError('could not choose a non-special synthetic token')
                ids = [token]*r.prompt_len
            if len(ids) != r.prompt_len:
                raise ValueError('tokenized prompt length differs from trace; rebuild trace for this tokenizer')
            token_ids[r.uid] = torch.tensor([ids], dtype=torch.long)  # CPU until admission
        manifest['input_token_sha256'] = hashlib.sha256(json.dumps(
            [(r.uid, token_ids[r.uid].tolist()) for r in trace], separators=(',', ':')).encode()).hexdigest()
        def prepare(r):
            return init_state(token_ids[r.uid].to(device), params, mask_id=engine._loaded.mask_id)
        sync = lambda: torch.cuda.synchronize(device)
        from dataclasses import replace
        # Distinct lengths spanning the input; all warmup rows arrive at zero.
        ordered = sorted(trace, key=lambda r: r.prompt_len)
        count = min(args.max_batch, len(ordered))
        warm_trace = [replace(ordered[i*(len(ordered)-1)//max(1,count-1)], arrival_ms=0)
                      for i in range(count)]
        for _ in range(args.warmup_batches):
            _, _, warm_summary = replay(engine, warm_trace, params, prepare,
                synchronize=sync, timeout_s=args.timeout_s)
            if not warm_summary['valid_for_comparison']:
                raise RuntimeError('warmup did not complete')
            engine.reset_scheduler(args.policy)
        sync()
        torch.cuda.reset_peak_memory_stats(device)
        rows, steps, summary = replay(engine, trace, params, prepare,
            synchronize=sync, timeout_s=args.timeout_s)
        summary.update(policy=args.policy, seed=args.seed, protocol=manifest['protocol'],
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       input_mode=args.input_mode)
        start_write = time.perf_counter()
        with (args.output_dir/'requests.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        with (args.output_dir/'steps.jsonl').open('w') as f:
            for row in steps:
                f.write(json.dumps(row, allow_nan=False)+'\n')
        write_json(args.output_dir/'summary.json', summary)
        manifest.update(status='complete' if summary['valid_for_comparison'] else 'invalid',
                        artifact_write_ms=(time.perf_counter()-start_write)*1000,
                        measured_step_count=len(steps))
        write_json(args.output_dir/'manifest.json', manifest)
        print(json.dumps(summary, allow_nan=False))
        return 0 if summary['valid_for_comparison'] else 2
    except Exception as exc:
        manifest.update(status='failed', error_type=type(exc).__name__)
        write_json(args.output_dir/'manifest.json', manifest)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
