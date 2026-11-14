"""Compatibility entry point for the v2 timestamp replay benchmark.

CLI changed intentionally: one --policy per process, explicit --trace and
--output_dir. See docs/SMALL_MATRIX.md. No implicit sampling or shuffling.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shapeshift.benchmark import main
from shapeshift.replay import load_trace


def load_lengths(path, *, source, n_requests, min_total_len, max_total_len, seed):
    """Legacy API: preserve order; seed no longer changes the workload."""
    del seed
    trace = load_trace(path, source=source, arrival_mode='all_ready', limit=n_requests)
    if any(not min_total_len <= r.prompt_len+r.gen_len <= max_total_len for r in trace):
        raise ValueError('trace length outside requested bounds; no implicit filtering')
    return [r.prompt_len for r in trace]


if __name__ == '__main__':
    raise SystemExit(main())
