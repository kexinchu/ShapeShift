"""Read-only provenance audit; emit JSON without importing plotting dependencies."""
import ast
import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'cache-sparse-attention/results/paper'


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    summaries = [(p, row) for p in sorted(RESULTS.glob('*summary.csv')) for row in read(p)]
    counts = {key: dict(collections.Counter(row.get(key, 'not_recorded') for _, row in summaries))
              for key in ['steps', 'seed', 'n_requests', 'arrival_gap_ms', 'policy']}
    literals = {}
    for node in ast.parse((ROOT/'paper/plot_fullstep_llada8b.py').read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                literals[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    matches = []
    for series in ['FCFS', 'SHAPESHIFT_SHAPE', 'SHAPESHIFT_AGING', 'SHAPESHIFT']:
        for workload, (tput, p95) in zip(literals['WORKLOADS'], literals[series]):
            matches.append({'figure_series': series, 'figure_workload': workload,
                'figure_tok_s': tput, 'figure_p95_s': p95,
                'rounded_value_matches': [
                    {'file': str(p.relative_to(ROOT)), 'policy': r['policy']}
                    for p, r in summaries if p.name.startswith('dlmserve_32step_')
                    and round(float(r['tok_per_s']), 2) == tput
                    and round(float(r['p95_latency_ms'])/1000, 1) == p95]})
    print(json.dumps({'audited_base_commit': '867ac04f15f57025b1a396529492f3369b54a438',
        'summary_files': len({p for p, _ in summaries}), 'summary_rows': len(summaries),
        'summary_field_counts': counts, 'fullstep_rounded_value_matches': matches,
        'dinfer_rows': [{'file': str(p.relative_to(ROOT)), 'policy': r['policy'],
                        'tok_per_s': r['tok_per_s'], 'speedup': r.get('speedup_vs_fifo_req_s')}
                       for p, r in summaries if p.name.startswith('dinfer_')],
        'interpretation': 'Numeric correspondence is not proof of original provenance; unmatched values require source records.'}, indent=2))


if __name__ == '__main__':
    main()
