> Historical artifact. The original runtime is unavailable. The reconstructed
> runtime is in the root `shapeshift/` package; see the root README and
> `docs/REPRODUCIBILITY.md`. Archived performance numbers do not validate it.

# Curated Paper Results

This directory contains small CSV summaries used to audit the ShapeShift paper
figures and claims.

Included files are summary-level outputs only:

- `dlmserve_mainmatrix_aggregate_v1.csv`: aggregated same-protocol online
  serving results used for the main dlmserve comparison.
- `dlmserve_mainmatrix_workload_stats_v1.csv`: workload length statistics.
- `dlmserve_32step_*_summary.csv`: longer-horizon validation summaries.
- `dinfer_online_*_summary.csv`: dInfer-wrapper composition summaries.
- `dlmserve_llada_shapeshift_*_summary.csv`: LLaDA policy-ablation summaries.
- `dlmserve_model_*_shapeshift_*_summary.csv`: cross-model ShapeShift
  summaries for Dream, CoDA, and LLaDA2.0-mini.

Raw per-request rows, scheduler traces, logs, downloaded datasets, and model
checkpoints are intentionally excluded from git.

Public copies replace machine-local `scheduler_trace_csv` paths with relative
references. The referenced raw traces are not included. Experimental values are
unchanged. `docs/historical-results-manifest.json` hashes these sanitized copies.
