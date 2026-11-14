> Historical artifact. The original runtime is unavailable. The reconstructed
> runtime is in the root `shapeshift/` package; see the root README and
> `docs/REPRODUCIBILITY.md`. Archived performance numbers do not validate it.

# ShapeShift Experiment Scripts

This directory keeps the scripts needed to construct traces, run the online
scheduler benchmark, and aggregate paper results.

Primary entry points:

- `run_dlmserve_online_trace_benchmark.py`: same-protocol online dlmserve
  benchmark with FCFS, shape-only, aging, ShapeShift, and offline sorted
  policies.
- `run_dinfer_online_trace_benchmark.py`: dInfer-wrapper composition benchmark.
- `build_serving_workload_traces.py`: ShareGPT/HotpotQA-style workload traces.
- `build_instruction_longctx_traces.py`: instruction and OpenOrca/FLAN traces.
- `build_dinfer_trace_dataset.py`: dInfer-compatible trace conversion.
- `build_dlmserve_trace_from_rows.py`: dlmserve trace conversion helper.
- `build_scheduler_stress_traces.py`: synthetic stress traces.
- `aggregate_dlmserve_mainmatrix.py`: aggregate same-protocol main matrix runs.
- `plot_main_online_policy_bars.py`: regenerate main online-policy bar charts
  from result summaries.

Scripts resolve the repository root from their own location. Supply model,
dataset and trace paths through their CLI options. Historical optional probes
may require dependencies that were not preserved; see the root reproduction guide.
