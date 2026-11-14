# Reproduction and reconstruction status

## What can be reproduced now

The standard-library unit suite validates the new selection rules without a GPU.
`tests/test_runtime.py` additionally uses the pinned upstream denoising loop with
a deterministic toy model on CPU, when runtime dependencies are installed.
Neither test establishes real-model quality, latency, or historical speedups.

The reconstructed defaults are **new choices**: padding ratio 1.25, service gap
5000 ms, active pool factor 4. They are not recovered historical settings.
`rho = batch_size * max(prompt_length + generation_length) / sum(lengths)`.
The anchor always runs; incompatible or overly padded companions remain ready.
Aging prioritizes overdue active requests but does not promise a deadline under
arbitrary load. FIFO admission bounds the active pool. FIFO, shape-only, and
aging-only policies share the same admission limit and compatibility rules.

The dlmserve adapter preserves the upstream per-request denoising state and
only replaces the scheduler. Compatibility requires equal block position,
block size, steps per block, generation length, and sampling settings.
Nonzero temperature is rejected because upstream does not preserve individual
request RNGs through batched execution. Pending cancelled futures are removed.

## GPU benchmark

Install `python -m pip install -e '.[runtime]'` on a CUDA Linux host. The extra
pins dlmserve to commit `dd3efe743a65e0142627e105e8d18bcc1c33d6cb`; this is the
reconstruction's dependency, not a claimed original experiment revision.
Download an authorized LLaDA checkpoint separately. Supply a CSV with columns
`source,request_uid,prompt_len,gen_len,total_len`, with one row per request.
An artificial example is provided in `examples/synthetic_lengths.csv`.

```bash
python cache-sparse-attention/code/run_dlmserve_online_trace_benchmark.py \
  --model /path/to/LLaDA-8B-Instruct \
  --trace examples/synthetic_lengths.csv --source synthetic \
  --n_requests 8 --max_batch 4 --steps 8 --gen_length 128 \
  --arrival_gap_ms 0 --seed 13 --shape_slo_ms 5000 \
  --active_pool_factor 4 --padding_ratio 1.25 \
  --policies fifo,length_bucket,aging_bucket,shapeshift \
  --tag reconstructed_smoke
```

The runner synthesizes prompt text to match requested lengths; it does not replay
original prompt semantics. It writes request rows, summaries, and shape traces
under `cache-sparse-attention/results/`. These generated files are ignored.
A synthetic trace run is a smoke test, not a paper workload reproduction.

For real measurements preserve: exact command and Git SHA; model and tokenizer
revisions; Python/PyTorch/Transformers/CUDA versions; GPU model/count/memory;
input-trace hash and construction command; scheduler settings; seeds; failures;
request-level timestamps and completion counts; and shape traces. Run matched
FIFO and ShapeShift settings, repeat seeds, and separately check output quality.
Never overwrite archived `results/paper/` files with reconstruction measurements.

## Historical limitations

- The old runtime and cross-model adapters are unavailable. Only LLaDA is wired
  to the reconstructed upstream adapter.
- dInfer and Fast-dLLM wrappers are historical optional scripts; exact original
  dependency revisions were not recorded and are not validated here.
- `run_fastdllm_llada_compare.py` imports missing `early-stop` helpers and is a
  legacy probe, not a supported reproduction command.
- Aggregation scripts expect original per-seed outputs that are not all present.
  Archived aggregates alone cannot recreate variance or audit every figure.
- Historical plot scripts include hardcoded table values. Regenerating plots
  does not verify the correspondence between those numbers and archived CSVs.
- Requirements for the old analysis scripts are listed in
  `cache-sparse-attention/requirements.txt`; this is not a historical lockfile.
- Workload builders accept local files via CLI. Their defaults now refer to
  `cache-sparse-attention/data/`, never the original author's home directory.

## Paper build

The canonical camera-ready source and PDF are under `paper/`. The obsolete
anonymous draft is not included. See [paper/README.md](../paper/README.md) for
build commands and exact input-file checksums. Paper measurements predate the
reconstructed runtime and must not be presented as its validation.
