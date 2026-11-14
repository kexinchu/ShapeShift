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
only replaces the scheduler. The v2 default allows different block/step progress; compatibility still requires
equal block size, steps per block, generation length, and sampling settings.
Nonzero temperature is rejected because upstream does not preserve individual
request RNGs through batched execution. Pending cancelled futures are removed.

## GPU benchmark (v2)

Use [SMALL_MATRIX.md](SMALL_MATRIX.md) for the current timestamp-replay CLI and
36-run pilot. The v1 `--policies`/`--tag` commands are retired. The corrected
runner uses one policy per process and requires explicit checkpoint revision,
trace and exclusive output directory. The source and results in `paper/` and
`cache-sparse-attention/results/paper/` are not changed by this protocol update.

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
