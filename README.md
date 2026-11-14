# ShapeShift

**Active-shape scheduling for diffusion language model serving.** ShapeShift
selects a new batch at each denoising step using nearest-length companions,
a padding-ratio budget, and service-gap aging.

> **Reconstructed research release.** The original modified runtime was lost.
> `shapeshift/` is a new implementation of the paper algorithm, integrated with
> a pinned upstream dlmserve revision. Archived CSVs and paper figures were
> produced by the original implementation, **not this reconstruction**.
> No reproduced GPU speedup or quality claim is made for this release.

## Quick start (CPU, no model download)

Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python cache-sparse-attention/code/simulate_shape_fragmentation_scheduler.py \
  --synthetic_profile longctx_mixture --n_requests 64 \
  --max_batches 4 --tolerances 0 --seeds 13 \
  --out /tmp/shapeshift-simulation.csv
```

The historical simulator is an illustrative cost model, not the runtime or a
GPU benchmark. Unit tests exercise the reconstructed scheduler directly.

## GPU integration

The adapter uses [dlmserve](https://github.com/iOptimizeThings/dlmserve) at
`dd3efe743a65e0142627e105e8d18bcc1c33d6cb`. Install on a CUDA-capable Linux host:

```bash
python -m pip install -e '.[runtime]'
python cache-sparse-attention/code/run_dlmserve_online_trace_benchmark.py --help
```

See [reproduction instructions](docs/REPRODUCIBILITY.md) for an executable
benchmark example, trace schema, limitations, and evidence requirements.
The reconstructed adapter supports LLaDA deterministic decoding; the old
Dream, CoDA, and LLaDA2-mini integrations have not been recovered.

## Repository contents

| Path | Purpose |
| --- | --- |
| `shapeshift/` | Installable scheduler and dlmserve engine adapter |
| `tests/` | Scheduling behavior and optional real-upstream CPU integration tests |
| `cache-sparse-attention/code/` | Historical trace builders, benchmarks, analysis |
| `cache-sparse-attention/results/paper/` | Archived, curated historical summaries |
| `paper/` | Canonical camera-ready PDF, LaTeX sources, and plotting scripts |
| `docs/` | Reproduction, provenance, and publication inventory |

The paper has one canonical public version: [camera-ready PDF](paper/ShapeShift-camera-ready.pdf)
and [LaTeX source](paper/main.tex). See [paper provenance and build instructions](paper/README.md).
The earlier anonymous AAAI draft is not distributed. Paper results describe the
original implementation; this repository's reconstructed runtime is a separate
artifact. Some plot scripts transcribe table values rather than regenerate experiments.

## Public artifact policy

Keep source, tests, dependency revisions, configurations, small sanitized traces,
result provenance, and figure-generation code in Git. Keep weights, downloaded
datasets, secrets, machine logs, agent notes, and conference paperwork out.
See [the detailed inventory (中文)](docs/PUBLICATION_PLAN.zh-CN.md).

## License and contributions

Original project code uses the existing [MIT license](LICENSE).
[Third-party notices](THIRD_PARTY_NOTICES.md) identify external components and
paper templates. See [CONTRIBUTING.md](CONTRIBUTING.md) for validation requirements.
