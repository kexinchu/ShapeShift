# v2 replay and 36-run pilot

This is a new, explicitly configured reconstruction protocol. It does not alter
the camera-ready paper or historical result files and does not reproduce their
numbers by construction. See the dated audit for the old implementation's gaps.

## Policies and execution domain

| CLI policy | Definition |
| --- | --- |
| `fifo` | Oldest admitted request anchors; fill eligible requests in arrival order. Paper-style FCFS, not upstream-default. |
| `static_bucket_fifo` | Oldest request anchors its fixed total-length band; FIFO fill within that band. Band is `(P+G-1)//bucket_width`, default width 256. |
| `sjf_step` | Shortest total-length anchor, then shortest total-length companions; tie by last service, then ID. |
| `length_bucket` | Shortest anchor; nearest-length companions; no aging/compactness. Equivalent to `sjf_step` in the same candidate domain under these definitions. |
| `aging_bucket` | Overdue oldest-service anchor, otherwise shortest; nearest-length fill, no compactness. |
| `shapeshift` | Aging plus nearest-length fill subject to padding ratio 1.25 by default. |
| `upstream_default` | Unmodified pinned upstream scheduler, including its 2B active pool. A separate diagnostic, not the paper FIFO baseline. |

Reconstructed policies use the same 4B active-pool bound and pending-queue limit.
The v2 default `progress_independent` mode allows different block indices and
step counters in a batch. The pinned upstream reads these per row; tests compare
mixed-progress execution with individual execution using a context-dependent
CPU model. Equal block length, steps-per-block, G and sampling settings remain
required. This matches the pilot's uniform settings and makes the length-based
padding budget correspond to the actual rectangle. It does not establish real
checkpoint quality or arbitrary mixed-configuration support. `strict` retains
the previous same-progress constraint for a sensitivity diagnostic.

Nonzero temperature is rejected. Defaults are reconstructed protocol choices,
not recovered historical defaults. `token_budget` remains a deprecated alias
for ratio-budget ShapeShift, not an implementation of a fixed token cap C.

## Timing contract

The runner drives the real engine's `_step_batch` directly. It is a
**step-boundary replay**, not an HTTP server/client benchmark. CSV rows remain
in input order, and explicit `arrival_ms` values define immutable external
arrivals. The first timestamp is normalized to zero; gaps are preserved.
At each completed-step boundary, all due requests are admitted before the next
schedule. An arrival during a synchronous GPU step is therefore admitted late:
its original scheduled timestamp is preserved and its admission delay recorded.
Future arrivals never become `previous actual arrival + gap`.

Rows contain request UID, scheduled arrival, actual ingress processing,
admission, first GPU-step dispatch, completion, final-output hash, status and
number of dispatches. Latency starts at **scheduled** arrival, including late
admission, initialization, queuing, service and output processing. Throughput is
completed generated tokens divided by first scheduled arrival to final terminal
event (including failures); idle gaps are included. Rejected/failed/timed-out
runs are saved but marked `valid_for_comparison=false` and exit nonzero.

Steps contain request IDs/UIDs, dispatch/finish, scheduler time, synchronized
step wall time, actual rectangular shape, padding ratio, and queue depth.
Synchronization waits for GPU work before recording finish; this is host wall
time around a synchronized step, **not kernel-only GPU time**. Inter-service
statistics use dispatch-to-dispatch intervals for each request, excluding the
first and terminal intervals; first-service wait is reported separately.
Mean step padding and work-weighted padding are both reported.

Metadata collection and output hashing have some cost inside the measurement.
File writes occur after the timed loop; scheduler CSV logging is buffered and
requires `flush_trace()` when used separately. Measure instrumentation overhead
before making a production-overhead claim. There is no independent ingress
thread; late-admission distributions are part of the reported result.

## Prepare inputs

Supply `data/alpaca.csv`, `data/sharegpt.csv`, `data/openorca.csv`, each with at
least 96 selected rows. `data/` and `runs/` are ignored by Git. These inputs are
not included; the runner refuses to execute a matrix with missing inputs.
Schema:

```text
source,request_uid,prompt_len,gen_len,total_len,arrival_ms
sharegpt,session0-turn0,128,128,256,0
sharegpt,session0-turn1,1024,128,1152,25
```

UIDs must be unique; lengths positive and consistent; timestamps finite,
nonnegative and nondecreasing. `gen_len` must match the run's G. No silent
length filtering or random sampling occurs. `--limit 96` selects the ordered
prefix; omit the limit for a full trace. Seeds do not reshuffle requests.

To convert an existing local length trace while preserving timestamps:

```bash
python tools/prepare_length_trace.py --input /path/to/lengths.csv \
  --source sharegpt_multiturn_accum --preserve-arrivals --output data/sharegpt.csv
```

If the source has no timestamps, use `--constant-gap-ms 0` (all ready) or an
explicit nonzero constant instead of `--preserve-arrivals`. Such timing is
marked `synthetic_constant`; it is not a recovered session/burst trace. The
converter strips prompt text and never overwrites the output. For prompt-mode
quality tests, prepare a separate CSV with a `prompt` column and exact lengths.

Default `length_proxy` input repeats a non-special token to create exactly the
specified prompt length. `prompt` mode uses the actual text and rejects tokenizer
length mismatches. Tokenization is outside measurement; canvas initialization is
on admission. Length-proxy output agreement does not validate task accuracy.

## Plan and preflight

Install `python -m pip install -e '.[runtime]'` on a CUDA host. Use a local,
pinned **LLaDA** checkpoint directory and explicitly record its revision. The
runner records config/tokenizer metadata hashes and actual loaded upstream
source hashes; a supplied revision label alone is not a weight-file checksum.

```bash
# Produces 36 commands without loading weights or touching a GPU.
python tools/run_small_matrix.py \
  --model /path/to/LLaDA-8B-Instruct --model-revision CHECKPOINT_REVISION

# Validate all trace schemas/counts and checkpoint directory before any run.
python tools/run_small_matrix.py --preflight \
  --model /path/to/LLaDA-8B-Instruct --model-revision CHECKPOINT_REVISION
```

`configs/small_matrix.json` defines 3 workloads × 4 policies × seeds13/17/23:
FIFO, Shape-only, Aging, Full. Each run uses the first96 requests, B8, G128,
128 steps, BF16, tau5000ms, beta1.25, active-pool4B and 2 warmup batches.
This is a pilot, not the paper's full-split protocol. Warmup uses representative
lengths including the extremes, then resets the scheduler before measurement.
The policy execution order is reproducibly randomized per seed; request order
is unchanged. GPU runtime/quality have not been validated locally.

## Execute and resume

```bash
python tools/run_small_matrix.py --execute \
  --model /path/to/LLaDA-8B-Instruct --model-revision CHECKPOINT_REVISION \
  --output-root runs/pilot-v2

python tools/run_small_matrix.py --execute --resume \
  --model /path/to/LLaDA-8B-Instruct --model-revision CHECKPOINT_REVISION \
  --output-root runs/pilot-v2
```

Execution requires committed source. All traces are checked before launching
run1. Runs are sequential, each in a fresh process. The matrix stops on a failed
run. Exclusive output directories prevent overwrites. Resume skips only runs
with matching frozen plan/code/trace fingerprints and successful manifests;
incomplete/mismatched directories must be preserved and a new output root used.

Each run writes `manifest.json`, `requests.csv`, `steps.jsonl`, `summary.json`.
Local logs may contain local paths and should not be published without review.
The matrix writes `paired-summary.json`: paired input IDs/timestamps, token
hashes, configs and upstream hashes must match FIFO. If any final output hash
differs, the paired speedup is withheld and the matrix returns nonzero. Hash
agreement verifies deterministic agreement, not task quality; real-model
numerical differences still require investigation.

```bash
python tools/summarize_small_matrix.py runs/pilot-v2
```

Strong baselines can be added to a separate copy of the config via
`static_bucket_fifo`, `sjf_step`, and `upstream_default`. Do not change the frozen
36-run plan midway. The old benchmark script path now forwards to v2 and has an
intentionally changed CLI: one `--policy`, explicit `--trace`, `--model_revision`
and `--output_dir`; old `--policies`/`--tag` are not silently reinterpreted.

## Remaining experimental limits

The actual 36 GPU runs need the local checkpoint and three workload inputs.
They have not been run here. Real LLaDA quality/output equivalence, hardware
replication on RTX A6000, multi-model adapters and dInfer compactness remain
separate tasks. The historical dInfer wrapper is unchanged and must still be
labelled aging-only when interpreting its archived results.
