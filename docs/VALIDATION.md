# Validation record — 2026-11-13

This records local checks of the reconstructed release, not GPU experiments.

- Editable package build/install succeeded.
- 12 tests passed: 9 scheduler tests and 3 CPU integration tests using actual
  upstream dlmserve at `dd3efe743a65e0142627e105e8d18bcc1c33d6cb`.
- Integration covered real denoising steps versus individual execution on a
  deterministic toy model, async engine completion, and an injected model error
  that must be counted as a failed request rather than a completion.
- Environment: Python 3.12.13, PyTorch 2.15.0, Transformers 4.57.6,
  Accelerate 1.15.0, prometheus-client 0.26.0; macOS CPU, no weights downloaded.
- Python compilation, benchmark `--help`, a 64-request historical synthetic
  simulation, and `git diff --check` passed.
- Added CI definitions; hosted GitHub Actions have not yet run.

A limited pattern scan covered 683 historical Git blobs and found no matches
for private-key headers, GitHub tokens, Hugging Face tokens or AWS access-key IDs.
This is not a complete secret/privacy review. Historical research notes and
machine details remain in older commits. Removed tracked logs still exist in
history. A source-only release archive excludes that history.

`historical-results-manifest.json` records SHA-256 and row counts for 53 archived
CSV files. Checksums preserve identity, not scientific validity or complete
provenance. GPU throughput, actual checkpoint output quality, cross-model
compatibility, and exact reproduction of paper figures remain unverified.

## Clean public snapshot checks

The release snapshot was exported without the previous repository's `.git`
objects. It replaces the superseded anonymous draft with the author's unchanged
camera-ready PDF and matching LaTeX source. Conference forms and bundles are
excluded. In 47 historical CSV files, 62 machine-local trace paths were replaced
with repository-relative references. Every other CSV field was compared with
the private originals and is unchanged; public checksums were regenerated.
All public text files plus extracted PDF text and metadata were scanned for
machine-home paths and common credential patterns, with no matches. These are
bounded checks, not a guarantee against every possible sensitive string.
