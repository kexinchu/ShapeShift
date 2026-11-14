# Contributing

Use Python 3.11+. Install with `python -m pip install -e .` and run
`python -m unittest discover -s tests -v` before submitting changes.
Runtime integration tests need the pinned upstream and its dependencies.

For scheduler changes cover compactness, overdue outliers, per-request state,
sampling compatibility, completion, cancellation and admission limits.
Describe any changed default or limitation. Do not claim GPU gains from CPU
simulation or toy-model tests. Attach matched-baseline configurations and
measurement provenance when proposing performance claims.

Do not submit credentials, private prompts, weights, downloaded datasets,
local logs or unpublished conference paperwork.
