# Code and paper backup / local cleanup — 2026-11-14

The backup scope is **code and paper only**, in the existing
[kexinchu/ShapeShift](https://github.com/kexinchu/ShapeShift) repository.
No separate repository is created.

## Differences and preservation

- Canonical camera-ready LaTeX, bibliography, LNCS templates and PDF match the
  local working copies byte-for-byte. The supplied final PDF is identical too.
- The current plotting launcher changes only child-script path resolution.
  Its original local version is preserved in `paper/archive/local-plot-entry/`.
- All local figure assets, including the editable architecture diagram and
  supplementary plots, are preserved in `paper/figures/`. Added plots remain
  historical/unvalidated artifacts; they are not new experimental evidence.
- The superseded anonymous manuscript and its plotting code are preserved under
  `paper/archive/aaai-draft/`. The camera-ready is still the sole current paper.
- The current code and experimental harness are versioned normally. Earlier
  reconstructed code is recoverable from public Git history. One older analysis
  script differs only in trailing whitespace; no logic is missing.
- `LOCAL_FILE_BACKUP.json` maps the preserved local paper/source files to their
  repository paths and SHA-256 checksums. Old private Git history is excluded.
- Dependencies remain pinned in `pyproject.toml`; disposable external checkouts
  and Python environments can be rebuilt from the documented setup.

## Explicitly not backed up

Per the author's scope decision: conference/submission forms and signed files,
submission ZIP packages, presentation slides/scripts/build records, private
notes, raw logs and old private Git history are excluded. OS metadata, bytecode,
LaTeX build intermediates and virtual environments are also excluded.

Deleting the entire former local workspace also deletes these excluded local
materials (including any unuploaded local private-archive staging directory).
If any must be retained, move them to separate personal storage before deleting
that workspace. This repository restores the authorized code/paper scope, not
every former local file.

## Recover

```bash
git clone git@github.com:kexinchu/ShapeShift.git
```

Current code is at the repository root; camera-ready paper and all figures are
in `paper/`; the superseded draft is in `paper/archive/aaai-draft/`.
No tests, paper builds or GPU experiments were run for this backup operation;
verification consists of Git/file differences and remote checksum comparison.
