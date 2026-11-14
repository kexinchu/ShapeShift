# ShapeShift camera-ready paper

This directory is the single canonical paper entry point for this release.

- `ShapeShift-camera-ready.pdf`: unchanged camera-ready PDF supplied by the author.
- `main.tex`, `references.bib`, `figures/`: source and referenced figures extracted
  from the author's camera-ready source package.
- `llncs.cls`, `splncs04.bst`: original LNCS templates; retain their notices.
- `plot_*.py`: author-supplied plotting scripts and shared helpers.
- `manifest.json`: original release checksums of the canonical source, figures, and PDF.
- `archive/`: clearly superseded paper materials.
- `../docs/LOCAL_FILE_BACKUP.json`: local-file mapping and checksums, including extra figures.

The camera-ready package's main source, bibliography and templates were checked
byte-for-byte against the author's current working directory and matched. Its
PDF matched the current working PDF as well. The earlier anonymous AAAI draft
is superseded and retained in `archive/aaai-draft/` for recovery.
No paragraphs or experimental numbers were merged from it.

Conference submission bundles, signed license/copyright forms, spreadsheet
forms, presentation drafts, editor files and build logs are excluded. The
published author name, institutional affiliation and academic correspondence
address remain intentionally in the manuscript.

## Build

Install a TeX distribution with LNCS dependencies, then from this directory:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The generated `main.pdf` is ignored. The canonical supplied PDF is not overwritten.
Different TeX versions may produce byte-different PDFs.

## Plots and evidence

```bash
python -m pip install matplotlib numpy
python paper/plot_all_figures.py  # from repository root
```

Plots contain transcribed historical result tables. Running this command
regenerates selected graphics, not measurements or every figure. The paper's
reported speedups and cross-model results came from the unavailable original
runtime. They are not evidence for the reconstructed implementation.
