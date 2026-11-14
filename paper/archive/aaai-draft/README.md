# Active-Shape Scheduling Paper

This folder contains the AAAI-27 ShapeShift paper draft.

## Files

- `main.tex`: LaTeX source.
- `references.bib`: bibliography.
- `main.pdf`: latest compiled PDF.
- `figures/`: paper figures.
- `plot_ablation_llada8b.py`: LLaDA-8B scheduler ablation plots.
- `plot_dinfer_wrapper.py`: dInfer-wrapper composition plots.
- `plot_fullstep_llada8b.py`: longer-horizon validation plots.
- `aaai2027.sty`, `aaai2027.bst`: AAAI-27 author-kit files used by this draft.

## Build

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The draft is compiled in anonymous submission mode through:

```latex
\usepackage[submission]{aaai2027}
```

The current PDF uses the AAAI-27 submission style. Re-check page limits after
any substantial edits.

## Figure Regeneration

```bash
python plot_ablation_llada8b.py
python plot_dinfer_wrapper.py
python plot_fullstep_llada8b.py
```

The main multi-model bars are stored as generated figure PDFs. Their underlying
audit summaries are curated in `../../results/paper/`.
