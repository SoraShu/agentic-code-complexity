# Results figures

Prepare `work/full` using authorized private Verified inputs, then run:

```bash
uv run --project packages/regression --locked python analysis/figures/build_figures.py --workspace work/full
uv run --project packages/regression --locked python analysis/figures/verify_figures.py --workspace work/full
uv run scripts/build_distribution.py work/full
```

Outputs go to `work/full/generated/figures/`. The main generator produces the
feature-correlation plot, regression fold plot, Judge diagnostics and repository
correlations. The distribution script produces the fifth empirical figure.
The Arrow panels use the saved 55-task oracle-pass source-only analysis.
All 19 features exclude `tests/`; Verified's features and fitted results are
unchanged. The common R² axis includes every saved fold value, including Arrow's
most negative fold.
`verify_figures.py` checks 1,231 Judge points, 20 bins, 80 fold scores, 38 feature
correlations, 21 repository subsets and the three case overview rows.

Add `--compile` to the main generator to build its four PDFs. It requires XeLaTeX,
PGFPlots, standalone and TeX Gyre fonts; TeX Live fonts are located with `kpsewhich`
when no font configuration is supplied. The distribution TeX compiles with the same
font configuration. `--output-dir` can select another ignored output directory.

The seven final TeX/PDF assets, including the two conceptual diagrams, are already
in [master-thesis/figures](https://github.com/SoraShu/master-thesis/tree/main/figures).
They are not duplicated here.
