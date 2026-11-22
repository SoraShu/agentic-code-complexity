# Adopted source-only Arrow regression

All 19 input features exclude repository-root `tests/` paths. The original
complete-PR feature records and fitted results remain unchanged in their
existing directories. `comparison.json` and `comparison.md` compare those
results with the adopted scope; this comparison stays outside the thesis.

| Scenario | Tasks | Definition |
|---|---:|---|
| `arrow_oracle_pass/` | 55 | Main analysis, inclusion determined by Harbor Oracle pass |
| `arrow/` | 59 | Extended sensitivity including four Oracle-fail tasks with solver success |
| `arrow_errors_as_failures/` | 59 | Same extended cohort, counting two process errors as failures |
| `arrow_oracle_pass_grouped/` | 55 | Main cohort with 50 benchmark base-commit groups held together at both CV levels |

Every scenario retains its original task order, targets and inner/outer splits.
Lasso uses the full 100-alpha grid and MSE selection; XGBoost uses the full
972-candidate grid and MAE selection. Training-fold median imputation,
Lasso standardization, seed 42, five-by-five nesting and constant baselines
are unchanged. `metadata.json` records actual versions, grids, input/code
hashes and run times. No solver or Judge was called.

Each scenario stores its source-only feature matrix, splits, OOF predictions,
fold-specific selected parameters and feature summaries. `metrics_with_median.json`
adds the matched training-fold median baseline. `correlations.json` contains
Pearson/Spearman coefficients and available-pair counts for all 19 features in
the main Arrow and Verified cohorts, without imputation. Verified's feature
vectors and saved regression results are unchanged; its correlations and
prediction scores are verified with authorized private outcomes.

Reproduce from the repository root:

```bash
uv run scripts/run_source_only.py --output work/source-only-new --workers 8
uv run scripts/prepare_workspace.py --destination work/public-new
uv run scripts/verify_results.py work/public-new
```

The first command fits models. The other two use saved outputs without fitting.
To also verify Verified, prepare a workspace with `--private-root` as described
in the root README. The original complete-PR results are copied separately so
verification can check that all task metadata and nested splits are preserved.
