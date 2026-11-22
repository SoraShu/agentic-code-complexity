# Features and regression

The scientific implementation preserves the 19 features, nested outer/inner folds,
Lasso/XGBoost search grids, training-fold imputation/scaling and constant baselines.
The root preparation command copies these files beside their inputs in `work/`.

- `extract_reference_features.py`: statically apply reference patches and compute independent MI, Halstead and LLOC measures. Inputs are obtained through `scripts/fetch_reference_sources.py`.
- `evaluate.py`: Arrow nested evaluation; the default is the 55-task `arrow_oracle_pass` main analysis. `arrow` and `arrow_errors_as_failures` are extended 59-task sensitivities.
- `evaluate_verified_outcomes.py`: adopted Verified Boolean-only targets, preserving its saved outer/inner folds and the all-report sensitivity comparator.
- `report.py`: audit saved predictions and produce final performance tables, without historical revision comparisons.

The base-commit-grouped sensitivity is run from the repository root with:

```bash
uv run scripts/run_arrow_grouped.py --output work/arrow-grouped-new --workers 8
```

It reads the main analysis's saved 55-task dataset directly and groups tasks by
repository and benchmark base commit (50 groups). Both outer and inner splitters
use shuffled five-fold `GroupKFold` with seed 42. The shared `evaluate.py` fitting
path retains the full 100-alpha Lasso grid, 972-candidate XGBoost grid,
training-fold preprocessing, losses and estimator settings. It preserves every
task and target, and evaluates matched mean and median baselines. Group IDs,
folds, OOF predictions, selected parameters, input/code hashes and versions are
saved alongside the results. Original task-level results are not overwritten.
The grouping and its scope are described in
[the grouped-result guide](../../data/results/regression/arrow_oracle_pass_grouped/README.md).

To report saved results, after workspace preparation:

```bash
uv run --project packages/regression --locked python packages/regression/report.py \
  work/public/experiments/regression/results/reference-features-2026-11-09 \
  --output work/arrow-tables
```

Optional new fits are separate from verification and were not run during packaging:

```bash
cd work/public/experiments/regression
../../../../packages/regression/.venv/bin/python evaluate.py --workers 8 \
  --feature-root inputs-reference --splits-from results/reference-features-2026-11-09 \
  --output new-arrow-results --scenarios arrow_oracle_pass
```

To include the preserved sensitivity definitions in a new fitting run, explicitly
add `arrow arrow_errors_as_failures` to `--scenarios`. Reporting saved results
does not fit models.

For Verified, prepare `work/full` with authorized data, change to its regression
directory and run `evaluate_verified_outcomes.py --previous
results/reference-features-2026-11-09 --output new-verified-results --workers 8`.
Use a new output directory. The `verified` scenario in `evaluate.py` preserves the
all-report denominator for the sensitivity comparator, not the adopted target.

Tests cover reference reconstruction, missing metric handling, outcome policies,
fold isolation and mean/median baselines. Data-dependent tests need the prepared
private workspace; the scientific unit tests use synthetic inputs.
