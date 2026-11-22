# arrow_errors_as_failures: source-only

Extended sensitivity: 59 tasks, counting two process errors as failures.

All 19 features exclude `tests/`. Task IDs, targets and all nested splits match
`../../arrow_errors_as_failures/`, whose original complete-PR results remain unchanged.

| Model | Mean outer R² | Mean outer MAE |
|---|---:|---:|
| baseline | -0.017 | 0.318 |
| median | -0.432 | 0.238 |
| lasso | -0.017 | 0.318 |
| xgboost | -0.077 | 0.274 |

`dataset.json` contains the adopted features and fixed targets; `splits.json`
contains inner/outer task assignments; `oof_predictions.csv` contains held-out
predictions. Per-fold files retain selected parameters and training statistics.
`results.json` includes feature summaries; `metrics_with_median.json` adds the
matched median baseline. The [parent guide](../README.md) describes reproduction
and links the shared configuration and input/code hashes in `../metadata.json`.
