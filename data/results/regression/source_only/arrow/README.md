# arrow: source-only

Extended sensitivity: 59 tasks, adding four Oracle-fail tasks with solver success.

All 19 features exclude `tests/`. Task IDs, targets and all nested splits match
`../../arrow/`, whose original complete-PR results remain unchanged.

| Model | Mean outer R² | Mean outer MAE |
|---|---:|---:|
| baseline | -0.021 | 0.320 |
| median | -0.421 | 0.235 |
| lasso | -0.021 | 0.320 |
| xgboost | -0.022 | 0.265 |

`dataset.json` contains the adopted features and fixed targets; `splits.json`
contains inner/outer task assignments; `oof_predictions.csv` contains held-out
predictions. Per-fold files retain selected parameters and training statistics.
`results.json` includes feature summaries; `metrics_with_median.json` adds the
matched median baseline. The [parent guide](../README.md) describes reproduction
and links the shared configuration and input/code hashes in `../metadata.json`.
