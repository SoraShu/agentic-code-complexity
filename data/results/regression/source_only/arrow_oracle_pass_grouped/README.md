# arrow_oracle_pass_grouped: source-only

Grouped sensitivity: 55 main tasks in 50 benchmark base-commit groups at both CV levels.

All 19 features exclude `tests/`. Task IDs, targets and all nested splits match
`../../arrow_oracle_pass_grouped/`, whose original complete-PR results remain unchanged.

| Model | Mean outer R² | Mean outer MAE |
|---|---:|---:|
| baseline | -0.202 | 0.320 |
| median | -0.411 | 0.226 |
| lasso | -0.202 | 0.320 |
| xgboost | -0.308 | 0.291 |

`dataset.json` contains the adopted features and fixed targets; `splits.json`
contains inner/outer task assignments; `oof_predictions.csv` contains held-out
predictions. Per-fold files retain selected parameters and training statistics.
`results.json` includes feature summaries; `metrics_with_median.json` adds the
matched median baseline. The [parent guide](../README.md) describes reproduction
and links the shared configuration and input/code hashes in `../metadata.json`.

`groups.json` retains the exact task-to-base-commit grouping used at both levels.
Outer test sizes are 13, 10, 11, 10 and 11.
