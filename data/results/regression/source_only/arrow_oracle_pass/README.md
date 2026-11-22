# arrow_oracle_pass: source-only

Main analysis: 55 Harbor Oracle-pass tasks.

All 19 features exclude `tests/`. Task IDs, targets and all nested splits match
`../../arrow_oracle_pass/`, whose original complete-PR results remain unchanged.

| Model | Mean outer R² | Mean outer MAE |
|---|---:|---:|
| baseline | -0.215 | 0.326 |
| median | -0.466 | 0.227 |
| lasso | -0.215 | 0.326 |
| xgboost | -0.243 | 0.291 |

`dataset.json` contains the adopted features and fixed targets; `splits.json`
contains inner/outer task assignments; `oof_predictions.csv` contains held-out
predictions. Per-fold files retain selected parameters and training statistics.
`results.json` includes feature summaries; `metrics_with_median.json` adds the
matched median baseline. The [parent guide](../README.md) describes reproduction
and links the shared configuration and input/code hashes in `../metadata.json`.
