# arrow_errors_as_failures

Execution errors counted as failures; 59 tasks.

These are preserved complete-PR features/results, including PR tests. The
adopted [source-only sensitivity](../source_only/arrow_errors_as_failures/README.md)
retains these tasks, targets and splits.

- `dataset.json`: feature names, task rows and empirical complexities.
- `splits.json`: preserved outer and inner task-ID splits.
- `oof_predictions.csv`: held-out predictions and outer fold assignments.
- `results.json`: final aggregate metrics, feature summaries and fitted-fold details.
- `lasso_fold_*.json`, `xgboost_fold_*.json`: selected parameters and training preprocessing state.

The median baseline is derived exclusively from each outer training fold by `report.with_median`. No models need to be refitted to inspect these results.
