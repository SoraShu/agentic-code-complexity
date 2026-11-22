# arrow

Extended sensitivity analysis; 59 tasks. The main analysis is the 55-task
`../arrow_oracle_pass/` cohort. This extension additionally includes four
oracle-fail tasks with recorded agent success.

- `dataset.json`: feature names, task rows and empirical complexities.
- `splits.json`: preserved outer and inner task-ID splits.
- `oof_predictions.csv`: held-out predictions and outer fold assignments.
- `results.json`: final aggregate metrics, feature summaries and fitted-fold details.
- `lasso_fold_*.json`, `xgboost_fold_*.json`: selected parameters and training preprocessing state.

The median baseline is derived exclusively from each outer training fold by `report.with_median`. No models need to be refitted to inspect these results.
