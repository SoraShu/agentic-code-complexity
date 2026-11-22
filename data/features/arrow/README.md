# Arrow features

`pr_metrics.jsonl` preserves the original complete-PR reference-version features
for all 73 constructed tasks, including PR test files. Adopted source-only
features are in [`../source_only/arrow/`](../source_only/arrow/README.md).
The 55-task main and 59-task extended cohorts retain 548 and 588 valid outcomes.
In these original records, PR 700 and PR 798 have unavailable structural
features. After excluding tests, PR 700's source metrics are available, while
PR 798 still has unavailable source files. Neither original data nor outcomes
are changed by that scope choice.
