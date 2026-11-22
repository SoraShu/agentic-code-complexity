# Final study results

`regression/source_only/` contains Arrow's adopted reference-feature results and
retained scientific sensitivities; the original complete-PR results remain in
the other `regression/` subdirectories. `judge/` contains Pro predictions from
the shared three-step rubric. `published-summary.json` contains the adopted
regression and Judge aggregates, including Verified, without private per-task
outcomes. Discarded rubric experiments are not included.

The main Arrow analysis is `regression/source_only/arrow_oracle_pass/` (55 tasks).
Its sibling `arrow/` and `arrow_errors_as_failures/` results use the extended
59-task set; `arrow_oracle_pass_grouped/` holds 50 base-commit groups together.
All use source-only features and preserve their original nested splits.
The reporting role is specified by `primary_arrow_analysis` and `feature_scope`
in the published summary. Pro's
complete-coverage sensitivity uses the 381 tasks with 13 valid outcomes.
`pro_complete_traceable` changes their targets to the nine traceable reports,
keeping those task IDs and predictions fixed. `pro_constant_half` evaluates a
fixed probability of 0.50 on all 731 tasks; its correlations are JSON null.
