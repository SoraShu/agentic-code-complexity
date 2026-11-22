# Final study results

`regression/` contains Arrow's final reference-feature results and the two retained scientific sensitivities. `judge/` contains Pro predictions from the shared three-step rubric. `published-summary.json` contains aggregate values reported in the thesis, including Verified, without private per-task outcomes. Older feature revisions and discarded rubric experiments are not included.

The main Arrow analysis is `regression/arrow_oracle_pass/` (55 tasks).
The preserved `regression/arrow/` and `regression/arrow_errors_as_failures/`
results use the extended 59-task set and are sensitivity analyses. The stored
directory names and historical run metadata are retained; the reporting role is
specified by `primary_arrow_analysis` in the published summary. Pro's
complete-coverage sensitivity uses the 381 tasks with 13 valid outcomes.
