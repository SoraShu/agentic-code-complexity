# Arrow source-only features

`pr_metrics.jsonl` contains the adopted features for all 73 constructed tasks.
All 19 features exclude `tests/` paths. It also records retained structural
measurements, retained per-file addition/deletion counts, excluded test paths
and reference hashes. `change-counts.json` preserves the derived complete-PR
per-file counts, so both scopes can be checked without distributing raw diffs.

The main cohort contains 55 oracle-pass tasks with 548 valid outcomes; the
extended cohort contains 59 tasks with 588. Six main tasks have no retained
Python files: PRs 602, 679, 761, 1076, 1164 and 1236. PRs 1076 and 1236 have
test-only changes, so all six size/directory features are also zero. These
tasks remain in the cohort under the original empty/missing conventions.

PR 700's unavailable file is a test file; its retained source measurements are
available. PR 798 still has unavailable source files, so its 13 structural
features remain missing and are imputed within training folds. Both occur only
in the extended cohort; their historical Oracle and solver outcomes are unchanged.
