# Saved Arrow regression results

The adopted results are in [`source_only/`](source_only/README.md), which
excludes `tests/` from all 19 features. The following original complete-PR
results remain unchanged for comparison outside the thesis.

`arrow_oracle_pass/` uses the 55 oracle-passing tasks.
`arrow/` is the extended 59-task sensitivity set, adding four oracle-fail tasks
with recorded agent success. `arrow_errors_as_failures/` counts two execution
errors as failures for that same 59-task set. Each uses its preserved nested
outer/inner split assignments; the 55- and 59-task analyses have different folds.

These original regressions include original PR test files in their features.
Their source-only counterparts preserve each task, outcome target and nested
split while refitting both models on the filtered features.

`arrow_oracle_pass_grouped/` uses the identical 55-task input dataset, with
50 groups defined by repository and benchmark base commit. Both nested levels
hold each group together. This sensitivity has its own preserved splits and
refitted models; the original three result directories remain unchanged.

Verified's final run and all-report denominator sensitivity remain in the authorized private archive.
