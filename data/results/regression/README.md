# Saved Arrow regression results

`arrow_oracle_pass/` is the primary analysis, with 55 oracle-passing tasks.
`arrow/` is the extended 59-task sensitivity set, adding four oracle-fail tasks
with recorded agent success. `arrow_errors_as_failures/` counts two execution
errors as failures for that same 59-task set. Each uses its preserved nested
outer/inner split assignments; the 55- and 59-task analyses have different folds.

`arrow_oracle_pass_grouped/` uses the identical 55-task input dataset, with
50 groups defined by repository and benchmark base commit. Both nested levels
hold each group together. This sensitivity has its own preserved splits and
refitted models; the original three result directories remain unchanged.

Verified's final run and all-report denominator sensitivity remain in the authorized private archive.
