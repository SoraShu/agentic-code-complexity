# Arrow base-commit-grouped sensitivity

These are the preserved complete-PR results, including PR tests. The adopted
[source-only grouped sensitivity](../source_only/arrow_oracle_pass_grouped/README.md)
uses identical task groups, targets and nested splits.

These results retain all 55 oracle-pass tasks, 548 valid outcomes, 19 features and
targets from `../arrow_oracle_pass/dataset.json`. No solver or Judge was rerun.
Only Lasso and XGBoost are refitted, using the same complete search grids and
training-fold preprocessing as the main analysis.

## Groups and evidence

The group key is repository plus benchmark base commit from
`../../../reference-inputs.json`. There are 50 groups. The four non-singleton
groups are 383/385/387, 744/745, 1043/1044 and 1092/1093; `groups.json` gives the
complete task-to-group mapping. This rule is independent of outcomes and
feature similarity, and also holds together different changes made against
the same base version. It does not claim that all tasks in a group are duplicates.

Inspection of hash-matched public reference patches identifies related changes:

- PRs 1043/1044 implement the same Sinhala locale change. Their source changes
  differ by one docstring word, and the reference test changes are identical.
- PRs 1092/1093 implement the same Amharic locale change. The source changes
  are identical; 1093 adds an ordinal-number test.
- PRs 383/385 share the dehumanize implementation, while 385 also adds interval
  functionality. The common-base rule keeps this overlap in one group.

Generated task descriptions and grading tests can differ between related PRs;
the Sinhala tasks are not duplicate outcome records. Public patch hashes and
acquisition instructions are in `data/reference-inputs.json` and
`scripts/fetch_reference_sources.py`. Raw patches and internal construction
materials are not bundled here.

## Evaluation and results

Outer and inner CV both use `GroupKFold(n_splits=5, shuffle=True,
random_state=42)`. Hyperparameter tuning receives only outer-training data and
uses group-disjoint inner splits. Outer test sizes are 13, 10, 11, 10 and 11.
Tasks have equal weight within a fold; the reported mean and population SD
give each outer fold equal weight. The split changes fold composition, so
differences from ordinary task-level CV are not an isolated estimate of the
effect of repeated changes.

| Model | Mean outer R² | Mean outer MAE |
|---|---:|---:|
| Training-fold mean | -0.202 | 0.320 |
| Training-fold median | -0.411 | 0.226 |
| Lasso | -0.170 | 0.317 |
| XGBoost | -0.257 | 0.315 |

The median retains the lowest MAE. Original task-level results remain in their
own directories. `metrics_with_median.json` stores unrounded summaries;
`splits.json` and `oof_predictions.csv` retain task assignments and predictions.
Per-fold JSON files retain selected parameters and fitted preprocessing
statistics. `metadata.json` records versions, grids, seeds, timestamps and hashes.

From the repository root:

```bash
uv run scripts/run_arrow_grouped.py --feature-scope complete-pr --output work/arrow-grouped-original-new --workers 8
```

The output path must be new and under ignored `work/`. To verify the published
saved results without any fitting, prepare a new workspace and run
`uv run scripts/verify_results.py work/public` as described in the root README.
