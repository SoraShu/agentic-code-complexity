# Source-only feature inputs

The adopted 19 features exclude repository-relative paths beginning with
`tests/`. This is a directory rule, not a test-filename classifier. Other file
types, including documentation and configuration, remain in the six change-size
and directory-spread statistics. Structural metrics use complete affected
Python files reconstructed at the benchmark base plus reference patch.

`arrow/` contains the 73 candidate feature records and derived per-file change
counts. `manifest.json` records input/code hashes and checks on all 573 reference
patches. Verified uses `../verified/task_metrics.jsonl` directly: all 500 feature
vectors are unchanged because its reference patches contain no test paths.

The same aggregation conventions apply after filtering: empty control-flow
summaries and HV/LLOC totals are zero, empty HV/LLOC means are missing, and MI
uses only defined file values. An unavailable retained Python file makes all
13 structural features missing; excluded test files do not affect availability.

To rebuild from a hash-matched source bundle acquired as described in
[`../../README.md`](../../README.md):

```bash
uv run scripts/build_source_only_features.py \
  --source-bundle work/reference-sources.jsonl.gz --output work/source-only-features-new
```

The script verifies the original full-diff counts and compares them with Git's
independent `--numstat` parser before filtering. Raw patches are read locally,
never copied into the release. Original complete-PR feature records remain in
`../arrow/`; original outcomes and analysis cohorts are unchanged.
