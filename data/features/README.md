# Final reference-code features

`verified/` contains the 500 adopted Verified feature records;
`source_only/arrow/` contains the 73 adopted Arrow candidate records.
All 19 features exclude repository-root `tests/` paths. Structural metrics use
the benchmark base plus reference patch. These are numerical results, not
copies of repository source code; source identities and hashes are in
`../reference-inputs.json`. No internal success labels are included here.

The original `arrow/` records remain unchanged and include original PR tests.
Their per-file measurements support the source-only aggregation; change-size
and directory features are also recomputed after excluding tests. All 500
Verified reference patches already exclude test files, and their 19-feature
vectors are unchanged. See [the source-only guide](source_only/README.md).
