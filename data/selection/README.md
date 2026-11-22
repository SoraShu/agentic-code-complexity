# Arrow PR selection outputs

`arrow-candidates.txt` contains the 100 construction inputs, in their original
order. They are the first 100 records after sorting the saved 701-PR screening
snapshot by `lines_changed` descending, retaining snapshot order for ties.
The historical parser counts lines in the complete PR diff beginning with `+`
or `-`, except any line beginning with `+++` or `---`. This prefix rule omits
file headers but also some changed content, such as documentation separators.
The selected boundary is 135 by this historical count; the next PR has 133.
It is not patch byte size or a sum over individual commit patches.

The screening measure is distinct from the regression feature `lines_changed`,
which includes these content lines. Hash-matched reference diffs give the
following comparisons (independently checked with `git apply --numstat`):

| PR | Historical screening count | Changed lines / regression feature |
|---|---:|---:|
| 608 | 433 | 444 |
| 647 | 159 | 161 |
| 791 | 140 | 141 |
| 1135 | 1208 | 1213 |

`arrow-screening.json` contains the 701 derived line counts and the original
metric-snapshot SHA-256. Applying the historical prefix rule to the retained
construction diffs reproduces all 100 selected counts. The selection does not impose a merged-only filter.
The separately retained `arrow_labels*.jsonl` are exploratory labelling outputs;
the construction list is reproduced without using those labels.

`arrow-construction.json` records the 100 terminal construction outcomes: 73
constructed tasks, two issue-stage exits, two scaffold-stage exits, sixteen
test-implementation-stage exits and seven reported test-implementation failures.
These are recorded terminal categories, not inferred root causes. The original
internal construction logs remain outside this repository. Their successful IDs
match the 73 oracle inputs exactly. Of those tasks, 55 pass the oracle and form
the main analysis; 18 fail. The extended 59-task sensitivity includes four of
the oracle-fail tasks with solving-agent success.

```bash
uv run scripts/verify_arrow_selection.py
```

The check reproduces the selection and joins construction, oracle and analysis
IDs. Public PR text and metadata can be obtained from
`https://github.com/arrow-py/arrow/pull/<number>`; no raw PR snapshot or diff text
is bundled. The preserved counts, rather than live GitHub state, define this
historical screening snapshot.
