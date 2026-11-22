# PR selection

`process_prs.py` extracts PR metadata from a public GitHub PR JSONL snapshot;
`label_prs.py` provides manual labels; `llm_label_prs.py` provides optional model
labels according to `label.md`. Input and output paths are explicit CLI arguments.

Public PR records can be obtained through the [GitHub pull-request API](https://docs.github.com/en/rest/pulls/pulls#list-pull-requests)
for `arrow-py/arrow`; download them into ignored `work/`. Candidate IDs and saved
labels produced for the study are in `data/selection/`. Full PR metadata snapshots,
exploratory plots and placeholder programs are not kept here.

The thesis construction inputs are reproduced by
`uv run scripts/verify_arrow_selection.py` from the saved 701-PR derived screening
counts. Ranking by complete-diff added/deleted lines gives all 100 input IDs in
their recorded order. The collector `process_prs.py` reads GitHub's `.patch`
payload, which can contain multiple commit patches; its counts must not be
substituted for that historical complete-diff screening snapshot. Neither the
manual labels nor the optional model labels define the final top-100 selection.

These tools can access GitHub or model services when explicitly run. They are not
invoked by the offline verification commands. Supply credentials through environment
variables; no credentials are part of this repository.
