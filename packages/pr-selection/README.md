# PR selection

`process_prs.py` extracts PR metadata from a public GitHub PR JSONL snapshot;
`label_prs.py` provides manual labels; `llm_label_prs.py` provides optional model
labels according to `label.md`. Input and output paths are explicit CLI arguments.

Public PR records can be obtained through the [GitHub pull-request API](https://docs.github.com/en/rest/pulls/pulls#list-pull-requests)
for `arrow-py/arrow`; download them into ignored `work/`. Candidate IDs and saved
labels produced for the study are in `data/selection/`. Full PR metadata snapshots,
exploratory plots and placeholder programs are not kept here.

These tools can access GitHub or model services when explicitly run. They are not
invoked by the offline verification commands. Supply credentials through environment
variables; no credentials are part of this repository.
