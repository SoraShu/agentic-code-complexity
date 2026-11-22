# Arrow run summaries

`run_1.json` through `run_10.json` summarize the ten solver attempts; `oracle.json` records reference-solution validation. These summaries were produced for this thesis. The solver used Harbor's native `copilot-cli` integration with GPT-5.4.

The main analysis excludes oracle-failing tasks with zero observed successes, retains 59 tasks and scores 588 valid outcomes. Oracle-only and error-as-failure policies are retained as scientific sensitivity analyses. Full execution logs and internal construction code are not included.
