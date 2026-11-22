# Arrow run summaries

`run_1.json` through `run_10.json` summarize the ten solver attempts; `oracle.json` records reference-solution validation. These summaries were produced for this thesis. The solver used Harbor's native `copilot-cli` integration with GPT-5.4.

The main analysis retains the 55 oracle-pass tasks and scores 548 valid outcomes. The extended sensitivity set adds four oracle-failing tasks with observed agent success, for 59 tasks and 588 valid outcomes; another sensitivity counts its two execution errors as failures. The grouped-split sensitivity uses the same 55 main-analysis tasks and outcomes. Full execution logs and internal construction code are not included.
