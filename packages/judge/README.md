# Judge

The active package uses the **shared three-step reasoning rubric** adopted in the
thesis. The recovered rubric is in `llm_judge/three_step_rubric.txt`; the default
model is `github_copilot/claude-opus-4.7`, with reference patch and evaluation test
context enabled. The package retains the saved context budgets and cache/retry
mechanism. It does not imply repeated independent scores for each task.

```bash
uv sync --locked --python 3.14
uv run --locked pytest tests -q
uv run --locked judge --help
```

For thesis numbers, use the repository's `scripts/verify_results.py` on saved
predictions. `judge calibrate` now excludes `error` and `missing`, reports task-level
MSE and MAE, and invalidates truth caches when source bytes or the outcome policy
change. Historical 20-shot labels keep the all-report denominator used during
prompt development; this does not change the final evaluation denominator.
The precise historical early 20-shot exemplar membership was not recovered, so
selecting examples again is a new run, not exact replay of the saved comparison.

To make **new** model calls, explicitly run `judge run` with a dataset and a new
output path. The root README documents data preparation. Authentication is supplied
through the provider's normal environment/device login; no credentials are bundled.
New calls are optional and were not made while assembling this repository.
