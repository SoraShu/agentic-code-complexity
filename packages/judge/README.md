# Judge

The active package uses the **shared three-step reasoning rubric** adopted in the
thesis. The recovered rubric is in `llm_judge/three_step_rubric.txt`; the default
model is `github_copilot/claude-opus-4.7`, with reference patch and evaluation test
context enabled. The package retains the saved context budgets and cache/retry
mechanism. It does not imply repeated independent scores for each task.

## Historical messages and the released wrapper

The early source revisions `1d888ed` and `9f60361` supplied the rubric as the
system message and task materials as the user message. Their rubric texts are
identical to `three_step_rubric.txt` after stripping surrounding whitespace
(SHA-256 `a26d27843f17db962a15a1a9abb30cb18e52212f55d36b213b76a58b4653236a`).
The later wrapper retained here embeds those instructions in the user message,
before the task materials, and sends the system message
`You output ONLY the single JSON object requested below.`
Thus the released wrapper is not an exact replay of the early message layout.

Historical source and run records support the reconstructed shared three-step
rubric. The retained Verified launch record precedes the later wrapper change;
complete invocation records for the early Pro and twenty-shot runs were not
recovered. Per-call prompt snapshots and message-role fingerprints were not
saved, so a particular message layout cannot be assigned to every score.
The `cached` flag records reuse, not rubric or message-format identity: 499 of
500 Verified zero-shot scores, four of 731 Pro scores and none of 479 Verified
twenty-shot scores are marked cached. Each task contributes one retained score.

The response-cache key hashes the model/task identity and rubric plus task text.
It does not fingerprint the message-role wrapper or all API settings. Changing
those requires a fresh cache (or `--no-cache`) and a new output file; output
resume skips existing task IDs independently of the response cache.

## Use

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
