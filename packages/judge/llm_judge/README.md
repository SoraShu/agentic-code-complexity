# Judge modules

`prompt.py` and `three_step_rubric.txt` define the adopted shared prompt. `loader.py` parses tasks, `repo_context.py` retrieves bounded source windows, `schema.py` validates outputs, `judge.py` performs cached/retried calls, and `runner.py` handles batches. `fewshot.py` selects/examples labels; exact historical early 20-shot membership cannot be reconstructed. `ground_truth.py` and `calibration.py` implement final outcome aggregation and saved-prediction metrics. `cli.py` exposes the commands.
