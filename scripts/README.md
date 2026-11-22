# Entry points

Run scripts from the repository root with `uv run scripts/<name>.py`. Each script
declares Python 3.14 and its dependencies in a PEP 723 `# /// script` block, following
the [uv script dependency guide](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies).
uv automatically supplies the interpreter and an environment for that script;
no activated virtual environment, `--project`, or prior `uv sync` is required.

```bash
uv run scripts/prepare_workspace.py
uv run scripts/verify_results.py work/public
uv run scripts/verify_arrow_selection.py
uv run scripts/check_public_tree.py
```

Standard-library-only scripts declare `dependencies = []` explicitly.
`verify_results.py` pins NumPy, SciPy, scikit-learn, XGBoost and joblib to the versions
used by the saved regression analysis. The latter three are needed by the local
`report`/`evaluate` modules it imports; no packages are read from their virtual
environment. Those source modules must remain in the repository.

| Script | Inputs / effect |
|---|---|
| `prepare_workspace.py` | Copy final derived artifacts and optional authorized Verified data into ignored `work/` |
| `verify_results.py` | Recompute saved regression and Judge metrics, including the retained scientific sensitivities |
| `run_arrow_grouped.py` | Refit only Arrow Lasso/XGBoost on the saved 55 tasks using 50 base-commit groups at both nested CV levels; write a new directory under `work/` |
| `fetch_pro_reports.py` | Download and hash-check the nine matched public Pro reports; no model calls |
| `aggregate_pro_targets.py` | Rebuild the 731 Pro targets from 13 reports, or the matched 381-task nine-report sensitivity with `--traceable-only` |
| `verify_arrow_selection.py` | Check the saved 701→100→73→55 Arrow selection and construction categories |
| `fetch_reference_sources.py` | Fetch/hash-check pinned public base files and reference patches into a temporary bundle |
| `build_distribution.py` | Generate the complexity histogram from saved integer outcome counts |
| `check_public_tree.py` | Inspect publication files and exclusions without staging or committing |

The public-data instructions are in `data/README.md`. Preparation and verification
do not fetch research data or call models. uv may download Python or dependencies
on first use; once cached, `uv run --offline scripts/<name>.py` can resolve the
environment offline. The two acquisition scripts access the network when invoked;
uv's `--offline` flag does not disable a script's own HTTP requests. No script
creates a Git commit or uploads the repository.
