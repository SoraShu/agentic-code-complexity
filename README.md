# Agentic Code Complexity

Code and derived research results for **Measuring and Predicting Agentic
Code Complexity**.

This repository contains the final feature extraction, regression, shared-rubric
LLM Judge, figure generation and case-analysis code used in the thesis. Internal
Benchpress code, unused adapters and intermediate experiments are excluded. Arrow
solver runs used Harbor's native GitHub Copilot integration.

## Recompute saved results

```bash
uv run scripts/prepare_workspace.py
uv run scripts/verify_results.py work/public
```

Each entry-point script declares its Python version and dependencies inline; uv
creates its environment automatically, without a separate project environment.

This checks Arrow's 55-task oracle-pass main regression, the two 59-task
sensitivity analyses, and the Pro Judge on 731 tasks and the 381-task subset
covered by all 13 historical reports. It uses saved outputs and derived targets; no model is
trained or called. Public original datasets are obtained separately as described in
[data/README.md](data/README.md).

Verified's historical Microsoft outcome reports and task-level results are private.
With an authorized local copy, additionally run:

```bash
uv run scripts/prepare_workspace.py --destination work/full --private-root /path/to/private-archive
uv run scripts/verify_results.py work/full
```

The private extension verifies Verified's adopted regression, outcome-policy
sensitivity, zero-shot Judge and the paired 479-task 20-shot comparison. Only the
required final data are imported into the ignored working directory.

## Contents

| Directory | Purpose |
|---|---|
| [packages/](packages/README.md) | Feature extraction, regression, Judge and PR selection code |
| [analysis/](analysis/README.md) | Results figures and Appendix B evidence extraction |
| [data/](data/README.md) | Derived features, targets, final predictions/folds and public-source instructions |
| [scripts/](scripts/README.md) | Data acquisition, workspace preparation and saved-result verification |

Each component README explains its inputs and entry points. Downloaded raw data and
all generated outputs go under Git-ignored `work/`. Final TeX/PDF figures remain in
the [thesis repository](https://github.com/SoraShu/master-thesis); no copy is kept here.
The private companion archive remains outside this repository.
