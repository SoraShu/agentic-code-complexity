# Implementations

- `regression/`: final reference-code features, nested Lasso/XGBoost evaluation and training-fold mean/median baselines.
- `judge/`: the shared three-step rubric, task/context loading, prediction runner and Boolean-only evaluation.
- `pr-selection/`: PR metadata extraction and the labelling tools used during Arrow selection.

Each component keeps its own dependency file/lock.

