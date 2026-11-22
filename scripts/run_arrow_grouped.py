# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = [
#     "numpy==2.5.0",
#     "scipy==1.18.0",
#     "scikit-learn==1.9.0",
#     "xgboost==3.3.0",
#     "joblib==1.6.0",
# ]
# ///
"""Fit Arrow's base-commit-grouped sensitivity from the saved 55-task dataset."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'packages/regression'))
import evaluate as ev
import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory under work/; existing results are never overwritten')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(REPO / 'work') or args.workers < 1:
        parser.error('Use a new output directory under work/ and a positive worker count')
    source = REPO / 'data/results/regression/arrow_oracle_pass/dataset.json'
    reference = REPO / 'data/reference-inputs.json'
    dataset = json.loads(source.read_text(encoding='utf-8'))
    ids = {r['task_id'] for r in dataset['rows']}
    assert len(ids) == len(dataset['rows']) == 55
    assert all(r['oracle_pass'] for r in dataset['rows'])
    records = [r for r in json.loads(reference.read_text(encoding='utf-8'))
               if r['dataset'] == 'arrow' and r['task_id'] in ids]
    groups = {r['task_id']: r['repo'] + '@' + r['base_commit'] for r in records}
    assert set(groups) == ids and len(set(groups.values())) == 50
    members = defaultdict(list)
    for task, group in groups.items():
        members[group].append(task)
    metadata = {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'platform': platform.platform(),
        'entry_point': str(Path(__file__).relative_to(REPO)),
        'workers': args.workers, 'seed': args.seed,
        'versions': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'scipy', 'scikit-learn', 'xgboost', 'joblib', 'threadpoolctl']},
        'input_sha256': {str(p.relative_to(REPO)): ev.digest(p) for p in [source, reference]},
        'code_sha256': {str(p.relative_to(REPO)): ev.digest(p) for p in
                        [Path(__file__), Path(ev.__file__), Path(report.__file__)]},
        'splitter': 'GroupKFold(n_splits=5, shuffle=True, random_state=seed) at both levels',
        'group_definition': 'Repository and benchmark base commit; no outcomes or features used for grouping',
        'n_groups': len(members), 'outer_folds': 5, 'inner_folds': 5,
        'lasso_alphas': ev.ALPHAS, 'lasso_selection_loss': 'MSE',
        'xgb_grid': ev.XGB_GRID, 'xgb_candidates': len(ev.ParameterGrid(ev.XGB_GRID)),
        'xgb_selection_loss': 'MAE',
        'preprocessing': 'Training-fold median imputation; Lasso standardization within each fit',
        'summary': 'Task-weighted metrics within each fold; unweighted mean and population SD across outer folds; pooled OOF retained; no clipping',
    }
    output.mkdir(parents=True, exist_ok=False)
    ev.dump(output / 'metadata.json', metadata)
    result = ev.run('arrow', 'oracle_pass', output, args.workers, args.seed,
                    dataset_snapshot=dataset, groups=groups)
    folder = output / 'arrow_oracle_pass_grouped'
    ev.dump(folder / 'groups.json', {
        'definition': metadata['group_definition'], 'task_groups': groups,
        'multi_task_groups': {g: sorted(ts) for g, ts in members.items() if len(ts) > 1},
    })
    ev.validate_splits([r['task_id'] for r in dataset['rows']],
                       json.loads((folder / 'splits.json').read_text()), groups)
    _, y, predictions, folds = report.audit(folder, result)
    _, metrics = report.with_median(y, predictions, folds)
    ev.dump(folder / 'metrics_with_median.json', metrics)
    metadata['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    ev.dump(output / 'metadata.json', metadata)
    ev.dump(folder / 'metadata.json', metadata)
    print(json.dumps({m: {k: v for k, v in row.items() if k.endswith(('_mean', '_std'))}
                      for m, row in metrics.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
