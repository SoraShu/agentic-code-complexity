# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = [
#     "numpy==2.5.0", "scipy==1.18.0", "scikit-learn==1.9.0",
#     "xgboost==3.3.0", "joblib==1.6.0",
# ]
# ///
"""Refit source-only Arrow Lasso/XGBoost with the original nested splits and grids."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'packages/regression'))
import evaluate as ev
import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(REPO / 'work') or output.exists() or args.workers < 1:
        parser.error('Use a new output directory under work/ and a positive worker count')
    old_root = REPO / 'data/results/regression'
    feature_root = REPO / 'data/features/source_only'
    manifest = json.loads((feature_root / 'manifest.json').read_text())
    for path, expected in manifest['files'].items():
        assert ev.digest(feature_root / path) == expected, path
    records = {f"pr-{r['pr_number']}": r for r in map(json.loads, (feature_root / 'arrow/pr_metrics.jsonl').read_text().splitlines())}
    scenarios = ['arrow_oracle_pass', 'arrow', 'arrow_errors_as_failures', 'arrow_oracle_pass_grouped']
    inputs = [feature_root / 'manifest.json', feature_root / 'arrow/pr_metrics.jsonl']
    inputs += [old_root / name / fn for name in scenarios for fn in ['dataset.json', 'splits.json']]
    inputs.append(old_root / 'arrow_oracle_pass_grouped/groups.json')
    metadata = {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'entry_point': str(Path(__file__).relative_to(REPO)), 'python': sys.version,
        'workers': args.workers, 'seed': 42, 'outer_folds': 5, 'inner_folds': 5,
        'scope': manifest['definition'], 'scenarios': scenarios,
        'input_sha256': {str(p.relative_to(REPO)): ev.digest(p) for p in inputs},
        'code_sha256': {str(p.relative_to(REPO)): ev.digest(p) for p in [Path(__file__), Path(ev.__file__), Path(report.__file__)]},
        'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'scikit-learn', 'xgboost', 'joblib', 'threadpoolctl']},
        'lasso_alphas': ev.ALPHAS, 'lasso_selection_loss': 'MSE',
        'xgb_grid': ev.XGB_GRID, 'xgb_candidates': len(ev.ParameterGrid(ev.XGB_GRID)), 'xgb_selection_loss': 'MAE',
        'preprocessing': 'Training-fold median imputation; Lasso standardization within each fit',
        'splits': 'Identical saved outer and inner task assignments in each original scenario; grouped scenario retains all 50 base-commit groups.',
    }
    output.mkdir(parents=True)
    ev.dump(output / 'metadata.json', metadata)
    for name in scenarios:
        old = json.loads((old_root / name / 'dataset.json').read_text())
        dataset = {**old, 'rows': [{**r, 'features': ev.feature_vector(records[r['task_id']])} for r in old['rows']]}
        groups = None
        if name.endswith('_grouped'):
            group_record = json.loads((old_root / name / 'groups.json').read_text())
            groups = group_record['task_groups']
        policy = 'oracle_pass' if groups else name.removeprefix('arrow_') if name != 'arrow' else 'historical'
        result = ev.run('arrow', policy, output, args.workers, 42, splits_from=old_root,
                        dataset_snapshot=dataset, groups=groups)
        folder = output / name
        assert json.loads((folder / 'splits.json').read_text()) == json.loads((old_root / name / 'splits.json').read_text())
        if groups:
            ev.dump(folder / 'groups.json', group_record)
        _, y, predictions, folds = report.audit(folder, result)
        _, metrics = report.with_median(y, predictions, folds)
        ev.dump(folder / 'metrics_with_median.json', metrics)
    metadata['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    ev.dump(output / 'metadata.json', metadata)
    print('Finished all four source-only scenarios; original targets and nested splits preserved.', flush=True)


if __name__ == '__main__':
    main()
