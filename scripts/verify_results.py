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

"""Recompute thesis statistics from saved predictions. Never fits or calls a model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'packages/regression'))
import report


def outcomes(folder):
    counts = defaultdict(Counter)
    files = sorted(folder.glob('report-*.csv'))
    if not files:
        raise FileNotFoundError(f'No outcome reports at {folder}')
    for path in files:
        seen = set()
        for row in csv.DictReader(path.open()):
            task, status = row['InstanceId'], row['Resolved'].strip().lower()
            assert task not in seen and status in {'true', 'false', 'error', 'missing'}
            seen.add(task)
            counts[task][status] += 1
    assert all(c['true'] + c['false'] > 0 for c in counts.values())
    return counts


def predictions(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == len({r['instance_id'] for r in rows}), 'Duplicate predictions'
    assert all(0 <= r['p_success'] <= 1 for r in rows)
    return {r['instance_id']: r for r in rows}


def judge_metrics(pred, truth, include_unknown=False):
    ids = sorted(pred)
    assert set(ids) <= set(truth), 'Predictions missing an outcome target'
    p = np.array([pred[t]['p_success'] for t in ids])
    y = np.array([truth[t]['true'] / (sum(truth[t].values()) if include_unknown
                 else truth[t]['true'] + truth[t]['false']) for t in ids])
    constant = bool(np.ptp(p) == 0)
    return dict(n=len(ids), spearman=None if constant else float(spearmanr(p, y).statistic),
                pearson=None if constant else float(pearsonr(p, y).statistic),
                mae=float(np.mean(abs(p-y))), mse=float(np.mean((p-y)**2)),
                mean_prediction=float(p.mean()), mean_observed=float(y.mean()),
                signed_mean_error=float(np.mean(p-y)))


def main(workspace):
    root = workspace.resolve()
    result = {'regression': {}, 'judge': {}, 'outcomes': {},
              'primary_arrow_analysis': 'arrow_oracle_pass', 'feature_scope': 'source-only (exclude tests/)'}
    reg = root / 'experiments/regression/results'
    scenarios = [(name, reg / 'source-only' / name)
                 for name in ('arrow', 'arrow_oracle_pass', 'arrow_errors_as_failures',
                              'arrow_oracle_pass_grouped')]
    verified = reg / 'verified-boolean-outcomes-2026-11-09/verified'
    has_private = verified.is_dir()
    if has_private:
        scenarios += [('verified', verified),
                      ('verified_all_reports', reg / 'reference-features-2026-11-09/verified')]
    for name, folder in scenarios:
        saved = json.loads((folder / 'results.json').read_text())
        dataset_name = saved['dataset']
        metric_name = 'pr_metrics.jsonl' if dataset_name == 'arrow' else 'task_metrics.jsonl'
        feature_records = [json.loads(line) for line in (root/'experiments/regression/inputs-reference'/dataset_name/metric_name).read_text().splitlines()]
        features_by_id = {(f"pr-{r['pr_number']}" if dataset_name == 'arrow' else r['task_name']): report.ev.feature_vector(r)
                          for r in feature_records}
        for row in json.loads((folder/'dataset.json').read_text())['rows']:
            assert row['features'] == features_by_id[row['task_id']], (name, row['task_id'])
        _, y, p, folds = report.audit(folder, saved)
        _, metrics = report.with_median(y, p, folds)
        result['regression'][name] = {'n': len(y), 'metrics': metrics}
        if name == 'arrow_oracle_pass_grouped':
            data = json.loads((folder/'dataset.json').read_text())
            original = json.loads((folder.with_name('arrow_oracle_pass')/'dataset.json').read_text())
            assert data == original, 'Grouped sensitivity must preserve all main-analysis inputs'
            groups = json.loads((folder/'groups.json').read_text())['task_groups']
            ids = [r['task_id'] for r in data['rows']]
            refs = json.loads((REPO/'data/reference-inputs.json').read_text())
            expected = {r['task_id']: r['repo']+'@'+r['base_commit'] for r in refs
                        if r['dataset'] == 'arrow' and r['task_id'] in ids}
            assert groups == expected and len(set(groups.values())) == 50
            report.ev.validate_splits(ids, json.loads((folder/'splits.json').read_text()), groups)
            recorded = json.loads((folder/'metrics_with_median.json').read_text())
            for model, values in metrics.items():
                for metric in ['mae_mean', 'mae_std', 'r2_mean', 'r2_std']:
                    assert abs(values[metric]-recorded[model][metric]) < 1e-12
            result['regression'][name]['n_groups'] = 50
    assert result['regression']['arrow']['n'] == 59
    assert abs(result['regression']['arrow']['metrics']['median']['mae_mean'] - .235) < .0005
    primary = result['regression']['arrow_oracle_pass']
    assert primary['n'] == 55
    for name in ('arrow', 'arrow_oracle_pass', 'arrow_errors_as_failures', 'arrow_oracle_pass_grouped'):
        current = reg/'source-only'/name
        previous = reg/'reference-features-2026-11-09'/name
        assert json.loads((current/'splits.json').read_text()) == json.loads((previous/'splits.json').read_text())
        new_rows = json.loads((current/'dataset.json').read_text())['rows']
        old_rows = json.loads((previous/'dataset.json').read_text())['rows']
        assert [{k:v for k,v in r.items() if k != 'features'} for r in new_rows] == [
                {k:v for k,v in r.items() if k != 'features'} for r in old_rows]
        recorded = json.loads((current/'metrics_with_median.json').read_text())
        for model, values in result['regression'][name]['metrics'].items():
            for metric in ['mae_mean','mae_std','r2_mean','r2_std']:
                assert abs(values[metric]-recorded[model][metric]) < 1e-12
    result['correlations'] = {}
    expected_corr = json.loads((REPO/'data/results/regression/source_only/correlations.json').read_text())
    for dataset, folder in [('arrow', reg/'source-only/arrow_oracle_pass')] + ([('verified', verified)] if has_private else []):
        data = json.loads((folder/'dataset.json').read_text())
        X = np.asarray([r['features'] for r in data['rows']], dtype=float)
        y = np.asarray([r['complexity'] for r in data['rows']])
        rows = []
        for i, feature in enumerate(data['feature_names']):
            keep = np.isfinite(X[:, i])
            rows.append(dict(feature=feature, n=int(keep.sum()),
                             pearson=float(pearsonr(X[keep, i], y[keep]).statistic),
                             spearman=float(spearmanr(X[keep, i], y[keep]).statistic)))
        for actual, expected in zip(rows, expected_corr[dataset], strict=True):
            assert actual['feature'] == expected['feature'] and actual['n'] == expected['n']
            assert abs(actual['pearson'] - expected['pearson']) < 1e-12
            assert abs(actual['spearman'] - expected['spearman']) < 1e-12
        result['correlations'][dataset] = rows
    if has_private:
        v = result['regression']['verified']
        assert v['n'] == 500
        assert abs(v['metrics']['xgboost']['r2_mean'] - .122112) < 1e-6
        assert abs(v['metrics']['xgboost']['mae_mean'] - .270499) < 1e-6
    for dataset in (['pro', 'verified'] if has_private else ['pro']):
        name = 'pro_opus47_showtests.jsonl' if dataset == 'pro' else 'opus47_showtests_run.jsonl'
        pred = predictions(root / 'localdataset/results/judge' / name)
        if dataset == 'pro':
            target_rows = json.loads((root/'localdataset/data/pro-targets.json').read_text())
            assert len(target_rows) == len({r['instance_id'] for r in target_rows}) == 731
            truth = {r['instance_id']: Counter(true=r['n_resolved'], false=r['n_valid']-r['n_resolved'],
                       error=r['n_errors'], missing=r['n_missing']) for r in target_rows}
            assert all(sum(truth[r['instance_id']].values()) == r['n_reports'] for r in target_rows)
        else:
            truth = outcomes(root / f'localdataset/results/swe-bench-{dataset}')
        assert set(pred) == set(truth)
        totals = sum(truth.values(), Counter())
        result['outcomes'][dataset] = dict(tasks=len(truth), statuses=dict(totals),
            valid_counts=dict(sorted(Counter(c['true']+c['false'] for c in truth.values()).items())))
        metrics = judge_metrics(pred, truth)
        result['judge'][dataset] = metrics
        if dataset == 'pro':
            assert len(pred) == 731 and totals['missing'] == 3
            assert totals['true'] + totals['false'] == 8887
            assert abs(metrics['spearman'] - .568350) < 1e-6
            result['judge']['pro_missing_as_failure'] = judge_metrics(pred, truth, True)
            complete = {t: r for t, r in pred.items() if truth[t]['true'] + truth[t]['false'] == 13}
            result['judge']['pro_complete_coverage'] = judge_metrics(complete, truth)
            assert len(complete) == 381
            assert abs(result['judge']['pro_complete_coverage']['spearman'] - .6063605635712134) < 1e-12
            reference = {t: {'p_success': .5} for t in pred}
            result['judge']['pro_constant_half'] = judge_metrics(reference, truth)
            assert abs(result['judge']['pro_constant_half']['mae'] - .28412383330303986) < 1e-12
            matched_rows = json.loads((root/'localdataset/data/pro-complete-traceable.json').read_text())
            assert len(matched_rows) == len({r['instance_id'] for r in matched_rows}) == 381
            assert {r['instance_id'] for r in matched_rows} == set(complete)
            assert all(r['n_valid'] == r['n_reports'] == 9 and r['n_errors'] == r['n_missing'] == 0
                       and 0 <= r['n_resolved'] <= 9 for r in matched_rows)
            matched_truth = {r['instance_id']: Counter(true=r['n_resolved'], false=9-r['n_resolved'])
                             for r in matched_rows}
            matched = judge_metrics(complete, matched_truth)
            result['judge']['pro_complete_traceable'] = matched
            assert abs(matched['spearman'] - .5625997661674826) < 1e-12
            assert abs(matched['mae'] - .3352085156022164) < 1e-12
        else:
            assert totals == Counter(true=9005, false=3936, error=50, missing=9)
            assert abs(metrics['spearman'] - .627477) < 1e-6
            few = predictions(root / 'localdataset/results/judge/opus47_showtests_20shot.jsonl')
            assert len(few) == 479 and set(few) <= set(pred)
            result['judge']['verified_paired_zero_shot'] = judge_metrics({t: pred[t] for t in few}, truth)
            result['judge']['verified_20_shot'] = judge_metrics(few, truth)
            assert abs(result['judge']['verified_20_shot']['spearman'] - .639630) < 1e-6
    result['verification_scope'] = 'public plus private' if has_private else 'public Arrow and Pro; Verified outcomes not distributed'
    output = root / 'verification.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: {n: (v['n'] if 'n' in v else v) for n,v in result[k].items()}
                      for k in ('regression','judge')}, indent=2))
    print(f'Verified saved results: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    main(parser.parse_args().workspace)
