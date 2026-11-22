"""Verified Boolean-outcome targets with the preserved nested-CV protocol.

Only True/False enter the main denominator. The existing fixed-denominator
reference-feature run is the sensitivity comparator; it is never overwritten.
No agent/Judge calls, network access, or feature extraction are performed.
"""
from __future__ import annotations
import argparse
import copy
import csv
import importlib.metadata
import io
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import evaluate as ev  # sets thread limits before importing numerical packages
import numpy as np
import report


def scored_outcomes(values):
    counts = Counter(v.strip().lower() for v in values)
    assert set(counts) <= {'true', 'false', 'error', 'missing'}, counts
    valid = counts['true'] + counts['false']
    assert valid > 0, 'No Boolean outcomes; cannot define a task target'
    return dict(n_resolved=counts['true'], n_valid=valid,
                n_errors=counts['error'], n_missing=counts['missing'],
                n_reports=sum(counts.values()), denominator=valid,
                complexity=1-counts['true']/valid)


def build_dataset(previous):
    source = ev.ROOT/'inputs/verified/results'
    reports = sorted(source.glob('report-*.csv'))
    assert len(reports) == 26
    raw = defaultdict(list)
    for f in reports:
        seen = set()
        for row in csv.DictReader(io.StringIO(f.read_text())):
            task = row['InstanceId']
            assert task not in seen, (f, task)
            seen.add(task)
            raw[task].append(row['Resolved'])
        assert len(seen) == 500
    data = copy.deepcopy(previous)
    assert set(raw) == {r['task_id'] for r in data['rows']}
    for row in data['rows']:
        counts = scored_outcomes(raw[row['task_id']])
        assert counts['n_reports'] == 26 and counts['n_resolved'] == row['n_resolved']
        assert abs(row['complexity']-(1-counts['n_resolved']/26)) < 1e-14
        row.update(counts)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--previous', type=Path, default=ev.ROOT/'results/reference-features-2026-11-09')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    assert args.workers > 0
    old = args.previous.resolve()
    out = args.output.resolve()
    assert not out.exists(), 'Choose a new result directory'
    protected = {}
    for folder in [ev.ROOT/'inputs', ev.ROOT/'inputs-reference', old,
                   ev.ROOT.parent.parent/'localdataset/results/judge']:
        for f in folder.rglob('*'):
            if f.is_file(): protected[str(f)] = ev.digest(f)
    out.mkdir(parents=True)
    dest = out/'verified'; dest.mkdir()
    ev.dump(out/'preserved-sha256.json', protected)
    code = out/'code'; code.mkdir()
    for name in ['evaluate_verified_outcomes.py','evaluate.py','report.py','pyproject.toml','uv.lock']:
        (code/name).write_bytes((ev.ROOT/name).read_bytes())
    metadata = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        python=sys.version,platform=platform.platform(),workers=args.workers,seed=42,
        previous=str(old),policy='Exclude error/missing, retain True/False only',
        sensitivity='Existing all-26-report reference-feature run, with non-Boolean statuses scored zero',
        versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','xgboost','joblib']},
        outer_folds=5,inner_folds=5,lasso_alphas=ev.ALPHAS,lasso_selection_loss='MSE',
        xgb_grid=ev.XGB_GRID,xgb_selection_loss='MAE',
        code_sha256={f.name:ev.digest(f) for f in code.iterdir()})
    ev.dump(out/'metadata.json', metadata)
    previous=json.loads((old/'verified/dataset.json').read_text())
    data=build_dataset(previous); rows=data['rows']; ids=[r['task_id'] for r in rows]
    assert ids==[r['task_id'] for r in previous['rows']] and data['feature_names']==ev.FEATURES
    X=np.asarray([r['features'] for r in rows],float);y=np.array([r['complexity'] for r in rows])
    np.testing.assert_array_equal(X,np.asarray([r['features'] for r in previous['rows']],float))
    ev.dump(dest/'dataset.json',data)
    (dest/'splits.json').write_bytes((old/'verified/splits.json').read_bytes())
    splits=json.loads((dest/'splits.json').read_text());lookup={t:i for i,t in enumerate(ids)}
    folds=[(np.array([lookup[t] for t in s['train_ids']]),np.array([lookup[t] for t in s['test_ids']])) for s in splits]
    predictions={m:np.full(len(y),np.nan) for m in ['baseline','lasso','xgboost']};details=defaultdict(list)
    for i,(train,test) in enumerate(folds,1):
        inner_lookup={ids[j]:k for k,j in enumerate(train)}
        inner=[(np.array([inner_lookup[t] for t in s['train_ids']]),np.array([inner_lookup[t] for t in s['validation_ids']])) for s in splits[i-1]['inner_splits']]
        assert not set(train)&set(test) and set(train)|set(test)==set(range(500))
        for a,b in inner:
            assert not set(a)&set(b) and set(a)|set(b)==set(range(len(train)))
            assert not np.isnan(X[train][a]).all(axis=0).any()
        predictions['baseline'][test]=y[train].mean()
        for model in ['lasso','xgboost']:
            start=time.monotonic();print(f'verified Boolean-only fold {i}/5 {model}: start',flush=True)
            pred,detail=ev.fit_outer_fold(model,X,y,train,test,args.workers,42,inner_splits=inner)
            predictions[model][test]=pred
            detail.update(fold=i,elapsed_seconds=time.monotonic()-start,**ev.scores(y[test],pred))
            details[model].append(detail);ev.dump(dest/f'{model}_fold_{i}.json',detail)
            print(f'{model}: R2={detail["r2"]:.6f} MAE={detail["mae"]:.6f}',flush=True)
    assert all(np.isfinite(v).all() for v in predictions.values())
    assert sorted(j for _,test in folds for j in test)==list(range(500))
    median,metrics=report.with_median(y,predictions,folds)
    result=dict(dataset='verified',verified_policy='boolean_only',n=len(y),target_mean=float(y.mean()),
                target_std=float(y.std()),metrics=metrics,model_folds=details)
    coefficients=np.asarray([d['coefficients'] for d in details['lasso']]);gains=np.asarray([d['gain_importance'] for d in details['xgboost']])
    result['features']=[dict(name=f,lasso_mean=float(coefficients[:,j].mean()),lasso_std=float(coefficients[:,j].std()),
        lasso_selected_folds=int((abs(coefficients[:,j])>1e-8).sum()),gain_mean=float(gains[:,j].mean()),gain_std=float(gains[:,j].std())) for j,f in enumerate(ev.FEATURES)]
    ev.dump(dest/'results.json',result)
    foldnums={j:i for i,(_,test) in enumerate(folds,1) for j in test}
    with (dest/'oof_predictions.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['task_id','outer_fold','complexity',*predictions,'median'])
        for j,t in enumerate(ids):w.writerow([t,foldnums[j],y[j],*[v[j] for v in predictions.values()],median[j]])
    report.audit(dest,result)
    oldresult=json.loads((old/'verified/results.json').read_text());_,oy,op,of=report.audit(old/'verified',oldresult)
    _,om=report.with_median(oy,op,of)
    ev.dump(out/'comparison.json',dict(boolean_only=metrics,all_reports=om,
        target_changed=sum(a['complexity']!=b['complexity'] for a,b in zip(previous['rows'],rows)),
        valid_counts=dict(sorted(Counter(r['n_valid'] for r in rows).items()))))
    lines=['# Verified outcome-policy sensitivity','',
      'Same 500 tasks, features, outer/inner folds, search grids and preprocessing; each policy has its own fitted/tuned models and training-target baselines. The all-report fit is reused from the preserved reference-feature run.','',
      '| Policy | Model | R2 | MAE | MAE reduction vs mean | vs median |','|---|---|---:|---:|---:|---:|']
    for name,m in [('Boolean only (adopted)',metrics),('All reports (sensitivity)',om)]:
        for model,v in m.items():lines.append(f'| {name} | {model} | {v["r2_mean"]:.6f} ± {v["r2_std"]:.6f} | {v["mae_mean"]:.6f} ± {v["mae_std"]:.6f} | {v["mae_improvement_pct"]:.3f}% | {v["mae_improvement_vs_median_pct"]:.3f}% |')
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    for f,h in protected.items():assert ev.digest(f)==h,f
    ev.dump(out/'verification.json',dict(protected_files_unchanged=len(protected),same_features=True,
        same_task_order=True,same_split_bytes=True,training_only_preprocessing_and_baselines='passed',full_grid=True))
    metadata['finished_at_utc']=datetime.now(timezone.utc).isoformat();ev.dump(out/'metadata.json',metadata)
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
