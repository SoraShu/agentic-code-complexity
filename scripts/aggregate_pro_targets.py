# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = []
# ///

"""Rebuild Pro targets from 13 reports, or the matched nine-report sensitivity."""
import argparse
from collections import Counter,defaultdict
import csv,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def aggregate(reports, traceable_only=False):
    source=json.loads((ROOT/'data/sources.json').read_text())['pro']
    expected={r['file']:r['sha256'] for r in source['historical_report_hashes']}
    actual={p.name:p for p in reports.glob('report-*.csv')}
    selected=set(expected)-set(source['unmapped_reports']) if traceable_only else set(expected)
    if not selected<=set(actual) or set(actual)-set(expected):
        raise ValueError(f'Report set mismatch; missing={sorted(selected-set(actual))}, extra={sorted(set(actual)-set(expected))}')
    cohort={json.loads(l)['instance_id'] for l in (ROOT/'data/results/judge/pro-zero-shot.jsonl').read_text().splitlines()}
    if traceable_only:
        cohort={r['instance_id'] for r in json.loads((ROOT/'data/targets/pro.json').read_text())
                if r['n_valid']==13}
        assert len(selected)==9 and len(cohort)==381
    counts=defaultdict(Counter)
    for name in sorted(selected):
        path=actual[name]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[name]:raise ValueError(f'Report bytes differ: {name}')
        seen=set()
        for row in csv.DictReader(path.open()):
            task,status=row['InstanceId'],row['Resolved'].strip().lower()
            assert task not in seen and status in {'true','false','error','missing'}
            seen.add(task)
            if task in cohort:counts[task][status]+=1
    rows=[]
    for task in sorted(cohort):
        c=counts[task];assert c['true']+c['false']>0
        if traceable_only:assert c['true']+c['false']==9 and not c['error'] and not c['missing']
        rows.append(dict(instance_id=task,n_resolved=c['true'],n_valid=c['true']+c['false'],
                         n_errors=c['error'],n_missing=c['missing'],n_reports=sum(c.values())))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reports',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--traceable-only',action='store_true',help='Use the nine mapped reports on the same 381 tasks covered by all 13 reports')
    a=p.parse_args();rows=aggregate(a.reports,a.traceable_only)
    target='pro-complete-traceable.json' if a.traceable_only else 'pro.json'
    assert rows==json.loads((ROOT/'data/targets'/target).read_text()),'Derived targets differ'
    if not a.output.resolve().is_relative_to(ROOT/'work'):raise ValueError('Write regenerated outputs under work/')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:f.write(json.dumps(rows,indent=2)+'\n')
    print('Reproduced 381 matched targets from nine traceable reports.' if a.traceable_only
          else 'Reproduced all 731 targets from 13 reports; excluded three unknown outcomes.')
