# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = []
# ///

"""Rebuild the published Pro targets from all 13 authorized source reports."""
import argparse
from collections import Counter,defaultdict
import csv,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def aggregate(reports):
    source=json.loads((ROOT/'data/sources.json').read_text())['pro']
    expected={r['file']:r['sha256'] for r in source['historical_report_hashes']}
    actual={p.name:p for p in reports.glob('report-*.csv')}
    if set(expected)!=set(actual):
        raise ValueError(f'Exact 13-report set required; missing={sorted(set(expected)-set(actual))}, extra={sorted(set(actual)-set(expected))}')
    cohort={json.loads(l)['instance_id'] for l in (ROOT/'data/results/judge/pro-zero-shot.jsonl').read_text().splitlines()}
    counts=defaultdict(Counter)
    for name,path in actual.items():
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
        rows.append(dict(instance_id=task,n_resolved=c['true'],n_valid=c['true']+c['false'],
                         n_errors=c['error'],n_missing=c['missing'],n_reports=sum(c.values())))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reports',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=aggregate(a.reports)
    assert rows==json.loads((ROOT/'data/targets/pro.json').read_text()),'Derived targets differ'
    if not a.output.resolve().is_relative_to(ROOT/'work'):raise ValueError('Write regenerated outputs under work/')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:f.write(json.dumps(rows,indent=2)+'\n')
    print('Reproduced all 731 targets from 13 reports; excluded three unknown outcomes.')
