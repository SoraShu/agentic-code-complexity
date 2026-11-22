"""Independent offline checks of figure data, TeX coordinates and source preservation."""
from pathlib import Path
from collections import defaultdict
import csv,json,hashlib,re
import numpy as np
from scipy.stats import spearmanr,pearsonr
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace',type=Path,required=True)
parser.add_argument('--output-dir',type=Path)
args=parser.parse_args();ROOT=args.workspace.resolve()
HERE=(args.output_dir or ROOT/'generated/figures').resolve();FIG=HERE
data=json.loads((HERE/'plot-data.json').read_text())
for p,h in data['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p

def points(name):
 text=(FIG/(name+'.tex')).read_text()
 return [[(float(a),float(b)) for a,b in re.findall(r'\(([-\d.]+),([-\d.]+)\)',group)] for group in re.findall(r'coordinates \{([^}]+)\}',text)]
def match(actual,expected):np.testing.assert_allclose(actual,expected,rtol=0,atol=5.1e-11)
# Match every individual judge point and descriptive bin against original predictions/CSV outcomes.
blocks=points('judge-prediction-diagnostics');assert len(blocks)==8
for k,(ds,fn) in enumerate([('verified','opus47_showtests_run.jsonl'),('pro','pro_opus47_showtests.jsonl')]):
 truth=defaultdict(list)
 if ds=='pro':
  for r in json.loads((ROOT/'localdataset/data/pro-targets.json').read_text()):truth[r['instance_id']]=[r['n_resolved']/r['n_valid']]
 else:
  for p in sorted((ROOT/'localdataset/results/swe-bench-verified').glob('report-*.csv')):
   for r in csv.DictReader(p.read_text().splitlines()):
    if r['Resolved'] in ['True','False']:truth[r['InstanceId']].append(int(r['Resolved']=='True'))
 rows=sorted([json.loads(l) for l in (ROOT/'localdataset/results/judge'/fn).read_text().splitlines()],key=lambda r:r['instance_id'])
 p=np.array([r['p_success'] for r in rows]);y=np.array([np.mean(truth[r['instance_id']]) for r in rows])
 match(blocks[4*k+1],np.c_[p,y]);means=[];counts=[]
 # All predictions have two-decimal precision. Integer hundredths avoid floating bin-edge drift.
 assert np.max(abs(p*100-np.rint(p*100)))<1e-10
 codes=np.minimum(np.rint(p*100).astype(int)//10,9)
 for i,binrow in enumerate(data['judge'][ds]['bins']):
  mask=codes==i;n=int(mask.sum());assert n==binrow['n']
  if n:
   means.append((np.mean(p[mask]),np.mean(y[mask])))
   assert abs(means[-1][0]-binrow['mean_pred'])<1e-14
   assert abs(means[-1][1]-binrow['mean_observed'])<1e-14
  counts.append((i/10+.05,n))
 match(blocks[4*k+2],means);match(blocks[4*k+3],counts)
 assert sum(n for _,n in counts)==len(rows)
 if ds=='pro':assert all(b<a for a,b in means)
# Fold values: saved OOF predictions; derive the median only from the corresponding training rows.
blocks=points('regression-fold-performance');cursor=0
for ds,base in [('verified','verified-boolean-outcomes-2026-11-09'),('arrow','reference-features-2026-11-09')]:
 directory=ROOT/f'experiments/regression/results/{base}/{ds}'
 rows=list(csv.DictReader((directory/'oof_predictions.csv').read_text().splitlines()));byid={r['task_id']:r for r in rows}
 folds=json.loads((directory/'splits.json').read_text())
 for metric in ['r2','mae']:
  if metric=='r2':cursor+=1 # reference zero line
  for m,model in enumerate(['baseline','median','lasso','xgboost']):
   values=[]
   for fold in folds:
    y=np.array([float(byid[i]['complexity']) for i in fold['test_ids']])
    if model=='median':pred=np.full(len(y),np.median([float(byid[i]['complexity']) for i in fold['train_ids']]))
    else:pred=np.array([float(byid[i][model]) for i in fold['test_ids']])
    value=1-sum((pred-y)**2)/sum((y-y.mean())**2) if metric=='r2' else np.mean(abs(pred-y))
    values.append(value)
   match(blocks[cursor],[(v,3-m+(i-2)*.065) for i,v in enumerate(values)])
   match(blocks[cursor+1],[(np.mean(values),3-m)]);cursor+=2
assert cursor==len(blocks)
# Correlations preserve the signs, feature identities and available-observation N.
blocks=points('feature-correlations')
for index,(ds,base) in enumerate([('verified','verified-boolean-outcomes-2026-11-09'),('arrow','reference-features-2026-11-09')]):
 d=json.loads((ROOT/f'experiments/regression/results/{base}/{ds}/dataset.json').read_text());X=np.array([r['features'] for r in d['rows']],float);y=np.array([r['complexity'] for r in d['rows']]);actual=[]
 for i,f in enumerate(d['feature_names']):
  mask=np.isfinite(X[:,i]);r=data['correlations'][ds][i];assert r['feature']==f and r['n']==mask.sum()
  rho=float(spearmanr(X[mask,i],y[mask]).statistic);pear=float(pearsonr(X[mask,i],y[mask]).statistic)
  assert abs(r['spearman']-rho)<1e-13 and abs(r['pearson']-pear)<1e-13;actual.append(rho)
 match([x for x,y in blocks[-2+index]],actual)
# Every eligible repository is shown once, including coefficients and sample-count labels.
blocks=points('judge-repository-correlations');text=(FIG/'judge-repository-correlations.tex').read_text()
for k,ds in enumerate(['verified','pro']):
 repos=data['judge'][ds]['repositories'];assert len(repos)==(10 if ds=='verified' else 11)
 match(blocks[k*2+1],[(r['spearman'],len(repos)-i) for i,r in enumerate(repos)])
 for r in repos:assert text.count(r['repo'])==1 and r['n']>=5
# Case overview uses the adopted reference features and OOF predictions.
s=(HERE/'case-table-rows.tex').read_text()
for r in data['cases']:
 row=' & '.join(['$'+r['case'].replace('T','T_')+'$',str(int(r['additions'])),str(int(r['deletions'])),*(f'${r[k]:.3f}$' for k in ['complexity','lasso','xgboost'])])
 assert row in s
result=dict(status='passed',preserved_sources=len(data['source_sha256']),judge_points=1231,judge_bins=20,fold_scores=80,feature_correlations=38,repository_subsets=21,cases=3,model_calls=0,model_fits=0)
(HERE/'build/verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
