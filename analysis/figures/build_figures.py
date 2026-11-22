"""Read-only analysis of saved results; generate editable PGFPlots and vector PDFs.

Figures are authored at the thesis's 130 mm text width, with 9 pt labels and
TeX Gyre Termes matching existing thesis figures. No estimators are fitted.
Inputs are joined by task ID. Source checksums and exact plotted summaries are
recorded in plot-data.json. Existing outcomes/predictions are never overwritten.
"""
from pathlib import Path
from collections import defaultdict
from decimal import Decimal
import csv,hashlib,json,math,os,subprocess,sys
import numpy as np
from scipy.stats import spearmanr,pearsonr
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace',type=Path,required=True)
parser.add_argument('--output-dir',type=Path)
parser.add_argument('--compile',action='store_true')
args=parser.parse_args()
ROOT=args.workspace.resolve()
HERE=(args.output_dir or ROOT/'generated/figures').resolve()
FIG=HERE
BUILD=HERE/'build'
BUILD.mkdir(parents=True,exist_ok=True)
SOURCES={}
def read(p):
    SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return p.read_text()
def load(p):return json.loads(read(p))
def tex(s):return s.replace('_',r'\_')
def coord(points):return ' '.join(f'({float(x):.10f},{float(y):.10f})' for x,y in points)
def node(x,y,text,opts=''):
    return rf'\node[{opts}] at ({x}mm,{y}mm) {{{text}}};'+'\n'
PRE=r'''% Generated from saved experimental results; editable vector figure.
\documentclass[tikz,border=1mm]{standalone}
\usepackage{lmodern,fontspec,amsmath,amssymb,pgfplots}
\setmainfont{TeX Gyre Termes}
\setmonofont{TeX Gyre Cursor}
\pgfplotsset{compat=1.18}
\definecolor{verified}{HTML}{315F7D}
\definecolor{arrow}{HTML}{AA5936}
\definecolor{ink}{HTML}{263746}
\begin{document}
\begin{tikzpicture}[font=\fontsize{9}{11}\selectfont,text=ink]
'''
POST='\\end{tikzpicture}\n\\end{document}\n'
COMMON=r'''scale only axis,axis lines=left,tick align=outside,
font=\fontsize{9}{11}\selectfont,
axis line style={black!65},tick style={black!65},
label style={font=\fontsize{9}{11}\selectfont},
title style={font=\fontsize{9.5}{11}\selectfont,align=center},
clip=false'''
def axis(x,y,w,h,opts):return rf'\begin{{axis}}[at={{({x}mm,{y}mm)}},anchor=south west,width={w}mm,height={h}mm,'+COMMON+','+opts+']\n'
def plot(points,opts):return '\\addplot['+opts+'] coordinates {'+coord(points)+'};\n'
def save(name,body):
    (FIG/f'{name}.tex').write_text(PRE+body+POST)

# Recover the exact adopted targets from source reports, excluding non-Boolean entries.
truth={}
for ds in ['verified','pro']:
    if ds=='pro':
        rows=load(ROOT/'localdataset/data/pro-targets.json')
        truth[ds]={r['instance_id']:r['n_resolved']/r['n_valid'] for r in rows}
    else:
        outcomes=defaultdict(list)
        for p in sorted((ROOT/'localdataset/results/swe-bench-verified').glob('report-*.csv')):
            for r in csv.DictReader(read(p).splitlines()):
                if r['Resolved'] in ['True','False']:outcomes[r['InstanceId']].append(r['Resolved']=='True')
        truth[ds]={k:sum(v)/len(v) for k,v in outcomes.items()}
judges={};raw_judges={}
for ds,fn in [('verified','opus47_showtests_run.jsonl'),('pro','pro_opus47_showtests.jsonl')]:
    rows=[json.loads(l) for l in read(ROOT/'localdataset/results/judge'/fn).splitlines()]
    assert len(rows)==len({r['instance_id'] for r in rows})==(500 if ds=='verified' else 731)
    rows=sorted(rows,key=lambda r:r['instance_id']);raw_judges[ds]=rows
    p=np.array([r['p_success'] for r in rows]);y=np.array([truth[ds][r['instance_id']] for r in rows])
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    # Decimal bin membership preserves intended boundaries at exactly .1, .2, ...
    membership=[min(int(Decimal(str(v))*10),9) for v in p]
    bins=[]
    for i in range(10):
        mask=np.array(membership)==i
        bins.append(dict(index=i,left=i/10,right=(i+1)/10,n=int(mask.sum()),mean_pred=float(p[mask].mean()) if mask.any() else None,mean_observed=float(y[mask].mean()) if mask.any() else None))
    byrepo=defaultdict(list)
    for r in rows:byrepo[r['repo']].append(r)
    repos=[]
    for repo,group in byrepo.items():
        if len(group)<5:continue
        v=[r['p_success'] for r in group];t=[truth[ds][r['instance_id']] for r in group]
        repos.append(dict(repo=repo,n=len(group),spearman=float(spearmanr(v,t).statistic),mean_pred=float(np.mean(v)),mean_observed=float(np.mean(t))))
    repos.sort(key=lambda r:(-r['spearman'],r['repo']))
    judges[ds]=dict(n=len(rows),spearman=float(spearmanr(p,y).statistic),mae=float(np.mean(abs(p-y))),bins=bins,repositories=repos)

# Independently check saved predictions and training-fold constant baselines.
sys.path.insert(0,str(ROOT/'experiments/regression'))
import report
reg={};datasets={};correlations={};oof={}
for ds,base in [('verified','verified-boolean-outcomes-2026-11-09'),('arrow','source-only')]:
    directory=ROOT/f'experiments/regression/results/{base}/{'arrow_oracle_pass' if ds=='arrow' else ds}'
    for name in ['dataset.json','splits.json','results.json','oof_predictions.csv']:read(directory/name)
    result=load(directory/'results.json');data=load(directory/'dataset.json');datasets[ds]=data
    rows,y,preds,folds=report.audit(directory,result)
    median,metrics=report.with_median(y,preds,folds)
    reg[ds]=metrics;oof[ds]={r['task_id']:r for r in rows}
    X=np.array([r['features'] for r in data['rows']],float)
    correlations[ds]=[]
    for j,f in enumerate(data['feature_names']):
        keep=np.isfinite(X[:,j]);a=X[keep,j];b=y[keep]
        correlations[ds].append(dict(feature=f,n=int(keep.sum()),spearman=float(spearmanr(a,b).statistic),pearson=float(pearsonr(a,b).statistic)))
case_ids=['django__django-16502','sympy__sympy-20916','django__django-15098']
cases=[]
for i,iid in enumerate(case_ids,1):
    row=oof['verified'][iid];d=next(r for r in datasets['verified']['rows'] if r['task_id']==iid)
    features=dict(zip(datasets['verified']['feature_names'],d['features']))
    cases.append(dict(case=f'T{i}',task_id=iid,additions=features['additions'],deletions=features['deletions'],complexity=float(row['complexity']),lasso=float(row['lasso']),xgboost=float(row['xgboost']),fold=int(row['outer_fold'])))

# 1. Judge predictions, empirical task outcomes, descriptive bin means and counts.
s=''
for j,ds in enumerate(['verified','pro']):
    x=12+65*j;info=judges[ds];color='verified' if j==0 else 'arrow'
    title=('(a) Verified' if j==0 else '(b) Pro')+rf' ($N={info["n"]}$)'
    s+=axis(x,15,49,49,r'xmin=0,xmax=1,ymin=0,ymax=1,xtick={0,0.5,1},xticklabels={},ytick={0,0.5,1},ymajorgrids=true,grid style={black!10},'+('ylabel={Empirical success rate}' if j==0 else ''))
    s+=plot([(0,0),(1,1)],'black!60,dashed,no marks,line width=0.65pt')
    s+=plot([(r['p_success'],truth[ds][r['instance_id']]) for r in raw_judges[ds]],'only marks,mark=*,mark size=0.9pt,mark options={draw=none,fill=black,fill opacity=0.22}')
    s+=plot([(b['mean_pred'],b['mean_observed']) for b in info['bins'] if b['n']],f'only marks,mark=square*,mark size=2.2pt,color={color}')
    s+='\\end{axis}\n'
    s+=node(x+24.5,73,title,'font=\\fontsize{9.5}{11}\\selectfont')
    s+=node(x+24.5,68.5,rf'$\rho_s={info["spearman"]:.3f}$; MAE $={info["mae"]:.3f}$','font=\\fontsize{8.5}{10}\\selectfont')
    s+=axis(x,0,49,9,'xmin=0,xmax=1,ymin=0,ymax=190,xtick={0,0.5,1},ytick=\\empty,xlabel={Predicted success probability},'+('ylabel={$n$},' if j==0 else '')+'ylabel style={rotate=-90},axis y line=none')
    s+=plot([(b['left']+.05,b['n']) for b in info['bins']],f'ybar,bar width=3.5mm,draw=none,fill={color}!25,nodes near coords,nodes near coords style={{font=\\fontsize{{7.5}}{{9}}\\selectfont,text=ink}},point meta=y')
    s+='\\end{axis}\n'
s+=node(66,-17,r'\textcolor{black!50}{$\bullet$} Task\quad\textcolor{verified}{$\blacksquare$}\,\textcolor{arrow}{$\blacksquare$} Bin mean\quad--\,--\ $y=x$', 'font=\\fontsize{8.5}{10}\\selectfont')
# \blacksquare uses amssymb; use a rule in the legend instead.
s=s.replace(r'$\blacksquare$',r'\rule{1.4mm}{1.4mm}')
save('judge-prediction-diagnostics',s)

# 2. Main regression results: 5 observed fold values, not confidence intervals.
s=''
models=['baseline','median','lasso','xgboost'];labels=['Mean baseline','Median baseline','Lasso','XGBoost']
for row,ds in enumerate(['verified','arrow']):
    y=51 if row==0 else 6
    for col,metric in enumerate(['r2','mae']):
        x=27 if col==0 else 86;w=42
        opts=('xmin=-1.8,xmax=0.6,xtick={-1.5,-1,-0.5,0,0.5},xlabel={$R^2$ (higher is better)},' if col==0 else 'xmin=0.05,xmax=0.45,xtick={0.1,0.2,0.3,0.4},xlabel={MAE (lower is better)},')
        title=f'({chr(97+row*2+col)}) '+('Verified' if ds=='verified' else 'Arrow')
        s+=axis(x,y,w,27,opts+'ymin=-0.5,ymax=3.5,ytick={0,1,2,3},yticklabels={'+(','.join(labels[::-1]) if col==0 else ',,,')+'},xmajorgrids=true,grid style={black!10},title={'+title+'}')
        if metric=='r2':s+=plot([(0,-.5),(0,3.5)],'black!60,dashed,no marks')
        for k,model in enumerate(models):
            v=[f[metric] for f in reg[ds][model]['folds']]
            lo,hi=(-1.8,.6) if metric=='r2' else (.05,.45)
            assert all(lo<=z<=hi for z in v),(ds,model,metric,v)
            s+=plot([(z,3-k+(i-2)*.065) for i,z in enumerate(v)],'only marks,mark=*,mark size=1.6pt,verified,opacity=0.65')
            s+=plot([(reg[ds][model][metric+'_mean'],3-k)],'only marks,mark=diamond*,mark size=2.5pt,black,mark options={fill=white,line width=0.9pt}')
        s+='\\end{axis}\n'
s+=node(77,-13,r'\textcolor{verified}{$\bullet$} Outer fold\qquad$\diamond$ Mean of five folds','font=\\fontsize{8.5}{10}\\selectfont')
save('regression-fold-performance',s)

# 3. Signed Spearman coefficients with feature-specific sample sizes.
s='';features=datasets['verified']['feature_names'];positions=[];v=0
for i,f in enumerate(features):
    if i in [4,6,9,12,15,17]:v+=.6
    positions.append(v);v+=1
Y=positions[-1];pos=[Y-z for z in positions];h=94
s+=axis(37,0,68,h,'xmin=-0.4,xmax=0.4,ymin=-0.6,ymax='+str(Y+.6)+',xtick={-0.4,-0.2,0,0.2,0.4},ytick={'+','.join(map(str,pos))+'},yticklabels={'+','.join(r'\texttt{'+tex(f)+'}' for f in features)+'},yticklabel style={font=\\fontsize{8.5}{10}\\selectfont},xlabel={Spearman correlation with $C(t)$},xmajorgrids=true,grid style={black!10}')
s+=plot([(0,-.6),(0,Y+.6)],'black!70,no marks,line width=0.7pt')
for i in [4,6,9,12,15,17]:
    sep=(pos[i-1]+pos[i])/2
    s+=plot([(-.4,sep),(.4,sep)],'black!12,no marks')
for ds,color,mark,offset in [('verified','verified','*',.11),('arrow','arrow','triangle*',-.11)]:
    s+=plot([(r['spearman'],p+offset) for r,p in zip(correlations[ds],pos)],f'only marks,mark={mark},mark size=2pt,{color}')
s+='\\end{axis}\n'
s+=node(113,h+3,r'$N_{\rm V}$','font=\\fontsize{8.5}{10}\\selectfont')+node(125,h+3,r'$N_{\rm A}$','font=\\fontsize{8.5}{10}\\selectfont')
for k,p in enumerate(pos):
    mm=(p+.6)/(Y+1.2)*h
    s+=node(113,mm,str(correlations['verified'][k]['n']),'font=\\fontsize{8.5}{10}\\selectfont')
    s+=node(125,mm,str(correlations['arrow'][k]['n']),'font=\\fontsize{8.5}{10}\\selectfont')
s+=node(66,h+10,r'\textcolor{verified}{$\bullet$} Verified\qquad\textcolor{arrow}{$\blacktriangle$} Arrow')
# Filled triangle matches the Arrow marks.
save('feature-correlations',s)

# 4. All eligible repository subsets; full names, coefficients and N.
s=''
for j,ds in enumerate(['verified','pro']):
    repos=judges[ds]['repositories'];n=len(repos);h=n*3.8;y=61 if j==0 else 0;color='verified' if j==0 else 'arrow'
    title=f'({chr(97+j)}) '+('Verified' if j==0 else 'Pro')+f' ({n} repositories)'
    s+=axis(42,y,72,h,'xmin=0,xmax=1,ymin=0.5,ymax='+str(n+.5)+',xtick={0,0.25,0.5,0.75,1},ytick={'+','.join(str(n-i) for i in range(n))+'},yticklabels={'+','.join(tex(r['repo']) for r in repos)+'},yticklabel style={font=\\fontsize{8.5}{10}\\selectfont},xlabel={Spearman correlation},xmajorgrids=true,grid style={black!10},title={'+title+'}')
    rho=judges[ds]['spearman'];s+=plot([(rho,.5),(rho,n+.5)],f'{color},dashed,no marks,line width=0.8pt')
    s+=plot([(r['spearman'],n-i) for i,r in enumerate(repos)],f'only marks,mark=*,mark size=2pt,{color},nodes near coords,nodes near coords align={{horizontal}},point meta=x,nodes near coords style={{font=\\fontsize{{8}}{{9}}\\selectfont,text=ink,fill=white,inner sep=0.5pt,xshift=3pt,/pgf/number format/fixed,/pgf/number format/precision=3,/pgf/number format/fixed zerofill}}')
    s+='\\end{axis}\n'
    s+=node(124,y+h+2,'$N$')
    for i,r in enumerate(repos):s+=node(124,y+(n-i-.5)/n*h,str(r['n']),'font=\\fontsize{8.5}{10}\\selectfont')
s+=node(80,-12,r'Dashed lines: dataset-level correlations','font=\\fontsize{8.5}{10}\\selectfont')
save('judge-repository-correlations',s)

# Exact cases come from the same adopted OOF files.
case_lines=[]
for r in cases:
    case_lines.append('        '+' & '.join([r'$'+r['case'].replace('T','T_')+'$',str(int(r['additions'])),str(int(r['deletions'])),*(f'${r[k]:.3f}$' for k in ['complexity','lasso','xgboost'])])+r' \\')
(HERE/'case-table-rows.tex').write_text('\n'.join(case_lines)+'\n')
summary=dict(judge=judges,regression=reg,correlations=correlations,cases=cases,binning='Decimal boundaries [0,.1),...,[.9,1]; equal task weighting; bin marker at mean prediction and mean observed rate',model_fits=0,model_calls=0,source_sha256=SOURCES)
(HERE/'plot-data.json').write_text(json.dumps(summary,indent=2)+'\n')
for path,h in SOURCES.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
print('Generated 4 figures and case rows from',len(SOURCES),'unchanged source files; no fitting or model calls.')
if '--compile' in sys.argv:
    env=os.environ.copy()
    # TeX Live fonts may be installed without being registered in system fontconfig.
    # Discover their location from TeX, without depending on the author's /tmp files.
    if 'FONTCONFIG_FILE' not in env:
        from xml.sax.saxutils import escape
        font_dirs = set()
        for font in ['texgyretermes-regular.otf', 'texgyrecursor-regular.otf']:
            found = subprocess.run(['kpsewhich', font], capture_output=True, text=True, check=False).stdout.strip()
            if found:
                font_dirs.add(str(Path(found).parent))
        if font_dirs:
            config = BUILD/'fonts.conf'
            cache = BUILD/'font-cache'; cache.mkdir(exist_ok=True)
            config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig>'
                + '<include ignore_missing="yes">/etc/fonts/fonts.conf</include>'
                + ''.join('<dir>'+escape(d)+'</dir>' for d in sorted(font_dirs))
                + '<cachedir>'+escape(str(cache))+'</cachedir></fontconfig>')
            env['FONTCONFIG_FILE'] = str(config)
    for name in ['judge-prediction-diagnostics','regression-fold-performance','feature-correlations','judge-repository-correlations']:
        with (BUILD/f'{name}-compile.txt').open('w') as log:
            subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={BUILD}',str(FIG/f'{name}.tex')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        (FIG/f'{name}.pdf').write_bytes((BUILD/f'{name}.pdf').read_bytes())
    print('Compiled all four vector PDFs.')
