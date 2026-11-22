# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = []
# ///

"""Generate the final complexity histogram from saved integer outcome counts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def main(workspace):
    pre=r'''\documentclass[tikz,border=1mm]{standalone}
\usepackage{lmodern,fontspec,pgfplots}
\setmainfont{TeX Gyre Termes}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
'''
    plots=[];stats={}
    for i,(dataset,run,title) in enumerate([
        ('verified','verified-boolean-outcomes-2026-11-09','SWE-bench Verified'),
        ('arrow','reference-features-2026-11-09','Arrow')]):
        rows=json.loads((workspace/f'experiments/regression/results/{run}/{'arrow_oracle_pass' if dataset=='arrow' else dataset}/dataset.json').read_text())['rows']
        counts=Counter(min(9,10*(r['denominator']-r['n_resolved'])//r['denominator']) for r in rows)
        assert sum(counts.values())==len(rows)
        median=statistics.median(1-r['n_resolved']/r['denominator'] for r in rows)
        stats[dataset]=dict(n=len(rows),counts=[counts[j] for j in range(10)],median=median)
        coordinates=' '.join(f'({j/10:.1f},{100*counts[j]/len(rows):.8f})' for j in range(10))+' (1,0)'
        plots.append(r'\begin{axis}[at={('+str(i*66)+r'mm,0)},anchor=south west,width=63mm,height=51mm,'
            r'font=\fontsize{9}{11}\selectfont,xmin=0,xmax=1,ymin=0,ymax=75,'
            r'axis lines=left,xtick={0,0.5,1},ytick={0,25,50,75},'
            r'title={'+title+r' ($N='+str(len(rows))+r'$)},xlabel={$C(t)$},'
            +(r'ylabel={Tasks (\%)},' if i==0 else '')
            +r'ymajorgrids=true,grid style={gray!20},tick align=outside]'+'\n'
            +r'\addplot[ybar interval,fill=blue!25!gray!30,draw=black!60] coordinates {'+coordinates+'};\n'
            +rf'\draw[dashed,blue!60!black] (axis cs:{median},0) -- (axis cs:{median},73);'+'\n'
            +r'\node[anchor=north east,font=\fontsize{8}{10}\selectfont] at (rel axis cs:0.98,0.97) {Median '+f'{median:.3f}'+'};\n'
            +r'\end{axis}'+'\n')
    output=workspace/'generated/figures';output.mkdir(exist_ok=True,parents=True)
    (output/'complexity-distributions.tex').write_text(pre+''.join(plots)+r'\end{tikzpicture}\end{document}'+'\n')
    (output/'distribution.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(output/'complexity-distributions.tex')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('workspace',type=Path)
    main(p.parse_args().workspace.resolve())
