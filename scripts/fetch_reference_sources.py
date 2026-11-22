# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = []
# ///

"""Build a temporary feature-extraction bundle from public pinned source identities.

Benchmark texts and repository source files are fetched/provided at runtime, never
stored in the public Git repository. Every patch/base file is checked before use.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]


def checked(content, expected, identity):
    if hashlib.sha256(content).hexdigest()!=expected:
        raise ValueError(f'Source bytes differ for {identity}; obtain the matching version instead of substituting a newer patch')
    return content.decode('utf-8')


def download(url):
    with urlopen(url,timeout=60) as response:return response.read()


def build(output, verified_jsonl=None, arrow_patches=None, base_cache=None, task_id=None):
    if not output.resolve().is_relative_to(ROOT/'work'):
        raise ValueError('Downloaded raw source must be written under ignored work/')
    if output.exists():raise FileExistsError(output)
    identities=json.loads((ROOT/'data/reference-inputs.json').read_text())
    if task_id:
        identities=[r for r in identities if r['task_id'] in task_id]
        if {r['task_id'] for r in identities}!=set(task_id):raise ValueError('Unknown task ID')
    verified={}
    if any(r['dataset']=='verified' for r in identities):
        if verified_jsonl is None:raise ValueError('Export the official Verified test split to JSONL and supply --verified-jsonl; see data/README.md')
        verified={r['instance_id']:r for r in map(json.loads,verified_jsonl.read_text().splitlines())}
    features={}
    for dataset,filename,key in [('verified','task_metrics.jsonl','task_name'),('arrow','pr_metrics.jsonl','pr_number')]:
        for line in (ROOT/'data/features'/dataset/filename).read_text().splitlines():
            record=json.loads(line);task=record[key] if dataset=='verified' else 'pr-'+str(record[key])
            features[task]={k:v for k,v in record.items() if k not in {'complexity','reference'}}
    output.parent.mkdir(parents=True,exist_ok=True)
    # Incomplete downloads are clearly separated from the requested bundle.
    partial=output.with_name(output.name+'.partial')
    with gzip.open(partial,'wt',encoding='utf-8') as stream:
        for identity in identities:
            task=identity['task_id']
            if identity['dataset']=='verified':
                record=verified[task]
                assert record['repo']==identity['repo'] and record['base_commit']==identity['base_commit']
                raw=record['patch'].encode()
            elif arrow_patches:
                raw=(arrow_patches/(task+'.diff')).read_bytes()
            else:
                raw=download(identity['reference_patch_url'])
            patch=checked(raw,identity['reference_patch_sha256'],task+' reference patch')
            files={}
            for path,expected in identity['base_files'].items():
                if base_cache:
                    raw=(base_cache/identity['repo']/identity['base_commit']/path).read_bytes()
                else:
                    url=f'https://raw.githubusercontent.com/{identity["repo"]}/{identity["base_commit"]}/{quote(path,safe="/")}'
                    raw=download(url)
                files[path]=checked(raw,expected,task+' '+path)
            stream.write(json.dumps(dict(dataset=identity['dataset'],task_id=task,repo=identity['repo'],
                base_commit=identity['base_commit'],reference_patch=patch,base_files=files,historical_record=features[task]))+'\n')
    partial.replace(output)
    print(f'Wrote {len(identities)} checked task inputs to {output}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verified-jsonl',type=Path)
    p.add_argument('--arrow-patches',type=Path,help='Optional matching public PR diffs, named pr-N.diff')
    p.add_argument('--base-cache',type=Path,help='Optional public source cache: repo/base_commit/path')
    p.add_argument('--task-id',action='append',help='Optional subset for checking source acquisition')
    a=p.parse_args();build(a.output,a.verified_jsonl,a.arrow_patches,a.base_cache,a.task_id)
