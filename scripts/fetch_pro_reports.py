# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = []
# ///

"""Download the nine independently matched public Pro reports at a pinned commit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
COLS=['RunId','BenchmarkName','InstanceId','Resolved','TotalSteps','TotalInputTokens',
      'TotalOutputTokens','TotalCachedInputTokens','TotalTokens']


def convert(data, model, destination):
    if not isinstance(data,dict) or not all(type(v) is bool for v in data.values()):
        raise ValueError('Upstream outcomes must be explicit JSON Boolean values')
    run='scaleai-'+model.replace('/','_').replace('.','_')
    with destination.open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(COLS)
        for instance,resolved in data.items():
            writer.writerow([run,'SWE-Bench Pro',instance,resolved,0,0,0,0,0])


def main(output):
    if not output.resolve().is_relative_to(ROOT/'work'):
        raise ValueError('Store downloaded raw reports under ignored work/')
    sources=json.loads((ROOT/'data/sources.json').read_text())['pro']
    output.mkdir(parents=True,exist_ok=False)
    expected={r['file']:r['sha256'] for r in sources['historical_report_hashes']}
    for row in sources['downloadable_reports']:
        with urlopen(row['url'],timeout=60) as response:content=response.read()
        if hashlib.sha256(content).hexdigest()!=row['sha256']:
            raise ValueError(f'Upstream bytes changed: {row["model"]}')
        name='report-scaleai-'+row['model'].replace('/','_').replace('.','_')+'.csv'
        dest=output/name;convert(json.loads(content),row['model'],dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=expected[name]:
            raise ValueError(f'Converted CSV differs from retained report: {name}')
        print(name)
    print('Downloaded 9 matched reports. Exact regeneration of the 13-report targets also needs the 4 unmapped historical reports listed in data/sources.json.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'work/pro-reports')
    main(p.parse_args().output)
