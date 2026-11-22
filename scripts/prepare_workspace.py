# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = []
# ///

"""Prepare disposable analysis inputs. Private Verified data require an explicit path."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]


def merge(source, target, include_readmes=True):
    source, target = Path(source), Path(target)
    if not source.exists():
        raise FileNotFoundError(source)
    files = [source] if source.is_file() else sorted(source.rglob('*'))
    for src in files:
        if not src.is_file() or {'.venv', '__pycache__', '.pytest_cache'}.intersection(src.parts):
            continue
        if not include_readmes and src.name == 'README.md':
            continue
        if src.is_symlink():
            raise ValueError(f'Symlink input: {src}')
        dest = target if source.is_file() else target/src.relative_to(source)
        if dest.exists() and dest.read_bytes() != src.read_bytes():
            raise ValueError(f'Conflicting inputs: {dest}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def prepare(destination, private_root=None):
    destination = destination.resolve()
    if not destination.is_relative_to(REPO/'work'):
        raise ValueError('Use a destination inside the Git-ignored work/ directory')
    if destination.exists():
        raise FileExistsError(f'Choose a new directory: {destination}')
    private = private_root.resolve()/'workspace' if private_root else None
    private_paths = [
        'experiments/regression/inputs/verified/results',
        'experiments/regression/results/verified-boolean-outcomes-2026-11-09/verified',
        'experiments/regression/results/reference-features-2026-11-09/verified',
        'localdataset/results/swe-bench-verified',
        'localdataset/results/judge/opus47_showtests_run.jsonl',
        'localdataset/results/judge/opus47_showtests_20shot.jsonl',
    ]
    if private:
        for name in private_paths:
            if not (private/name).exists():
                raise FileNotFoundError(private/name)
    destination.mkdir(parents=True)
    for source, target in [('packages/regression','experiments/regression'),
                           ('packages/judge','localdataset')]:
        merge(REPO/source, destination/target)
    reg = destination/'experiments/regression'
    # Features and outcomes share inputs/arrow, but their source READMEs differ.
    # Keep those descriptions in the repository instead of colliding in work/.
    merge(REPO/'data/features', reg/'inputs-reference', include_readmes=False)
    merge(REPO/'data/features', reg/'inputs', include_readmes=False)
    merge(REPO/'data/outcomes/arrow', reg/'inputs/arrow', include_readmes=False)
    merge(REPO/'data/results/regression', reg/'results/reference-features-2026-11-09')
    merge(REPO/'data/targets/pro.json', destination/'localdataset/data/pro-targets.json')
    merge(REPO/'data/targets/pro-complete-traceable.json',
          destination/'localdataset/data/pro-complete-traceable.json')
    merge(REPO/'data/results/judge/pro-zero-shot.jsonl',
          destination/'localdataset/results/judge/pro_opus47_showtests.jsonl')
    if private:
        for name in private_paths:
            merge(private/name, destination/name)
    for folder, manifest in [('inputs','inputs/manifest.json'),
                             ('inputs-reference','inputs-reference/manifest.json')]:
        hashes = {str(p.relative_to(reg/folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((reg/folder).rglob('*')) if p.is_file() and p.suffix in {'.csv','.json','.jsonl'}}
        value = {'files':hashes} if folder == 'inputs-reference' else [
            dict(snapshot=str(Path(folder)/name),sha256=value) for name,value in hashes.items()]
        (reg/manifest).write_text(json.dumps(value,indent=2)+'\n')
    (destination/'workspace.json').write_text(json.dumps({'private_verified_inputs':bool(private),
        'description':'Final saved-result inputs; no raw public datasets or internal construction code copied'},indent=2)+'\n')
    return destination


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,default=REPO/'work/public')
    p.add_argument('--private-root',type=Path)
    a=p.parse_args();print(prepare(a.destination,a.private_root))
