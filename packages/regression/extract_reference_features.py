"""Reconstruct benchmark-reference Python files and extract independent metrics.

Prepare an immutable source bundle from the audited local source cache, then
re-extract offline from that bundle. Never execute task code or call an agent.
"""
from __future__ import annotations
import argparse
import ast
import gzip
import hashlib
import json
import math
import re
import subprocess
import tempfile
import tomllib
import warnings
from pathlib import Path

from cognitive_complexity.api import get_cognitive_complexity
from radon.complexity import cc_visit
from radon.metrics import h_visit
from radon.raw import analyze
from radon.visitors import ComplexityVisitor
from file_scope import file_change_counts, source_only

ROOT = Path(__file__).resolve().parent
GROUPS = {
    'cyclomatic': ('total', 'mean', 'max', 'n_functions'),
    'cognitive': ('total', 'mean', 'max', 'n_functions'),
    'maintainability_index': ('total', 'mean', 'min', 'max', 'n_files'),
    'halstead_volume': ('total', 'mean'), 'loc': ('total', 'mean'),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def patch_files(patch):
    """Retain exact hunk bytes; only interpret file headers outside hunks."""
    files = []
    for block in re.split(r'(?m)(?=^diff --git )', patch):
        if not block.startswith('diff --git '):
            continue
        header = block.splitlines()[0]
        old, new = header[len('diff --git a/'):].split(' b/', 1)
        for path in (old, new):
            assert not Path(path).is_absolute() and '..' not in Path(path).parts
        preamble = block.split('@@', 1)[0]
        files.append({'old_path': old, 'path': new,
                      'status': 'removed' if 'deleted file mode' in preamble else
                                'added' if 'new file mode' in preamble else 'modified',
                      'patch': block})
    return files


def file_metrics(source):
    """HV=0 is valid; LLOC is independent of MI's logarithm domain."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        tree = ast.parse(source)
        blocks = cc_visit(source)
        cc = [b.complexity for b in blocks]
        cog = [get_cognitive_complexity(n) for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        hv = h_visit(source).total.volume
        lloc = analyze(source).lloc
        mi_cc = ComplexityVisitor.from_ast(tree).total_complexity
    mi = 171 - 5.2 * math.log(hv) - 0.23 * mi_cc - 16.2 * math.log(lloc) if hv > 0 and lloc > 0 else None
    return {'cc': cc, 'cog': cog, 'hv': hv, 'lloc': lloc, 'mi': mi,
            'mi_status': 'defined' if mi is not None else 'undefined_log_domain'}


def aggregate(files, unavailable=False):
    if unavailable:
        return {group: {k: None for k in keys} for group, keys in GROUPS.items()}
    result = {}
    for key, group in [('cc', 'cyclomatic'), ('cog', 'cognitive')]:
        values = [v for f in files for v in f[key]]
        result[group] = {'total': sum(values), 'mean': sum(values) / len(values) if values else 0,
                         'max': max(values, default=0), 'n_functions': len(values)}
    mi = [f['mi'] for f in files if f['mi'] is not None]
    result['maintainability_index'] = {
        'total': sum(mi) if mi else None, 'mean': sum(mi) / len(mi) if mi else None,
        'min': min(mi) if mi else None, 'max': max(mi) if mi else None, 'n_files': len(mi)}
    for key, group in [('hv', 'halstead_volume'), ('lloc', 'loc')]:
        values = [f[key] for f in files]
        result[group] = {'total': sum(values), 'mean': sum(values) / len(values) if values else None}
    return result


def reconstruct(record, file_scope='source-only'):
    evidence, measured = [], []
    unavailable = False
    with tempfile.TemporaryDirectory(prefix='reference-features-') as tmp:
        tmp = Path(tmp)
        for f in patch_files(record['reference_patch']):
            if not f['path'].endswith('.py') or f['status'] == 'removed':
                continue
            info = {'path': f['path'], 'patch_sha256': sha(f['patch'].encode())}
            if f['status'] != 'added':
                original = record['base_files'][f['old_path']].encode()
                dest = tmp / f['old_path']
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(original)
                info['base_sha256'] = sha(original)
            applied = subprocess.run(['git', 'apply', '-'], cwd=tmp,
                                     input=f['patch'].encode(), capture_output=True)
            if applied.returncode:
                unavailable = True
                info.update(status='reference_patch_unapplicable', error=applied.stderr.decode())
            else:
                source = (tmp / f['path']).read_bytes()
                info['reference_sha256'] = sha(source)
                try:
                    metrics = file_metrics(source.decode())
                    info.update(status='measured', **metrics)
                    measured.append(metrics)
                except (SyntaxError, ValueError) as error:
                    unavailable = True
                    info.update(status='parse_error', error=str(error))
            evidence.append(info)
    result = {k: v for k, v in record['historical_record'].items()
              if k != 'complexity' and not k.startswith('fn_level')}
    result['complexity'] = aggregate(measured, unavailable)
    result['reference'] = {
        'repo': record['repo'], 'base_commit': record['base_commit'],
        'reference_patch_sha256': sha(record['reference_patch'].encode()),
        'structural_status': 'unavailable' if unavailable else 'measured' if evidence else 'no_python_files',
        'n_python_files': len(evidence), 'files': evidence,
    }
    if file_scope == 'source-only':
        result = source_only(result, file_change_counts(patch_files(record['reference_patch'])), aggregate)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-bundle', type=Path, required=True)
    parser.add_argument('--file-scope', choices=['source-only', 'complete-pr'], default='source-only')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = args.output / 'sources.jsonl.gz'
    bundle.write_bytes(args.source_bundle.read_bytes())
    rows = {'verified': [], 'arrow': []}
    with gzip.open(bundle, 'rt', encoding='utf-8') as stream:
        for i, line in enumerate(stream, 1):
            record = json.loads(line)
            rows[record['dataset']].append(reconstruct(record, args.file_scope))
            if i % 50 == 0:
                print(f'Extracted {i} reference tasks', flush=True)
    manifest = {'source_bundle_sha256': sha(bundle.read_bytes()), 'extractor_sha256': sha(Path(__file__).read_bytes()),
                'file_scope': args.file_scope,
                'file_scope_sha256': sha((ROOT / 'file_scope.py').read_bytes()),
                'definition': 'Strict per-file application of benchmark gold patch to benchmark base; no PR-head fallback.',
                'missingness': 'MI undefined when HV or LLOC is nonpositive; no-file means missing; any unavailable Python file makes all structural aggregates missing.',
                'files': {}}
    for dataset, filename in [('verified', 'task_metrics.jsonl'), ('arrow', 'pr_metrics.jsonl')]:
        dest = args.output / dataset / filename
        dest.parent.mkdir()
        dest.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows[dataset]))
        manifest['files'][str(dest.relative_to(args.output))] = sha(dest.read_bytes())
        print(dataset, 'tasks', len(rows[dataset]), 'unavailable',
              [r.get('task_name', r.get('pr_number')) for r in rows[dataset] if r['reference']['structural_status'] == 'unavailable'])
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
