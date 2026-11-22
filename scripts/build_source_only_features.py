# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = [
#     "numpy==2.5.0", "scikit-learn==1.9.0", "xgboost==3.3.0",
#     "joblib==1.6.0", "radon==6.0.1", "cognitive-complexity==1.3.0",
# ]
# ///
"""Build source-only features from existing per-file measurements and pinned diffs."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'packages/regression'))
from evaluate import FEATURES, feature_vector
from extract_reference_features import aggregate, patch_files
from file_scope import change_features, file_change_counts, source_only


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(REPO / 'work') or output.exists():
        parser.error('Use a new output directory under work/')
    originals, input_files = {}, [REPO / 'data/reference-inputs.json']
    for dataset, filename, key in [('arrow', 'pr_metrics.jsonl', 'pr_number'),
                                   ('verified', 'task_metrics.jsonl', 'task_name')]:
        path = REPO / 'data/features' / dataset / filename
        input_files.append(path)
        for row in map(json.loads, path.read_text().splitlines()):
            task = f"pr-{row[key]}" if dataset == 'arrow' else row[key]
            originals[dataset, task] = row
    refs = {(r['dataset'], r['task_id']): r for r in json.loads(input_files[0].read_text())}
    rows, evidence, seen = [], [], set()
    verified_unchanged = 0
    with gzip.open(args.source_bundle, 'rt') as stream:
        for line in stream:
            record = json.loads(line)
            key = record['dataset'], record['task_id']
            assert key not in seen
            seen.add(key)
            assert hashlib.sha256(record['reference_patch'].encode()).hexdigest() == refs[key]['reference_patch_sha256']
            original = originals[key]
            files = file_change_counts(patch_files(record['reference_patch']))
            full = change_features(files)
            assert feature_vector({**original, **full}) == feature_vector(original), key
            # Independent Git parser check of hunk line totals before any filtering.
            numstat = subprocess.check_output(['git', 'apply', '--numstat', '-'],
                                             input=record['reference_patch'].encode()).decode().splitlines()
            assert len(numstat) == len(files), key
            totals = [sum(int(line.split('\t')[i]) for line in numstat) for i in (0, 1)]
            assert totals == [full['additions'], full['deletions']], key
            filtered = source_only(original, files, aggregate)
            if key[0] == 'verified':
                assert feature_vector(filtered) == feature_vector(original), key
                assert not filtered['excluded_test_paths'], key
                verified_unchanged += 1
            else:
                rows.append(filtered)
                evidence.append({'task_id': key[1], 'reference_patch_sha256': refs[key]['reference_patch_sha256'],
                                 'complete_pr_change_files': files})
    assert seen == set(originals) and len(rows) == 73 and verified_unchanged == 500
    rows.sort(key=lambda r: r['pr_number'])
    evidence.sort(key=lambda r: int(r['task_id'].split('-')[1]))
    output.mkdir(parents=True)
    (output / 'arrow').mkdir()
    (output / 'arrow/pr_metrics.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
    (output / 'arrow/change-counts.json').write_text(json.dumps(evidence, indent=2) + '\n')
    manifest = {
        'definition': 'All 19 features exclude repository-root tests/ paths. Non-test documentation/configuration changes remain in size/directory counts; structural metrics cover retained Python files.',
        'source_bundle_sha256': digest(args.source_bundle),
        'inputs': {str(p.relative_to(REPO)): digest(p) for p in input_files},
        'code': {str(p.relative_to(REPO)): digest(p) for p in
                 [Path(__file__), REPO / 'packages/regression/file_scope.py',
                  REPO / 'packages/regression/extract_reference_features.py']},
        'arrow_tasks': len(rows), 'verified_unchanged_tasks': verified_unchanged,
        'verified_features': 'data/features/verified/task_metrics.jsonl',
        'files': {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Built {len(rows)} Arrow source-only records; all {verified_unchanged} Verified feature vectors unchanged.')


if __name__ == '__main__':
    main()
