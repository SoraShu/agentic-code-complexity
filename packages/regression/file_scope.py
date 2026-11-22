"""Change statistics at an explicit reference-patch file scope."""
from pathlib import PurePosixPath


def is_test_path(path):
    """The study's source-only rule: exclude the repository-root tests/ tree."""
    return path.startswith('tests/')


def file_change_counts(files):
    """Count hunk content, retaining changed lines beginning with +++ or ---."""
    rows = []
    for file in files:
        additions = deletions = 0
        in_hunk = False
        for line in file['patch'].splitlines():
            if line.startswith('@@ '):
                in_hunk = True
            elif in_hunk and line.startswith('+'):
                additions += 1
            elif in_hunk and line.startswith('-'):
                deletions += 1
        rows.append({'path': file['path'], 'status': file['status'],
                     'additions': additions, 'deletions': deletions})
    return rows


def change_features(files):
    top, second = set(), set()
    for file in files:
        parts = PurePosixPath(file['path']).parts
        top.add(parts[0] if len(parts) > 1 else '.')
        if len(parts) > 2:
            second.add('/'.join(parts[:2]))
    additions = sum(f['additions'] for f in files)
    deletions = sum(f['deletions'] for f in files)
    return {'lines_changed': additions + deletions, 'additions': additions,
            'deletions': deletions, 'files_changed': len(files),
            'dispersity': {'top_level_dirs': len(top), 'second_level_dirs': len(second),
                          'top_level_list': sorted(top), 'second_level_list': sorted(second)}}


def source_only(record, changed_files, aggregate):
    """Filter existing per-file measurements; missingness applies to retained files."""
    retained_changes = [f for f in changed_files if not is_test_path(f['path'])]
    files = [f for f in record['reference']['files'] if not is_test_path(f['path'])]
    measured = [f for f in files if f['status'] == 'measured']
    unavailable = len(measured) != len(files)
    reference = {**record['reference'], 'files': files, 'n_python_files': len(files),
                 'structural_status': 'unavailable' if unavailable else 'measured' if files else 'no_python_files'}
    return {**record, **change_features(retained_changes),
            'complexity': aggregate(measured, unavailable=unavailable),
            'reference': reference, 'feature_scope': 'exclude repository-root tests/',
            'change_files': retained_changes,
            'excluded_test_paths': [f['path'] for f in changed_files if is_test_path(f['path'])]}
