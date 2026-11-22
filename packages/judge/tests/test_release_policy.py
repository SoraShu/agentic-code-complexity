"""Outcome policies, cache identity and the recovered primary prompt."""
import csv
from pathlib import Path
import pytest
from llm_judge.ground_truth import EXPECTED_COLS, build_or_load
from llm_judge.prompt import SYSTEM_PROMPT


def write_report(root, states):
    root.mkdir(exist_ok=True)
    for i, status in enumerate(states):
        with (root / f'report-{i}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(EXPECTED_COLS))
            writer.writeheader()
            row = {k: 0 for k in EXPECTED_COLS}
            row.update(RunId=str(i), BenchmarkName='synthetic', InstanceId='task', Resolved=status)
            writer.writerow(row)


def test_boolean_policy_and_historical_labels_do_not_share_cache(tmp_path):
    reports, cache = tmp_path / 'reports', tmp_path / 'cache.json'
    write_report(reports, ['True', 'False', 'error', 'missing'])
    actual = build_or_load(reports, cache).iloc[0]
    assert actual.n_runs == 2 and actual.success_rate == .5
    historical = build_or_load(reports, cache, policy='all_reports').iloc[0]
    assert historical.n_runs == 4 and historical.success_rate == .25
    assert build_or_load(reports, cache).iloc[0].success_rate == .5
    write_report(reports, ['True', 'True', 'error', 'missing'])
    assert build_or_load(reports, cache).iloc[0].success_rate == 1


def test_unknown_status_is_not_silently_a_failure(tmp_path):
    write_report(tmp_path, ['unexpected'])
    with pytest.raises(ValueError, match='Unknown outcome status'):
        build_or_load(tmp_path)


def test_primary_rubric_has_the_recovered_three_steps():
    assert '1. why_easy:' in SYSTEM_PROMPT
    assert '2. why_hard:' in SYSTEM_PROMPT
    assert '3. weigh:' in SYSTEM_PROMPT
    assert 'coverage_check' not in SYSTEM_PROMPT
    assert 'LOUD' not in SYSTEM_PROMPT and 'SILENT' not in SYSTEM_PROMPT
