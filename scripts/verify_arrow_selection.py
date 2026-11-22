# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = []
# ///

"""Check the saved Arrow screening, construction and oracle cohorts; no model calls."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    selection = ROOT / 'data/selection'
    screening = json.loads((selection/'arrow-screening.json').read_text())['rows']
    candidates = [int(n) for n in (selection/'arrow-candidates.txt').read_text().split()]
    assert len(screening) == len({r['pr_number'] for r in screening}) == 701
    assert len(candidates) == len(set(candidates)) == 100
    ranked = sorted(screening, key=lambda r: r['lines_changed'], reverse=True)
    assert [r['pr_number'] for r in ranked[:100]] == candidates
    assert ranked[99]['lines_changed'] == 135 and ranked[100]['lines_changed'] == 133
    construction = json.loads((selection/'arrow-construction.json').read_text())['rows']
    assert [r['pr_number'] for r in construction] == candidates
    counts = Counter(r['construction'] for r in construction)
    assert counts == dict(constructed=73, issue_exit=2, scaffold_exit=2,
                          implementation_exit=16, implementation_reported_failure=7)
    oracle = json.loads((ROOT/'data/outcomes/arrow/oracle.json').read_text())
    outcomes = {int(t.split('__')[0][3:]): bool(float(value))
                for e in oracle['stats']['evals'].values()
                for value, tasks in e['reward_stats']['reward'].items() for t in tasks}
    assert set(outcomes) == {r['pr_number'] for r in construction if r['construction'] == 'constructed'}
    assert all(r['oracle_pass'] == outcomes.get(r['pr_number']) for r in construction)
    data = json.loads((ROOT/'data/results/regression/arrow_oracle_pass/dataset.json').read_text())
    assert {r['task_id'] for r in data['rows']} == {f'pr-{n}' for n, passed in outcomes.items() if passed}
    assert len(data['rows']) == 55
    source = json.loads((ROOT/'data/sources.json').read_text())['arrow']
    assert source['screened_prs'] == 701 and source['construction_inputs'] == 100
    assert source['candidate_tasks'] == 73 and source['analysis_tasks'] == 55
    print('Verified Arrow selection: 701 screened -> 100 inputs -> 73 constructed -> 55 oracle passes.')
    print('Construction terminal categories:', dict(counts))


if __name__ == '__main__':
    main()
