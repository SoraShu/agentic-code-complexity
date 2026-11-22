import unittest
import json
from pathlib import Path
import evaluate_verified_outcomes as run

class OutcomePolicyTests(unittest.TestCase):
    def test_nonboolean_is_not_a_failed_test(self):
        r=run.scored_outcomes(['True','False','error','missing'])
        self.assertEqual(r['complexity'],0.5)
        self.assertEqual((r['n_valid'],r['n_reports'],r['n_errors'],r['n_missing']),(2,4,1,1))
        for labels in [['error','missing'],['True','timeout']]:
            with self.assertRaises(AssertionError):run.scored_outcomes(labels)

    def test_actual_targets_preserve_features_and_success_counts(self):
        prior=json.loads((run.ev.ROOT/'results/reference-features-2026-11-09/verified/dataset.json').read_text())
        current=run.build_dataset(prior)
        self.assertEqual(sum(r['denominator'] for r in current['rows']),12941)
        self.assertEqual(sum(r['n_errors'] for r in current['rows']),50)
        self.assertEqual(sum(r['n_missing'] for r in current['rows']),9)
        self.assertEqual(sum(a['complexity']!=b['complexity'] for a,b in zip(prior['rows'],current['rows'])),45)
        for a,b in zip(prior['rows'],current['rows']):
            self.assertEqual(a['features'],b['features'])
            self.assertEqual(a['n_resolved'],b['n_resolved'])
            self.assertEqual(a['task_id'],b['task_id'])

if __name__=='__main__':unittest.main()
