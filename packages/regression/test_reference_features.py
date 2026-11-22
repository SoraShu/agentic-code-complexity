"""Scientific boundary checks: zero HV, independent LLOC, exact reference patching."""
import unittest
from extract_reference_features import file_metrics, aggregate, reconstruct

class ReferenceFeaturesTests(unittest.TestCase):
    def test_zero_volume_keeps_lloc_and_participates_in_hv_mean(self):
        zero = file_metrics('value = 1\n')
        positive = file_metrics('def f(x):\n    return x + 1\n')
        self.assertEqual(zero['hv'], 0)
        self.assertEqual(zero['lloc'], 1)
        self.assertIsNone(zero['mi'])
        measured = aggregate([zero, positive])
        self.assertEqual(measured['loc']['total'], zero['lloc'] + positive['lloc'])
        self.assertEqual(measured['halstead_volume']['mean'], positive['hv'] / 2)
        self.assertEqual(measured['maintainability_index']['n_files'], 1)
        self.assertIsNone(aggregate([zero])['maintainability_index']['total'])
        self.assertIsNone(aggregate([])['loc']['mean'])
        self.assertEqual(aggregate([])['loc']['total'], 0)

    def test_strict_reference_reconstruction_and_failure_without_head_fallback(self):
        patch = 'diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-value = 1\n+value = 1 + 2\n'
        row = {'repo': 'example/repo', 'base_commit': 'base', 'reference_patch': patch,
               'base_files': {'a.py': 'value = 1\n'}, 'historical_record': {'task_name': 'task'}}
        result = reconstruct(row)
        self.assertEqual(result['reference']['structural_status'], 'measured')
        self.assertGreater(result['complexity']['halstead_volume']['total'], 0)
        row['base_files']['a.py'] = 'a_different_base = 10\n'
        failed = reconstruct(row)
        self.assertEqual(failed['reference']['structural_status'], 'unavailable')
        self.assertTrue(all(v is None for group in failed['complexity'].values() for v in group.values()))

if __name__ == '__main__':
    unittest.main()
