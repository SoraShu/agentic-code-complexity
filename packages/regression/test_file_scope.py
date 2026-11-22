"""Check file-scope boundaries that affect counts and missing-data handling."""
import unittest

from extract_reference_features import aggregate, file_metrics, patch_files
from file_scope import change_features, file_change_counts, source_only


class FileScopeTests(unittest.TestCase):
    def test_count_changed_separators_and_keep_deleted_non_test_files(self):
        patch = ('diff --git a/docs/a.rst b/docs/a.rst\n--- a/docs/a.rst\n+++ b/docs/a.rst\n'
                 '@@ -1 +1 @@\n----\n++++\n'
                 'diff --git a/old.py b/old.py\ndeleted file mode 100644\n'
                 '--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x = 1\n'
                 'diff --git a/tests/test_a.py b/tests/test_a.py\n--- a/tests/test_a.py\n'
                 '+++ b/tests/test_a.py\n@@ -1 +1 @@\n-assert a == 1\n+assert a == 2\n')
        files = file_change_counts(patch_files(patch))
        self.assertEqual(change_features(files)['lines_changed'], 5)
        result = source_only({'reference': {'files': []}}, files, aggregate)
        self.assertEqual((result['additions'], result['deletions'], result['files_changed']), (1, 2, 2))
        self.assertEqual(result['dispersity']['top_level_list'], ['.', 'docs'])
        self.assertEqual(result['dispersity']['second_level_dirs'], 0)
        self.assertEqual(result['complexity']['loc']['total'], 0)
        self.assertIsNone(result['complexity']['loc']['mean'])

    def test_excluded_failure_does_not_hide_available_source_metrics(self):
        measured = {'path': 'arrow/a.py', 'status': 'measured', **file_metrics('x = 1 + 2\n')}
        unavailable = {'path': 'tests/test_a.py', 'status': 'reference_patch_unapplicable'}
        record = {'reference': {'files': [measured, unavailable]}}
        result = source_only(record, [], aggregate)
        self.assertEqual(result['reference']['structural_status'], 'measured')
        self.assertGreater(result['complexity']['halstead_volume']['total'], 0)
        unavailable['path'] = 'arrow/b.py'
        result = source_only(record, [], aggregate)
        self.assertEqual(result['reference']['structural_status'], 'unavailable')
        self.assertIsNone(result['complexity']['loc']['total'])
        result = source_only({'reference': {'files': [dict(measured, path='tests/test_a.py')]}}, [], aggregate)
        self.assertEqual(result['reference']['structural_status'], 'no_python_files')
        self.assertEqual(result['complexity']['cyclomatic']['mean'], 0)
        self.assertIsNone(result['complexity']['maintainability_index']['total'])


if __name__ == '__main__':
    unittest.main()
