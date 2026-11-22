"""Exercise nested group isolation and preserve the ordinary split definition."""
import unittest

import evaluate as ev
import numpy as np
from sklearn.model_selection import KFold


class GroupedSplitsTests(unittest.TestCase):
    def test_nested_group_isolation_and_reject_split_group(self):
        ids = [f'task-{i}' for i in range(55)]
        groups = {t: f'base-{i // 2}' for i, t in enumerate(ids)}
        splits = ev.nested_splits(ids, 42, groups)
        ev.validate_splits(ids, splits, groups)
        self.assertEqual(splits, ev.nested_splits(ids, 42, groups))
        for outer in splits:
            train, test = outer['train_ids'], outer['test_ids']
            self.assertFalse({groups[t] for t in train} & {groups[t] for t in test})
        first = splits[0]
        broken = dict(groups)
        broken[first['test_ids'][0]] = broken[first['train_ids'][0]]
        with self.assertRaises(AssertionError):
            ev.validate_splits(ids, splits, broken)

    def test_ordinary_folds_are_unchanged(self):
        ids = [f'task-{i}' for i in range(55)]
        splits = ev.nested_splits(ids, 42)
        ev.validate_splits(ids, splits)
        for actual, (train, test) in zip(splits, KFold(5, shuffle=True, random_state=42).split(ids)):
            self.assertEqual(actual['test_ids'], [ids[i] for i in test])
            for inner, (a, b) in zip(actual['inner_splits'], KFold(5, shuffle=True, random_state=42).split(train)):
                self.assertEqual(inner['validation_ids'], [ids[train[i]] for i in b])


if __name__ == '__main__':
    unittest.main()
