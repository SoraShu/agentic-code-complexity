"""Checks for postprocessing saved folds; no model fitting is performed."""
import unittest

import numpy as np

import report


class MedianBaselineTests(unittest.TestCase):
    def test_uses_training_median_including_even_sample_sizes(self):
        y = np.array([0.0, 0.0, 0.6, 1.0, 1.0, 1.0])
        folds = [(np.array([0, 1, 2, 3]), np.array([4, 5])),
                 (np.array([4, 5]), np.array([0, 1, 2, 3]))]
        prediction = report.median_predictions(y, folds)
        np.testing.assert_allclose(prediction, [1, 1, 1, 1, 0.3, 0.3])
        changed = y.copy()
        changed[4:] = -10
        np.testing.assert_array_equal(report.median_predictions(changed, folds)[4:], prediction[4:])

    def test_reports_fold_means_not_pooled_errors(self):
        y = np.array([0.0, 0.0, 0.6, 1.0, 1.0, 1.0])
        folds = [(np.array([0, 1, 2, 3]), np.array([4, 5])),
                 (np.array([4, 5]), np.array([0, 1, 2, 3]))]
        prediction = np.array([1, 1, 1, 1, 0.4, 0.4])
        _, metrics = report.with_median(y, {"baseline": prediction}, folds)
        # Median fold MAEs: 0.7 and 0.6. Their unweighted mean is 0.65,
        # whereas pooling the six predictions gives 3.8/6.
        self.assertAlmostEqual(metrics["median"]["mae_mean"], 0.65)
        self.assertAlmostEqual(metrics["median"]["mae_std"], 0.05)
        self.assertAlmostEqual(metrics["median"]["pooled_oof"]["mae"], 3.8 / 6)
        self.assertEqual(metrics["median"]["mae_improvement_vs_median_pct"], 0)
        np.testing.assert_array_equal(prediction, [1, 1, 1, 1, 0.4, 0.4])


if __name__ == "__main__":
    unittest.main()
