"""Check fold isolation, outcome policies, and out-of-fold accounting."""
import os
import unittest

import evaluate as ev
import numpy as np
from sklearn.model_selection import KFold


class EvaluationTests(unittest.TestCase):
    def test_outer_holdout_cannot_change_fitted_model(self):
        rng = np.random.default_rng(13)
        X = rng.normal(size=(40, 3))
        y = 0.5 + X[:, 0] * 0.1
        X[[1, 6, 31], 0] = np.nan
        train, test = np.arange(30), np.arange(30, 40)
        X_changed, y_changed = X.copy(), y.copy()
        X_changed[test] += 1000
        y_changed[test] = -1000
        for model in ("lasso", "xgboost"):
            _, original = ev.fit_outer_fold(model, X, y, train, test, 1, 42, tiny=True)
            _, changed = ev.fit_outer_fold(model, X_changed, y_changed, train, test, 1, 42, tiny=True)
            self.assertEqual(original, changed)
            median = np.nanmedian(X[train], axis=0)
            np.testing.assert_allclose(original["imputer_statistics"], median)
            if model == "lasso":
                imputed = np.where(np.isnan(X[train]), median, X[train])
                np.testing.assert_allclose(original["scaler_mean"], imputed.mean(axis=0))
                self.assertEqual(original["scaler_n_samples"], len(train))

    def test_inner_scaler_is_a_pipeline_step(self):
        search = ev.make_search("lasso", 1, 42)
        self.assertEqual(list(search.estimator.named_steps), ["impute", "scale", "model"])
        self.assertEqual(search.param_grid["model__alpha"], ev.ALPHAS)
        xgb = ev.make_search("xgboost", 1, 42)
        self.assertEqual(list(xgb.estimator.named_steps), ["impute", "model"])

    def test_arrow_sensitivity_and_denominators(self):
        historical, _, y, excluded = ev.load_dataset("arrow")
        strict, _, _, _ = ev.load_dataset("arrow", "oracle_pass")
        failures, _, y_fail, excluded_fail = ev.load_dataset("arrow", "errors_as_failures")
        self.assertEqual((len(historical), len(strict), len(excluded)), (59, 55, 14))
        self.assertEqual(excluded, excluded_fail)
        self.assertEqual(sum(r["denominator"] == 9 for r in historical), 2)
        self.assertTrue(all(r["denominator"] == 10 for r in failures))
        self.assertEqual(np.count_nonzero(y != y_fail), 2)
        self.assertTrue(all(r["oracle_pass"] for r in strict))

    def test_fold_accounting_and_baseline(self):
        y = np.arange(59) / 58
        folds = list(KFold(5, shuffle=True, random_state=42).split(y))
        covered = np.zeros(len(y), dtype=int)
        prediction = np.full(len(y), np.nan)
        for train, test in folds:
            self.assertFalse(set(train) & set(test))
            covered[test] += 1
            prediction[test] = y[train].mean()
        np.testing.assert_array_equal(covered, np.ones(len(y)))
        self.assertTrue(np.isfinite(prediction).all())
        summary = ev.summarize(y, {"baseline": prediction}, folds)["baseline"]
        expected = np.mean([np.abs(y[test] - y[train].mean()).mean() for train, test in folds])
        self.assertAlmostEqual(summary["mae_mean"], expected)
        self.assertEqual(summary["mae_improvement_pct"], 0)


if __name__ == "__main__":
    unittest.main()
