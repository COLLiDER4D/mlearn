# pyright: reportMissingImports=false
import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure package import works when running tests from repository root.
ROOT_PARENT = Path(__file__).resolve().parents[2]  # .../capstone
PKG_ROOT = Path(__file__).resolve().parents[1]     # .../capstone/microMlKit
for p in (ROOT_PARENT, PKG_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from micromlkit.linear_model.linear_regression import LinearRegression
from micromlkit.linear_model.logistic_regression import LogisticRegression
from micromlkit.linear_model.ridge import Ridge
from micromlkit.linear_model.lasso import Lasso


class TestLogisticRegression(unittest.TestCase):
    def test_fit_predict_and_score_on_separable_data(self):
        X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 0, 1, 1, 1])

        model = LogisticRegression(learning_rate=0.2, n_iters=5000, tol=1e-8).fit(X, y)
        preds = model.predict(X)

        np.testing.assert_array_equal(preds, y)
        self.assertAlmostEqual(model.score(X, y), 1.0, places=10)
        self.assertEqual(model.n_features_in_, 1)

    def test_predict_proba_shape_and_row_sums(self):
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])

        model = LogisticRegression().fit(X, y)
        proba = model.predict_proba(X)

        self.assertEqual(proba.shape, (4, 2))
        np.testing.assert_allclose(np.sum(proba, axis=1), np.ones(4), atol=1e-10)
        self.assertTrue(np.all((proba >= 0.0) & (proba <= 1.0)))

    def test_fit_raises_for_non_binary_targets(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 2.0])

        with self.assertRaises(ValueError) as ctx:
            LogisticRegression().fit(X, y)

        self.assertIn("binary", str(ctx.exception).lower())


class TestRidge(unittest.TestCase):
    def test_alpha_zero_matches_linear_regression(self):
        X = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 1.0],
            [3.0, 2.0],
        ])
        y = 3.0 + 2.0 * X[:, 0] - 1.0 * X[:, 1]

        lr = LinearRegression().fit(X, y)
        ridge = Ridge(alpha=0.0).fit(X, y)

        np.testing.assert_allclose(ridge.predict(X), lr.predict(X), atol=1e-10)
        np.testing.assert_allclose(ridge.coef_, lr.coef_, atol=1e-10)
        self.assertAlmostEqual(ridge.intercept_, lr.intercept_, places=10)

    def test_regularization_shrinks_coefficient_norm(self):
        X = np.array([
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 3.0],
            [4.0, 5.0],
            [5.0, 4.0],
        ])
        y = 1.5 + 3.0 * X[:, 0] - 2.0 * X[:, 1]

        lr = LinearRegression().fit(X, y)
        ridge = Ridge(alpha=100.0).fit(X, y)

        self.assertLess(np.linalg.norm(ridge.coef_), np.linalg.norm(lr.coef_))

    def test_fit_raises_on_negative_alpha(self):
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])

        with self.assertRaises(ValueError) as ctx:
            Ridge(alpha=-1.0).fit(X, y)

        self.assertIn("non-negative", str(ctx.exception).lower())


class TestLasso(unittest.TestCase):
    def test_fit_predict_high_r2_on_linear_data(self):
        X = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 2.0],
        ])
        y = 2.0 + 1.5 * X[:, 0] - 0.5 * X[:, 1]

        model = Lasso(alpha=0.01, max_iter=5000, tol=1e-8).fit(X, y)
        score = model.score(X, y)

        self.assertGreater(score, 0.99)

    def test_l1_penalty_can_zero_irrelevant_feature(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 3))
        y = 4.0 * X[:, 0] - 2.0 * X[:, 1]  # feature 2 is irrelevant

        model = Lasso(alpha=0.2, max_iter=4000, tol=1e-6).fit(X, y)

        self.assertAlmostEqual(model.coef_[2], 0.0, places=4)

    def test_fit_raises_on_invalid_hyperparameters(self):
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])

        with self.assertRaises(ValueError):
            Lasso(alpha=-0.1).fit(X, y)
        with self.assertRaises(ValueError):
            Lasso(max_iter=0).fit(X, y)
        with self.assertRaises(ValueError):
            Lasso(tol=-1e-6).fit(X, y)


if __name__ == "__main__":
    unittest.main()
