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


class TestLinearRegression(unittest.TestCase):
    def test_fit_predict_and_params_on_perfect_data(self):
        # y = 3 + 2*x1 - 1*x2
        X = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 1.0],
            [3.0, 2.0],
        ])
        y = 3.0 + 2.0 * X[:, 0] - 1.0 * X[:, 1]

        model = LinearRegression().fit(X, y)
        preds = model.predict(X)

        np.testing.assert_allclose(preds, y, atol=1e-10)
        self.assertAlmostEqual(model.intercept_, 3.0, places=10)
        np.testing.assert_allclose(model.coef_, np.array([2.0, -1.0]), atol=1e-10)
        self.assertEqual(model.n_features_in_, 2)
        self.assertAlmostEqual(model.score(X, y), 1.0, places=10)

    def test_predict_before_fit_raises(self):
        model = LinearRegression()
        X = np.array([[1.0], [2.0]])

        with self.assertRaises(ValueError) as ctx:
            model.predict(X)

        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_fit_raises_when_X_and_y_lengths_differ(self):
        model = LinearRegression()
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0])

        with self.assertRaises(ValueError) as ctx:
            model.fit(X, y)

        self.assertIn("same number of samples", str(ctx.exception).lower())

    def test_fit_raises_when_X_not_2d(self):
        model = LinearRegression()
        X = np.array([1.0, 2.0, 3.0])  # 1D, invalid
        y = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError) as ctx:
            model.fit(X, y)

        self.assertIn("2d array", str(ctx.exception).lower())

    def test_fit_raises_when_y_not_1d(self):
        model = LinearRegression()
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[1.0], [2.0], [3.0]])  # 2D, invalid

        with self.assertRaises(ValueError) as ctx:
            model.fit(X, y)

        self.assertIn("1d array", str(ctx.exception).lower())

    def test_predict_raises_on_feature_count_mismatch(self):
        X_train = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
        y_train = np.array([1.0, 2.0, 3.0])

        model = LinearRegression().fit(X_train, y_train)

        X_bad = np.array([[1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError) as ctx:
            model.predict(X_bad)

        self.assertIn("features", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
