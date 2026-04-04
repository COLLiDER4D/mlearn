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

from micromlkit.neighbors import KNNClassifier, KNNRegressor


class TestKNNClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [6.0, 5.0],
        ])
        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_fit_predict_basic_contract(self):
        model = KNNClassifier(n_neighbors=3).fit(self.X, self.y)
        preds = model.predict(self.X)

        self.assertEqual(preds.shape, (self.X.shape[0],))
        self.assertEqual(model.n_features_in_, 2)
        self.assertGreaterEqual(model.score(self.X, self.y), 1.0)

    def test_fit_returns_self(self):
        model = KNNClassifier(n_neighbors=3)
        returned = model.fit(self.X, self.y)
        self.assertIs(returned, model)

    def test_predict_before_fit_raises(self):
        model = KNNClassifier()
        with self.assertRaises(ValueError) as ctx:
            model.predict(self.X)
        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_predict_feature_mismatch_raises(self):
        model = KNNClassifier(n_neighbors=3).fit(self.X, self.y)
        with self.assertRaises(ValueError) as ctx:
            model.predict(np.array([[1.0, 2.0, 3.0]]))
        self.assertIn("features", str(ctx.exception).lower())

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            KNNClassifier(n_neighbors=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            KNNClassifier(metric="invalid").fit(self.X, self.y)
        with self.assertRaises(ValueError):
            KNNClassifier(metric="minkowski", p=0.5).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            KNNClassifier(n_neighbors=10).fit(self.X, self.y)

    def test_all_metrics_supported(self):
        metric_configs = [
            ("euclidean", {}),
            ("manhattan", {}),
            ("cosine", {}),
            ("chebyshev", {}),
            ("minkowski", {"p": 3}),
        ]

        for metric, kwargs in metric_configs:
            with self.subTest(metric=metric):
                model = KNNClassifier(n_neighbors=1, metric=metric, **kwargs).fit(self.X, self.y)
                preds = model.predict(self.X)
                self.assertEqual(preds.shape[0], self.X.shape[0])

    def test_tie_breaking_is_deterministic(self):
        X = np.array([[-1.0], [1.0], [-2.0], [2.0]])
        y = np.array([1, 0, 1, 0])
        model = KNNClassifier(n_neighbors=4).fit(X, y)

        pred = model.predict(np.array([[0.0]]))[0]
        self.assertEqual(pred, 0)


class TestKNNRegressor(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        self.y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    def test_fit_predict_basic_contract(self):
        model = KNNRegressor(n_neighbors=2).fit(self.X, self.y)
        preds = model.predict(np.array([[1.1], [2.9]]))

        np.testing.assert_allclose(preds, np.array([1.5, 2.5]), atol=1e-10)
        self.assertEqual(model.n_features_in_, 1)

    def test_fit_returns_self(self):
        model = KNNRegressor(n_neighbors=2)
        returned = model.fit(self.X, self.y)
        self.assertIs(returned, model)

    def test_predict_before_fit_raises(self):
        model = KNNRegressor()
        with self.assertRaises(ValueError) as ctx:
            model.predict(self.X)
        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_predict_feature_mismatch_raises(self):
        model = KNNRegressor(n_neighbors=2).fit(self.X, self.y)
        with self.assertRaises(ValueError) as ctx:
            model.predict(np.array([[1.0, 2.0]]))
        self.assertIn("features", str(ctx.exception).lower())

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            KNNRegressor(n_neighbors=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            KNNRegressor(metric="invalid").fit(self.X, self.y)
        with self.assertRaises(ValueError):
            KNNRegressor(metric="minkowski", p=0.5).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            KNNRegressor(n_neighbors=10).fit(self.X, self.y)

    def test_all_metrics_supported(self):
        metric_configs = [
            ("euclidean", {}),
            ("manhattan", {}),
            ("cosine", {}),
            ("chebyshev", {}),
            ("minkowski", {"p": 3}),
        ]

        for metric, kwargs in metric_configs:
            with self.subTest(metric=metric):
                model = KNNRegressor(n_neighbors=1, metric=metric, **kwargs).fit(self.X, self.y)
                preds = model.predict(self.X)
                self.assertEqual(preds.shape[0], self.X.shape[0])

    def test_single_sample_with_k_one(self):
        X = np.array([[2.0, 3.0]])
        y = np.array([7.5])
        model = KNNRegressor(n_neighbors=1).fit(X, y)

        pred = model.predict(np.array([[2.0, 3.0]]))[0]
        self.assertAlmostEqual(pred, 7.5, places=10)

    def test_score_runs_with_inherited_mixin(self):
        model = KNNRegressor(n_neighbors=1).fit(self.X, self.y)
        score = model.score(self.X, self.y)
        self.assertAlmostEqual(score, 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
