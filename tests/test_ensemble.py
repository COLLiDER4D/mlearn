# pyright: reportMissingImports=false
import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure package import works when running tests from repository root.
ROOT_PARENT = Path(__file__).resolve().parents[2]  # repository parent directory
PKG_ROOT = Path(__file__).resolve().parents[1]     # repository root directory
for p in (ROOT_PARENT, PKG_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from micromlkit.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)


class TestRandomForestClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [0.0, 0.1],
            [0.1, 0.0],
            [0.2, 0.2],
            [1.0, 0.9],
            [0.9, 1.0],
            [0.8, 0.85],
        ])
        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_fit_predict_basic_contract(self):
        model = RandomForestClassifier(n_estimators=25, random_state=42)
        preds = model.fit(self.X, self.y).predict(self.X)

        self.assertEqual(preds.shape, (self.X.shape[0],))
        self.assertEqual(model.n_features_in_, 2)
        self.assertTrue(hasattr(model, "estimators_"))
        self.assertTrue(hasattr(model, "feature_indices_"))
        self.assertTrue(hasattr(model, "classes_"))
        self.assertGreaterEqual(model.score(self.X, self.y), 0.90)

    def test_fit_returns_self(self):
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        returned = model.fit(self.X, self.y)
        self.assertIs(returned, model)

    def test_predict_before_fit_raises(self):
        model = RandomForestClassifier()
        with self.assertRaises(ValueError) as ctx:
            model.predict(self.X)
        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_predict_feature_mismatch_raises(self):
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(self.X, self.y)
        with self.assertRaises(ValueError) as ctx:
            model.predict(np.array([[1.0, 2.0, 3.0]]))
        self.assertIn("features", str(ctx.exception).lower())

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            RandomForestClassifier(n_estimators=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            RandomForestClassifier(max_depth=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            RandomForestClassifier(min_samples_split=1).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            RandomForestClassifier(min_samples_leaf=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            RandomForestClassifier(random_state="seed").fit(self.X, self.y)
        with self.assertRaises(ValueError):
            RandomForestClassifier(bootstrap="yes").fit(self.X, self.y)
        with self.assertRaises(ValueError):
            RandomForestClassifier(max_features=0).fit(self.X, self.y)

    def test_random_state_is_deterministic(self):
        model1 = RandomForestClassifier(n_estimators=20, random_state=7).fit(self.X, self.y)
        model2 = RandomForestClassifier(n_estimators=20, random_state=7).fit(self.X, self.y)

        np.testing.assert_array_equal(model1.predict(self.X), model2.predict(self.X))


class TestRandomForestRegressor(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 2.0],
            [5.0, 2.0],
        ])
        self.y = 1.5 + 2.0 * self.X[:, 0] + 0.25 * self.X[:, 1]

    def test_fit_predict_basic_contract(self):
        model = RandomForestRegressor(n_estimators=30, random_state=42)
        preds = model.fit(self.X, self.y).predict(self.X)

        self.assertEqual(preds.shape, (self.X.shape[0],))
        self.assertEqual(model.n_features_in_, 2)
        self.assertTrue(hasattr(model, "estimators_"))
        self.assertTrue(hasattr(model, "feature_indices_"))
        self.assertGreaterEqual(model.score(self.X, self.y), 0.90)

    def test_fit_returns_self(self):
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        returned = model.fit(self.X, self.y)
        self.assertIs(returned, model)

    def test_predict_before_fit_raises(self):
        model = RandomForestRegressor()
        with self.assertRaises(ValueError) as ctx:
            model.predict(self.X)
        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_predict_feature_mismatch_raises(self):
        model = RandomForestRegressor(n_estimators=10, random_state=0).fit(self.X, self.y)
        with self.assertRaises(ValueError) as ctx:
            model.predict(np.array([[1.0, 2.0, 3.0]]))
        self.assertIn("features", str(ctx.exception).lower())

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            RandomForestRegressor(n_estimators=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            RandomForestRegressor(max_depth=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            RandomForestRegressor(min_samples_split=1).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            RandomForestRegressor(min_samples_leaf=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            RandomForestRegressor(random_state="seed").fit(self.X, self.y)
        with self.assertRaises(ValueError):
            RandomForestRegressor(bootstrap="yes").fit(self.X, self.y)
        with self.assertRaises(ValueError):
            RandomForestRegressor(max_features=0).fit(self.X, self.y)

    def test_random_state_is_deterministic(self):
        model1 = RandomForestRegressor(n_estimators=20, random_state=11).fit(self.X, self.y)
        model2 = RandomForestRegressor(n_estimators=20, random_state=11).fit(self.X, self.y)

        np.testing.assert_allclose(model1.predict(self.X), model2.predict(self.X), atol=1e-12)


class TestGradientBoostingRegressor(unittest.TestCase):
    def setUp(self):
        self.X = np.linspace(-2.0, 2.0, 30).reshape(-1, 1)
        self.y = self.X[:, 0] ** 2 + 0.1 * self.X[:, 0]

    def test_fit_predict_basic_contract(self):
        model = GradientBoostingRegressor(
            n_estimators=60,
            learning_rate=0.1,
            max_depth=2,
            random_state=42,
        )
        preds = model.fit(self.X, self.y).predict(self.X)

        self.assertEqual(preds.shape, (self.X.shape[0],))
        self.assertEqual(model.n_features_in_, 1)
        self.assertTrue(hasattr(model, "estimators_"))
        self.assertTrue(hasattr(model, "init_pred_"))
        self.assertGreaterEqual(model.score(self.X, self.y), 0.90)

    def test_fit_returns_self(self):
        model = GradientBoostingRegressor(n_estimators=20, random_state=42)
        returned = model.fit(self.X, self.y)
        self.assertIs(returned, model)

    def test_predict_before_fit_raises(self):
        model = GradientBoostingRegressor()
        with self.assertRaises(ValueError) as ctx:
            model.predict(self.X)
        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_predict_feature_mismatch_raises(self):
        model = GradientBoostingRegressor(n_estimators=20, random_state=0).fit(self.X, self.y)
        with self.assertRaises(ValueError) as ctx:
            model.predict(np.array([[1.0, 2.0]]))
        self.assertIn("features", str(ctx.exception).lower())

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            GradientBoostingRegressor(n_estimators=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            GradientBoostingRegressor(learning_rate=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            GradientBoostingRegressor(max_depth=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            GradientBoostingRegressor(min_samples_split=1).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            GradientBoostingRegressor(min_samples_leaf=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            GradientBoostingRegressor(random_state="seed").fit(self.X, self.y)

    def test_random_state_is_deterministic(self):
        model1 = GradientBoostingRegressor(n_estimators=30, random_state=5).fit(self.X, self.y)
        model2 = GradientBoostingRegressor(n_estimators=30, random_state=5).fit(self.X, self.y)

        np.testing.assert_allclose(model1.predict(self.X), model2.predict(self.X), atol=1e-12)


class TestGradientBoostingClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [-2.0, -1.0],
            [-1.5, -1.0],
            [-1.0, -0.5],
            [1.0, 0.5],
            [1.5, 1.0],
            [2.0, 1.0],
        ])
        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_fit_predict_basic_contract(self):
        model = GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.1,
            max_depth=2,
            random_state=42,
        )
        preds = model.fit(self.X, self.y).predict(self.X)

        self.assertEqual(preds.shape, (self.X.shape[0],))
        self.assertEqual(model.n_features_in_, 2)
        self.assertTrue(hasattr(model, "estimators_"))
        self.assertTrue(hasattr(model, "classes_"))
        self.assertGreaterEqual(model.score(self.X, self.y), 0.90)

    def test_fit_returns_self(self):
        model = GradientBoostingClassifier(n_estimators=20, random_state=42)
        returned = model.fit(self.X, self.y)
        self.assertIs(returned, model)

    def test_predict_before_fit_raises(self):
        model = GradientBoostingClassifier()
        with self.assertRaises(ValueError) as ctx:
            model.predict(self.X)
        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_predict_feature_mismatch_raises(self):
        model = GradientBoostingClassifier(n_estimators=20, random_state=0).fit(self.X, self.y)
        with self.assertRaises(ValueError) as ctx:
            model.predict(np.array([[1.0, 2.0, 3.0]]))
        self.assertIn("features", str(ctx.exception).lower())

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            GradientBoostingClassifier(n_estimators=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            GradientBoostingClassifier(learning_rate=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            GradientBoostingClassifier(max_depth=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            GradientBoostingClassifier(min_samples_split=1).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            GradientBoostingClassifier(min_samples_leaf=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            GradientBoostingClassifier(random_state="seed").fit(self.X, self.y)

    def test_rejects_non_binary_targets(self):
        y_multi = np.array([0, 1, 2, 0, 1, 2])
        with self.assertRaises(ValueError) as ctx:
            GradientBoostingClassifier(n_estimators=20, random_state=0).fit(self.X, y_multi)
        self.assertIn("binary", str(ctx.exception).lower())

    def test_random_state_is_deterministic(self):
        model1 = GradientBoostingClassifier(n_estimators=30, random_state=9).fit(self.X, self.y)
        model2 = GradientBoostingClassifier(n_estimators=30, random_state=9).fit(self.X, self.y)

        np.testing.assert_array_equal(model1.predict(self.X), model2.predict(self.X))


if __name__ == "__main__":
    unittest.main()
