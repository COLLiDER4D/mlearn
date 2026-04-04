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

from micromlkit.tree import DecisionTreeClassifier, DecisionTreeRegressor


class TestDecisionTreeClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.1, 0.2],
            [0.9, 0.8],
        ])
        self.y = np.array([0, 0, 1, 1, 0, 1])

    def test_fit_predict_basic_contract(self):
        model = DecisionTreeClassifier(random_state=42)
        preds = model.fit(self.X, self.y).predict(self.X)

        self.assertEqual(preds.shape, (self.X.shape[0],))
        self.assertEqual(model.n_features_in_, 2)
        self.assertTrue(hasattr(model, "tree_"))
        self.assertTrue(hasattr(model, "classes_"))
        np.testing.assert_array_equal(model.classes_, np.array([0, 1]))

    def test_fit_returns_self(self):
        model = DecisionTreeClassifier(random_state=42)
        returned = model.fit(self.X, self.y)
        self.assertIs(returned, model)

    def test_predict_before_fit_raises(self):
        model = DecisionTreeClassifier()
        with self.assertRaises(ValueError) as ctx:
            model.predict(self.X)
        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_predict_feature_mismatch_raises(self):
        model = DecisionTreeClassifier().fit(self.X, self.y)
        with self.assertRaises(ValueError) as ctx:
            model.predict(np.array([[1.0, 2.0, 3.0]]))
        self.assertIn("features", str(ctx.exception).lower())

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(max_depth=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(criterion="invalid").fit(self.X, self.y)
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(min_samples_split=1).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(min_samples_leaf=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(random_state="seed").fit(self.X, self.y)

    def test_max_depth_limits_tree(self):
        model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(self.X, self.y)
        self.assertLessEqual(model.depth_, 1)

    def test_random_state_is_deterministic(self):
        model1 = DecisionTreeClassifier(random_state=7).fit(self.X, self.y)
        model2 = DecisionTreeClassifier(random_state=7).fit(self.X, self.y)

        np.testing.assert_array_equal(model1.predict(self.X), model2.predict(self.X))
        self.assertEqual(model1.depth_, model2.depth_)
        self.assertEqual(model1.n_nodes_, model2.n_nodes_)

    def test_score_method(self):
        model = DecisionTreeClassifier(random_state=0).fit(self.X, self.y)
        score = model.score(self.X, self.y)
        self.assertAlmostEqual(score, 1.0, places=10)


class TestDecisionTreeRegressor(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 2.0],
            [5.0, 2.0],
        ])
        self.y = 2.0 * self.X[:, 0] + 0.5 * self.X[:, 1]

    def test_fit_predict_basic_contract(self):
        model = DecisionTreeRegressor(random_state=42)
        preds = model.fit(self.X, self.y).predict(self.X)

        self.assertEqual(preds.shape, (self.X.shape[0],))
        self.assertEqual(model.n_features_in_, 2)
        self.assertTrue(hasattr(model, "tree_"))
        self.assertTrue(hasattr(model, "n_nodes_"))
        self.assertTrue(hasattr(model, "depth_"))
        self.assertGreaterEqual(model.score(self.X, self.y), 0.99)

    def test_fit_returns_self(self):
        model = DecisionTreeRegressor(random_state=42)
        returned = model.fit(self.X, self.y)
        self.assertIs(returned, model)

    def test_predict_before_fit_raises(self):
        model = DecisionTreeRegressor()
        with self.assertRaises(ValueError) as ctx:
            model.predict(self.X)
        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_predict_feature_mismatch_raises(self):
        model = DecisionTreeRegressor().fit(self.X, self.y)
        with self.assertRaises(ValueError) as ctx:
            model.predict(np.array([[1.0, 2.0, 3.0]]))
        self.assertIn("features", str(ctx.exception).lower())

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            DecisionTreeRegressor(max_depth=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            DecisionTreeRegressor(criterion="invalid").fit(self.X, self.y)
        with self.assertRaises(ValueError):
            DecisionTreeRegressor(min_samples_split=1).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            DecisionTreeRegressor(min_samples_leaf=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            DecisionTreeRegressor(random_state="seed").fit(self.X, self.y)

    def test_max_depth_limits_tree(self):
        model = DecisionTreeRegressor(max_depth=1, random_state=0).fit(self.X, self.y)
        self.assertLessEqual(model.depth_, 1)

    def test_random_state_is_deterministic(self):
        model1 = DecisionTreeRegressor(random_state=7).fit(self.X, self.y)
        model2 = DecisionTreeRegressor(random_state=7).fit(self.X, self.y)

        np.testing.assert_allclose(model1.predict(self.X), model2.predict(self.X), atol=1e-12)
        self.assertEqual(model1.depth_, model2.depth_)
        self.assertEqual(model1.n_nodes_, model2.n_nodes_)

    def test_score_method(self):
        model = DecisionTreeRegressor(random_state=0).fit(self.X, self.y)
        score = model.score(self.X, self.y)
        self.assertAlmostEqual(score, 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
