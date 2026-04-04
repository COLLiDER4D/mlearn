import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Ensure imports work with this repository's current package layout.
ROOT_PARENT = Path(__file__).resolve().parents[2]  # .../capstone
PKG_ROOT = Path(__file__).resolve().parents[1]     # .../capstone/microMlKit
for p in (ROOT_PARENT, PKG_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_selection.split import KFold, train_test_split
from model_selection.cross_validation import cross_val_score
from model_selection.search import GridSearchCV, ParameterGrid


class DummyClassifier:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        self.was_fit_ = True
        return self

    def predict(self, X):
        X = np.asarray(X)
        return (X[:, 0] > self.threshold).astype(int)


class TwoFoldCV:
    """Simple CV object exposing .split(X), matching cross_val_score's expectation."""
    def split(self, X):
        indices = np.arange(len(X))
        yield indices[2:], indices[:2]   # train [2,3], test [0,1]
        yield indices[:2], indices[2:]   # train [0,1], test [2,3]


class TestSplitUtilities(unittest.TestCase):
    def test_train_test_split_shapes_and_reproducibility(self):
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)

        out1 = train_test_split(X, y, test_size=0.3, random_state=42)
        out2 = train_test_split(X, y, test_size=0.3, random_state=42)

        X_train, X_test, y_train, y_test = out1

        self.assertEqual(X_test.shape[0], 3)
        self.assertEqual(X_train.shape[0], 7)
        np.testing.assert_array_equal(out1[0], out2[0])  # X_train
        np.testing.assert_array_equal(out1[1], out2[1])  # X_test
        np.testing.assert_array_equal(out1[2], out2[2])  # y_train
        np.testing.assert_array_equal(out1[3], out2[3])  # y_test

    def test_kfold_invalid_splits_raises(self):
        with self.assertRaises(ValueError):
            KFold(n_splits=1)

    def test_kfold_split_function_yields_disjoint_folds(self):
        X = np.arange(10)
        split_fn = KFold(n_splits=3, shuffle=False)
        folds = list(split_fn(X))

        self.assertEqual(len(folds), 3)

        all_test_indices = []
        for train_idx, test_idx in folds:
            self.assertEqual(len(np.intersect1d(train_idx, test_idx)), 0)
            all_test_indices.extend(test_idx.tolist())

        # Every sample appears exactly once in test sets across folds
        self.assertEqual(sorted(all_test_indices), list(range(10)))


class TestCrossValidation(unittest.TestCase):
    def test_cross_val_score_with_custom_cv_and_scoring(self):
        X = np.array([[0.0], [1.0], [0.0], [1.0]])
        y = np.array([0, 1, 0, 1])

        est = DummyClassifier(threshold=0.5)
        cv = TwoFoldCV()

        def accuracy(y_true, y_pred):
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            return np.mean(y_true == y_pred)

        scores = cross_val_score(est, X, y, cv=cv, scoring=accuracy)

        self.assertEqual(scores.shape, (2,))
        np.testing.assert_allclose(scores, np.array([1.0, 1.0]))


class TestSearch(unittest.TestCase):
    def test_parameter_grid_cartesian_product(self):
        grid = ParameterGrid({"a": [1, 2], "b": ["x", "y"]})
        combos = list(grid)

        expected = [
            {"a": 1, "b": "x"},
            {"a": 1, "b": "y"},
            {"a": 2, "b": "x"},
            {"a": 2, "b": "y"},
        ]
        self.assertEqual(combos, expected)

    def test_grid_search_cv_selects_best_params(self):
        X = np.array([[0.0], [1.0], [0.0], [1.0]])
        y = np.array([0, 1, 0, 1])
        est = DummyClassifier()

        # Mean scores by threshold:
        # 0.2 -> 0.60, 0.5 -> 0.95, 0.8 -> 0.70
        score_map = {
            0.2: np.array([0.5, 0.7]),
            0.5: np.array([0.9, 1.0]),
            0.8: np.array([0.6, 0.8]),
        }

        def fake_cross_val_score(estimator, X, y, cv):
            return score_map[estimator.threshold]

        with patch("model_selection.search.cross_val_score", side_effect=fake_cross_val_score):
            gs = GridSearchCV(estimator=est, param_grid={"threshold": [0.2, 0.5, 0.8]}, cv=2)
            gs.fit(X, y)

        self.assertEqual(gs.best_params_, {"threshold": 0.5})
        self.assertAlmostEqual(gs.best_score_, 0.95)
        self.assertEqual(est.threshold, 0.5)


if __name__ == "__main__":
    unittest.main()