# pyright: reportMissingImports=false
import math
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

from micromlkit.metrics.classification import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from micromlkit.metrics.regression import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


class TestClassificationMetrics(unittest.TestCase):
    def test_accuracy_score(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 0.75)

    def test_precision_recall_f1(self):
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 1, 1, 0, 0]

        self.assertAlmostEqual(precision_score(y_true, y_pred), 2 / 3)
        self.assertAlmostEqual(recall_score(y_true, y_pred), 2 / 3)
        self.assertAlmostEqual(f1_score(y_true, y_pred), 2 / 3)

    def test_precision_when_no_predicted_positives(self):
        y_true = [1, 0, 1]
        y_pred = [0, 0, 0]
        self.assertEqual(precision_score(y_true, y_pred), 0.0)

    def test_recall_when_no_actual_positives(self):
        y_true = [0, 0, 0]
        y_pred = [1, 0, 1]
        self.assertEqual(recall_score(y_true, y_pred), 0.0)

    def test_shape_mismatch_raises_value_error(self):
        with self.assertRaises(ValueError):
            accuracy_score([1, 0], [1])

    def test_roc_auc_score_known_example(self):
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.4, 0.35, 0.8]
        self.assertAlmostEqual(roc_auc_score(y_true, y_scores), 0.75)

    def test_log_loss_matches_manual_formula(self):
        y_true = np.array([1, 0, 1, 0])
        y_proba = np.array([0.9, 0.2, 0.8, 0.1])

        expected = -np.mean(y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba))
        self.assertAlmostEqual(log_loss(y_true, y_proba), expected)


class TestRegressionMetrics(unittest.TestCase):
    def test_r2_score_perfect_predictions(self):
        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]
        self.assertAlmostEqual(r2_score(y_true, y_pred), 1.0)

    def test_r2_score_constant_target(self):
        y_true = [5, 5, 5]
        y_pred = [4, 5, 6]
        self.assertEqual(r2_score(y_true, y_pred), 0.0)

    def test_mse_rmse_mae(self):
        y_true = [1, 2, 3]
        y_pred = [2, 2, 4]

        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 2 / 3)
        self.assertAlmostEqual(root_mean_squared_error(y_true, y_pred), math.sqrt(2 / 3))
        self.assertAlmostEqual(mean_absolute_error(y_true, y_pred), 2 / 3)

    def test_mape(self):
        y_true = [100, 200]
        y_pred = [90, 220]
        self.assertAlmostEqual(mean_absolute_percentage_error(y_true, y_pred), 10.0)


if __name__ == "__main__":
    unittest.main()
