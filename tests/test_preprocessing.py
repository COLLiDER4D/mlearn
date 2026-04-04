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

from micromlkit.preprocessing import LabelEncoder, SimpleImputer, StandardScaler


class TestStandardScaler(unittest.TestCase):
    def test_fit_transform_centers_and_scales(self):
        X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        np.testing.assert_allclose(np.mean(X_scaled, axis=0), np.array([0.0, 0.0]), atol=1e-10)
        np.testing.assert_allclose(np.std(X_scaled, axis=0), np.array([1.0, 1.0]), atol=1e-10)
        self.assertEqual(scaler.n_features_in_, 2)

    def test_transform_before_fit_raises(self):
        scaler = StandardScaler()

        with self.assertRaises(ValueError) as ctx:
            scaler.transform(np.array([[1.0, 2.0]]))

        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_transform_feature_mismatch_raises(self):
        scaler = StandardScaler().fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

        with self.assertRaises(ValueError) as ctx:
            scaler.transform(np.array([[1.0, 2.0, 3.0]]))

        self.assertIn("features", str(ctx.exception).lower())


class TestSimpleImputer(unittest.TestCase):
    def test_mean_strategy_imputes_missing_values(self):
        X = np.array([
            [1.0, np.nan],
            [3.0, 4.0],
            [5.0, np.nan],
        ])

        imputer = SimpleImputer(strategy="mean")
        X_out = imputer.fit_transform(X)

        expected = np.array([
            [1.0, 4.0],
            [3.0, 4.0],
            [5.0, 4.0],
        ])
        np.testing.assert_allclose(X_out, expected, atol=1e-10)

    def test_constant_strategy_imputes_none_values(self):
        X = np.array([
            ["a", None],
            ["b", "x"],
            ["c", None],
        ], dtype=object)

        imputer = SimpleImputer(strategy="constant", fill_value="missing")
        X_out = imputer.fit_transform(X)

        expected = np.array([
            ["a", "missing"],
            ["b", "x"],
            ["c", "missing"],
        ], dtype=object)
        np.testing.assert_array_equal(X_out, expected)

    def test_transform_before_fit_raises(self):
        imputer = SimpleImputer(strategy="mean")

        with self.assertRaises(ValueError) as ctx:
            imputer.transform(np.array([[1.0], [np.nan]]))

        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_mean_strategy_all_missing_column_raises(self):
        X = np.array([
            [np.nan, 1.0],
            [np.nan, 2.0],
        ])

        with self.assertRaises(ValueError) as ctx:
            SimpleImputer(strategy="mean").fit(X)

        self.assertIn("only missing values", str(ctx.exception).lower())


class TestLabelEncoder(unittest.TestCase):
    def test_fit_transform_and_inverse_transform(self):
        y = np.array(["cat", "dog", "cat", "bird"], dtype=object)

        encoder = LabelEncoder()
        encoded = encoder.fit_transform(y)
        decoded = encoder.inverse_transform(encoded)

        self.assertEqual(encoded.shape, y.shape)
        np.testing.assert_array_equal(decoded, y)

    def test_transform_with_unseen_label_raises(self):
        encoder = LabelEncoder().fit(np.array(["red", "blue"], dtype=object))

        with self.assertRaises(ValueError) as ctx:
            encoder.transform(np.array(["red", "green"], dtype=object))

        self.assertIn("unseen", str(ctx.exception).lower())

    def test_transform_before_fit_raises(self):
        encoder = LabelEncoder()

        with self.assertRaises(ValueError) as ctx:
            encoder.transform(np.array(["a", "b"], dtype=object))

        self.assertIn("not fitted", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
