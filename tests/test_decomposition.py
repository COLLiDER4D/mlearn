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

from micromlkit.decomposition import PCA


class TestPCA(unittest.TestCase):
    def test_fit_transform_reduces_dimensions(self):
        X = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ])

        pca = PCA(n_components=1)
        X_reduced = pca.fit_transform(X)

        self.assertEqual(X_reduced.shape, (4, 1))
        self.assertEqual(pca.components_.shape, (1, 2))
        self.assertEqual(pca.explained_variance_.shape, (1,))
        self.assertEqual(pca.explained_variance_ratio_.shape, (1,))
        self.assertEqual(pca.n_features_in_, 2)
        np.testing.assert_allclose(np.mean(X_reduced, axis=0), np.array([0.0]), atol=1e-10)

    def test_transform_before_fit_raises(self):
        pca = PCA(n_components=1)

        with self.assertRaises(ValueError) as ctx:
            pca.transform(np.array([[1.0, 2.0]]))

        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_transform_feature_mismatch_raises(self):
        pca = PCA(n_components=1).fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

        with self.assertRaises(ValueError) as ctx:
            pca.transform(np.array([[1.0, 2.0, 3.0]]))

        self.assertIn("features", str(ctx.exception).lower())

    def test_default_n_components_uses_full_rank(self):
        X = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ])

        pca = PCA()
        X_reduced = pca.fit_transform(X)

        self.assertEqual(X_reduced.shape, (4, 3))
        self.assertEqual(pca.components_.shape, (3, 3))
        self.assertEqual(pca.explained_variance_.shape, (3,))
        self.assertEqual(pca.explained_variance_ratio_.shape, (3,))

    def test_invalid_n_components_raises(self):
        X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])

        invalid_values = [0, -1, 3, 1.5, "2"]
        for value in invalid_values:
            with self.subTest(n_components=value):
                with self.assertRaises(ValueError):
                    PCA(n_components=value).fit(X)

    def test_transform_matches_projection_formula(self):
        X = np.array([
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [3.0, 1.0],
        ])

        pca = PCA(n_components=2).fit(X)
        transformed = pca.transform(X)
        expected = (X - pca.mean_) @ pca.components_.T

        np.testing.assert_allclose(transformed, expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
