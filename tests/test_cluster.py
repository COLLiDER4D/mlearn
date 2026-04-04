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

from micromlkit.cluster import AgglomerativeClustering, DBSCAN, KMeans


class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
        ])

    def test_fit_predict_basic_contract(self):
        model = KMeans(n_clusters=2, random_state=42)
        labels = model.fit_predict(self.X)

        self.assertEqual(labels.shape, (self.X.shape[0],))
        self.assertEqual(model.cluster_centers_.shape, (2, 2))
        self.assertEqual(model.n_features_in_, 2)
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < 2))

    def test_fit_returns_self(self):
        model = KMeans(n_clusters=2, random_state=42)
        returned = model.fit(self.X)
        self.assertIs(returned, model)

    def test_predict_before_fit_raises(self):
        model = KMeans(n_clusters=2)
        with self.assertRaises(ValueError) as ctx:
            model.predict(self.X)
        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_predict_feature_mismatch_raises(self):
        model = KMeans(n_clusters=2, random_state=42).fit(self.X)
        with self.assertRaises(ValueError) as ctx:
            model.predict(np.array([[1.0, 2.0, 3.0]]))
        self.assertIn("features", str(ctx.exception).lower())

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            KMeans(n_clusters=0).fit(self.X)
        with self.assertRaises(ValueError):
            KMeans(max_iter=0).fit(self.X)
        with self.assertRaises(ValueError):
            KMeans(tol=0).fit(self.X)
        with self.assertRaises(ValueError):
            KMeans(metric="invalid").fit(self.X)
        with self.assertRaises(ValueError):
            KMeans(metric="minkowski", p=0.5).fit(self.X)

    def test_n_clusters_greater_than_samples_raises(self):
        X_small = np.array([[1.0, 1.0], [2.0, 2.0]])
        with self.assertRaises(ValueError):
            KMeans(n_clusters=3).fit(X_small)

    def test_random_state_is_deterministic(self):
        model1 = KMeans(n_clusters=2, random_state=7).fit(self.X)
        model2 = KMeans(n_clusters=2, random_state=7).fit(self.X)
        np.testing.assert_array_equal(model1.labels_, model2.labels_)
        np.testing.assert_allclose(model1.cluster_centers_, model2.cluster_centers_, atol=1e-12)

    def test_all_metrics_supported(self):
        metric_configs = [
            ("euclidean", {}),
            ("manhattan", {}),
            ("cosine", {}),
            ("chebyshev", {}),
            ("minkowski", {"p": 3}),
        ]

        X_metric = np.array([
            [1.0, 2.0],
            [1.1, 2.1],
            [4.0, 4.0],
            [4.1, 4.1],
        ])

        for metric, kwargs in metric_configs:
            with self.subTest(metric=metric):
                model = KMeans(n_clusters=2, random_state=0, metric=metric, **kwargs).fit(X_metric)
                self.assertEqual(model.labels_.shape[0], X_metric.shape[0])


class TestDBSCAN(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [10.0, 10.0],
        ])

    def test_fit_predict_and_noise_label(self):
        model = DBSCAN(eps=0.3, min_samples=2)
        labels = model.fit_predict(self.X)

        self.assertEqual(labels.shape, (self.X.shape[0],))
        self.assertIn(-1, labels.tolist())
        self.assertEqual(model.n_features_in_, 2)
        self.assertTrue(hasattr(model, "core_sample_indices_"))

    def test_predict_unseen_samples(self):
        model = DBSCAN(eps=0.3, min_samples=2).fit(self.X)

        near_first_cluster = model.predict(np.array([[0.05, 0.02]]))[0]
        far_point = model.predict(np.array([[20.0, 20.0]]))[0]

        self.assertNotEqual(near_first_cluster, -1)
        self.assertEqual(far_point, -1)

    def test_predict_before_fit_raises(self):
        model = DBSCAN()
        with self.assertRaises(ValueError) as ctx:
            model.predict(self.X)
        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_predict_feature_mismatch_raises(self):
        model = DBSCAN().fit(self.X)
        with self.assertRaises(ValueError) as ctx:
            model.predict(np.array([[1.0, 2.0, 3.0]]))
        self.assertIn("features", str(ctx.exception).lower())

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            DBSCAN(eps=0).fit(self.X)
        with self.assertRaises(ValueError):
            DBSCAN(min_samples=0).fit(self.X)
        with self.assertRaises(ValueError):
            DBSCAN(metric="invalid").fit(self.X)
        with self.assertRaises(ValueError):
            DBSCAN(metric="minkowski", p=0.9).fit(self.X)

    def test_all_metrics_supported(self):
        metric_configs = [
            ("euclidean", {}),
            ("manhattan", {}),
            ("cosine", {}),
            ("chebyshev", {}),
            ("minkowski", {"p": 3}),
        ]

        X_metric = np.array([
            [1.0, 1.0],
            [1.1, 1.0],
            [4.0, 4.0],
            [4.1, 4.0],
        ])

        for metric, kwargs in metric_configs:
            with self.subTest(metric=metric):
                model = DBSCAN(eps=0.25, min_samples=1, metric=metric, **kwargs).fit(X_metric)
                self.assertEqual(model.labels_.shape[0], X_metric.shape[0])


class TestAgglomerativeClustering(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
        ])

    def test_fit_predict_basic_contract(self):
        model = AgglomerativeClustering(n_clusters=2, linkage="average")
        labels = model.fit_predict(self.X)

        self.assertEqual(labels.shape, (self.X.shape[0],))
        self.assertEqual(model.n_features_in_, 2)
        self.assertEqual(np.unique(labels).size, 2)

    def test_predict_unseen_samples(self):
        model = AgglomerativeClustering(n_clusters=2, linkage="complete").fit(self.X)

        baseline_label = model.predict(np.array([[0.0, 0.0]]))[0]
        near_label = model.predict(np.array([[0.05, 0.05]]))[0]

        self.assertEqual(near_label, baseline_label)

    def test_predict_before_fit_raises(self):
        model = AgglomerativeClustering(n_clusters=2)
        with self.assertRaises(ValueError) as ctx:
            model.predict(self.X)
        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_predict_feature_mismatch_raises(self):
        model = AgglomerativeClustering(n_clusters=2).fit(self.X)
        with self.assertRaises(ValueError) as ctx:
            model.predict(np.array([[1.0, 2.0, 3.0]]))
        self.assertIn("features", str(ctx.exception).lower())

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            AgglomerativeClustering(n_clusters=0).fit(self.X)
        with self.assertRaises(ValueError):
            AgglomerativeClustering(linkage="ward").fit(self.X)
        with self.assertRaises(ValueError):
            AgglomerativeClustering(metric="invalid").fit(self.X)
        with self.assertRaises(ValueError):
            AgglomerativeClustering(metric="minkowski", p=0.5).fit(self.X)

    def test_n_clusters_greater_than_samples_raises(self):
        X_small = np.array([[1.0, 1.0], [2.0, 2.0]])
        with self.assertRaises(ValueError):
            AgglomerativeClustering(n_clusters=3).fit(X_small)

    def test_all_metrics_supported(self):
        metric_configs = [
            ("euclidean", {}),
            ("manhattan", {}),
            ("cosine", {}),
            ("chebyshev", {}),
            ("minkowski", {"p": 3}),
        ]

        X_metric = np.array([
            [1.0, 1.0],
            [1.1, 1.0],
            [4.0, 4.0],
            [4.1, 4.0],
        ])

        for metric, kwargs in metric_configs:
            with self.subTest(metric=metric):
                model = AgglomerativeClustering(
                    n_clusters=2,
                    linkage="single",
                    metric=metric,
                    **kwargs,
                ).fit(X_metric)
                self.assertEqual(model.labels_.shape[0], X_metric.shape[0])


if __name__ == "__main__":
    unittest.main()
