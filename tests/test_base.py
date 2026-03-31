# pyright: reportMissingImports=false
import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure package import works when running tests from repository root.
ROOT_PARENT = Path(__file__).resolve().parents[2]
if str(ROOT_PARENT) not in sys.path:
    sys.path.insert(0, str(ROOT_PARENT))

from mlearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
    BasePipeline,
)


class DummyEstimator(BaseEstimator):
    def __init__(self, alpha=1.0, beta=2.0):
        self.alpha = alpha
        self.beta = beta
        self.coef_ = np.array([1.0, 2.0])


class AddOneTransformer(TransformerMixin):
    def fit(self, x, y=None):
        self.fitted_ = True
        return self

    def transform(self, x):
        return np.asarray(x) + 1


class DummyRegressor(RegressorMixin):
    def predict(self, x):
        x = np.asarray(x)
        return x.sum(axis=1)


class DummyClassifier(ClassifierMixin):
    def predict(self, x):
        x = np.asarray(x)
        return (x[:, 0] > 0).astype(int)


class DummyClusterer(ClusterMixin):
    def fit(self, x, y=None):
        self.was_fit_ = True
        return self

    def predict(self, x):
        x = np.asarray(x)
        return np.zeros(x.shape[0], dtype=int)


class PipelineTransformer:
    def fit_transform(self, x, y=None):
        self.fit_transform_called_ = True
        return np.asarray(x) * 2

    def transform(self, x):
        self.transform_called_ = True
        return np.asarray(x) * 2


class PipelineEstimator:
    def fit(self, x, y=None):
        self.fit_x_ = np.asarray(x)
        self.fit_y_ = np.asarray(y)
        return self

    def predict(self, x):
        x = np.asarray(x)
        return x[:, 0]


class TestBaseEstimator(unittest.TestCase):
    def test_get_params_excludes_learned_attributes(self):
        est = DummyEstimator(alpha=10.0, beta=3.5)
        params = est.get_params()

        self.assertEqual(params["alpha"], 10.0)
        self.assertEqual(params["beta"], 3.5)
        self.assertNotIn("coef_", params)

    def test_set_params_updates_known_fields_and_returns_self(self):
        est = DummyEstimator(alpha=1.0, beta=2.0)
        out = est.set_params(alpha=4.2)

        self.assertIs(out, est)
        self.assertEqual(est.alpha, 4.2)
        self.assertEqual(est.beta, 2.0)

    def test_set_params_raises_on_unknown_parameter(self):
        est = DummyEstimator()
        with self.assertRaises(ValueError) as ctx:
            est.set_params(gamma=123)

        self.assertIn("Invalid parameter 'gamma'", str(ctx.exception))


class TestMixins(unittest.TestCase):
    def test_transformer_mixin_fit_transform(self):
        t = AddOneTransformer()
        x = np.array([[1, 2], [3, 4]])

        x_out = t.fit_transform(x)

        np.testing.assert_array_equal(x_out, np.array([[2, 3], [4, 5]]))
        self.assertTrue(getattr(t, "fitted_", False))

    def test_regressor_mixin_score_uses_r2(self):
        reg = DummyRegressor()
        x = np.array([[1, 0], [1, 1], [2, 1]])
        y = np.array([1, 2, 3])

        score = reg.score(x, y)
        self.assertAlmostEqual(score, 1.0)

    def test_classifier_mixin_score_uses_accuracy(self):
        clf = DummyClassifier()
        x = np.array([[1], [-1], [3], [-2]])
        y = np.array([1, 0, 1, 0])

        score = clf.score(x, y)
        self.assertAlmostEqual(score, 1.0)

    def test_cluster_mixin_fit_predict(self):
        c = DummyClusterer()
        x = np.array([[1, 2], [3, 4], [5, 6]])

        labels = c.fit_predict(x)

        self.assertTrue(getattr(c, "was_fit_", False))
        np.testing.assert_array_equal(labels, np.array([0, 0, 0]))


class TestBasePipeline(unittest.TestCase):
    def test_pipeline_fit_and_predict(self):
        transformer = PipelineTransformer()
        estimator = PipelineEstimator()
        pipe = BasePipeline(steps=[("scale", transformer), ("model", estimator)])

        x = np.array([[1, 2], [3, 4]])
        y = np.array([10, 20])

        pipe.fit(x, y)
        self.assertTrue(getattr(transformer, "fit_transform_called_", False))
        np.testing.assert_array_equal(estimator.fit_x_, np.array([[2, 4], [6, 8]]))
        np.testing.assert_array_equal(estimator.fit_y_, y)

        preds = pipe.predict(x)
        self.assertTrue(getattr(transformer, "transform_called_", False))
        np.testing.assert_array_equal(preds, np.array([2, 6]))


if __name__ == "__main__":
    unittest.main()
