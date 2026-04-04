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

from micromlkit.base import BaseEstimator
from micromlkit.pipeline import Pipeline


class DoubleTransformer:
    def fit_transform(self, X, y=None):
        self.fit_transform_called_ = True
        return np.asarray(X) * 2

    def transform(self, X):
        self.transform_called_ = True
        return np.asarray(X) * 2


class AddOneTransformer:
    def fit_transform(self, X, y=None):
        return np.asarray(X) + 1

    def transform(self, X):
        return np.asarray(X) + 1


class FirstColumnEstimator(BaseEstimator):
    def __init__(self, offset=0.0):
        self.offset = offset

    def fit(self, X, y=None):
        self.fit_X_ = np.asarray(X)
        self.fit_y_ = None if y is None else np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0] + self.offset


class TransformingFinalStep(BaseEstimator):
    def __init__(self, scale=3.0):
        self.scale = scale

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0]

    def transform(self, X):
        return np.asarray(X) * self.scale


class InvalidTransformer:
    def fit(self, X, y=None):
        return self


class TestPipeline(unittest.TestCase):
    def test_fit_and_predict_chain(self):
        transformer = DoubleTransformer()
        estimator = FirstColumnEstimator(offset=1.5)
        pipe = Pipeline(steps=[("scale", transformer), ("model", estimator)])

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([10.0, 20.0])

        out = pipe.fit(X, y)
        self.assertIs(out, pipe)
        self.assertTrue(getattr(transformer, "fit_transform_called_", False))
        np.testing.assert_array_equal(estimator.fit_X_, np.array([[2.0, 4.0], [6.0, 8.0]]))
        np.testing.assert_array_equal(estimator.fit_y_, y)

        preds = pipe.predict(X)
        self.assertTrue(getattr(transformer, "transform_called_", False))
        np.testing.assert_array_equal(preds, np.array([3.5, 7.5]))

    def test_constructor_validation_errors(self):
        with self.assertRaises(ValueError):
            Pipeline(steps=[])

        with self.assertRaises(ValueError):
            Pipeline(steps=[("a", DoubleTransformer()), ("a", FirstColumnEstimator())])

        with self.assertRaises(ValueError):
            Pipeline(steps=[("bad", InvalidTransformer()), ("model", FirstColumnEstimator())])

    def test_reserved_step_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            Pipeline(steps=[("steps", DoubleTransformer()), ("model", FirstColumnEstimator())])
        self.assertIn("reserved", str(ctx.exception))

    def test_step_name_with_double_underscore_raises(self):
        with self.assertRaises(ValueError) as ctx:
            Pipeline(steps=[("my__step", DoubleTransformer()), ("model", FirstColumnEstimator())])
        self.assertIn("__", str(ctx.exception))

    def test_named_steps_and_nested_set_params(self):
        model = FirstColumnEstimator(offset=0.0)
        pipe = Pipeline(steps=[("scale", DoubleTransformer()), ("model", model)])

        self.assertIs(pipe.named_steps["model"], model)

        out = pipe.set_params(model__offset=2.0)
        self.assertIs(out, pipe)
        self.assertEqual(model.offset, 2.0)

        params = pipe.get_params(deep=True)
        self.assertIn("model__offset", params)
        self.assertEqual(params["model__offset"], 2.0)

    def test_transform_requires_final_transform_method(self):
        pipe = Pipeline(steps=[("scale", AddOneTransformer()), ("model", FirstColumnEstimator())])
        pipe.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

        with self.assertRaises(ValueError) as ctx:
            pipe.transform(np.array([[2.0, 3.0]]))

        self.assertIn("does not support 'transform'", str(ctx.exception))

    def test_fit_transform_uses_final_transform_when_available(self):
        pipe = Pipeline(
            steps=[("add", AddOneTransformer()), ("final", TransformingFinalStep(scale=2.0))]
        )

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_out = pipe.fit_transform(X)

        expected = np.array([[4.0, 6.0], [8.0, 10.0]])
        np.testing.assert_array_equal(X_out, expected)


if __name__ == "__main__":
    unittest.main()
