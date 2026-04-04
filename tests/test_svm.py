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

from micromlkit.svm import SVC, SVR


class TestSVC(unittest.TestCase):
    def setUp(self):
        self.X_binary = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
        self.y_binary = np.array([0, 0, 0, 1, 1, 1])

        self.X_xor = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ])
        self.y_xor = np.array([0, 1, 1, 0])

        self.X_multi = np.array([
            [0.0, 0.0],
            [0.2, -0.1],
            [-0.2, 0.1],
            [3.0, 3.0],
            [3.2, 2.9],
            [2.8, 3.1],
            [-3.0, 3.0],
            [-3.2, 3.1],
            [-2.8, 2.9],
        ])
        self.y_multi = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    def test_fit_predict_basic_contract_linear(self):
        model = SVC(C=10.0, kernel="linear").fit(self.X_binary, self.y_binary)
        preds = model.predict(self.X_binary)

        self.assertEqual(preds.shape, (self.X_binary.shape[0],))
        self.assertEqual(model.n_features_in_, 1)
        self.assertTrue(hasattr(model, "support_"))
        self.assertTrue(hasattr(model, "support_vectors_"))
        np.testing.assert_array_equal(model.classes_, np.array([0, 1]))
        self.assertGreaterEqual(model.score(self.X_binary, self.y_binary), 0.99)

    def test_fit_returns_self(self):
        model = SVC(kernel="linear")
        returned = model.fit(self.X_binary, self.y_binary)
        self.assertIs(returned, model)

    def test_predict_before_fit_raises(self):
        model = SVC()
        with self.assertRaises(ValueError) as ctx:
            model.predict(self.X_binary)
        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_predict_feature_mismatch_raises(self):
        model = SVC(kernel="linear").fit(self.X_binary, self.y_binary)
        with self.assertRaises(ValueError) as ctx:
            model.predict(np.array([[1.0, 2.0]]))
        self.assertIn("features", str(ctx.exception).lower())

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            SVC(C=0.0).fit(self.X_binary, self.y_binary)
        with self.assertRaises(ValueError):
            SVC(kernel="poly").fit(self.X_binary, self.y_binary)
        with self.assertRaises(ValueError):
            SVC(gamma=0.0).fit(self.X_binary, self.y_binary)
        with self.assertRaises(ValueError):
            SVC(tol=0.0).fit(self.X_binary, self.y_binary)
        with self.assertRaises(ValueError):
            SVC(max_iter=0).fit(self.X_binary, self.y_binary)
        with self.assertRaises(ValueError):
            SVC(random_state="seed").fit(self.X_binary, self.y_binary)

    def test_fit_raises_for_single_class_target(self):
        y = np.zeros(self.X_binary.shape[0], dtype=int)
        with self.assertRaises(ValueError) as ctx:
            SVC().fit(self.X_binary, y)
        self.assertIn("at least two classes", str(ctx.exception).lower())

    def test_rbf_kernel_solves_xor(self):
        model = SVC(C=100.0, kernel="rbf", gamma=2.0).fit(self.X_xor, self.y_xor)
        preds = model.predict(self.X_xor)
        np.testing.assert_array_equal(preds, self.y_xor)

    def test_multiclass_ovr_predicts_well(self):
        model = SVC(C=20.0, kernel="rbf", gamma=0.8).fit(self.X_multi, self.y_multi)
        preds = model.predict(self.X_multi)

        self.assertEqual(preds.shape, (self.X_multi.shape[0],))
        self.assertGreaterEqual(model.score(self.X_multi, self.y_multi), 0.95)

    def test_decision_function_shapes(self):
        binary = SVC(kernel="linear").fit(self.X_binary, self.y_binary)
        binary_scores = binary.decision_function(self.X_binary)
        self.assertEqual(binary_scores.shape, (self.X_binary.shape[0],))

        multiclass = SVC(kernel="rbf", gamma=1.0).fit(self.X_multi, self.y_multi)
        multi_scores = multiclass.decision_function(self.X_multi)
        self.assertEqual(multi_scores.shape, (self.X_multi.shape[0], len(np.unique(self.y_multi))))

    def test_get_and_set_params(self):
        model = SVC()
        params = model.get_params()

        self.assertIn("C", params)
        self.assertIn("kernel", params)

        out = model.set_params(C=2.5, kernel="linear")
        self.assertIs(out, model)
        self.assertEqual(model.C, 2.5)
        self.assertEqual(model.kernel, "linear")


class TestSVR(unittest.TestCase):
    def setUp(self):
        self.X_linear = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        self.y_linear = 1.5 + 2.0 * self.X_linear[:, 0]

        self.X_nonlinear = np.linspace(0.0, 2.0 * np.pi, 40).reshape(-1, 1)
        self.y_nonlinear = np.sin(self.X_nonlinear[:, 0])

    def test_fit_predict_basic_contract_linear(self):
        model = SVR(C=100.0, epsilon=0.0, kernel="linear", tol=1e-6).fit(self.X_linear, self.y_linear)
        preds = model.predict(self.X_linear)

        self.assertEqual(preds.shape, (self.X_linear.shape[0],))
        self.assertEqual(model.n_features_in_, 1)
        self.assertTrue(hasattr(model, "support_"))
        self.assertTrue(hasattr(model, "support_vectors_"))
        self.assertGreaterEqual(model.score(self.X_linear, self.y_linear), 0.99)

    def test_fit_returns_self(self):
        model = SVR(kernel="linear")
        returned = model.fit(self.X_linear, self.y_linear)
        self.assertIs(returned, model)

    def test_predict_before_fit_raises(self):
        model = SVR()
        with self.assertRaises(ValueError) as ctx:
            model.predict(self.X_linear)
        self.assertIn("not fitted", str(ctx.exception).lower())

    def test_predict_feature_mismatch_raises(self):
        model = SVR(kernel="linear").fit(self.X_linear, self.y_linear)
        with self.assertRaises(ValueError) as ctx:
            model.predict(np.array([[1.0, 2.0]]))
        self.assertIn("features", str(ctx.exception).lower())

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            SVR(C=0.0).fit(self.X_linear, self.y_linear)
        with self.assertRaises(ValueError):
            SVR(epsilon=-0.1).fit(self.X_linear, self.y_linear)
        with self.assertRaises(ValueError):
            SVR(kernel="poly").fit(self.X_linear, self.y_linear)
        with self.assertRaises(ValueError):
            SVR(gamma=0.0).fit(self.X_linear, self.y_linear)
        with self.assertRaises(ValueError):
            SVR(tol=0.0).fit(self.X_linear, self.y_linear)
        with self.assertRaises(ValueError):
            SVR(max_iter=0).fit(self.X_linear, self.y_linear)
        with self.assertRaises(ValueError):
            SVR(random_state="seed").fit(self.X_linear, self.y_linear)

    def test_rbf_kernel_fits_nonlinear_signal(self):
        model = SVR(C=30.0, epsilon=0.01, kernel="rbf", gamma=1.0).fit(self.X_nonlinear, self.y_nonlinear)
        score = model.score(self.X_nonlinear, self.y_nonlinear)

        self.assertGreaterEqual(score, 0.90)

    def test_get_and_set_params(self):
        model = SVR()
        params = model.get_params()

        self.assertIn("C", params)
        self.assertIn("epsilon", params)

        out = model.set_params(C=2.0, epsilon=0.2)
        self.assertIs(out, model)
        self.assertEqual(model.C, 2.0)
        self.assertEqual(model.epsilon, 0.2)


if __name__ == "__main__":
    unittest.main()
