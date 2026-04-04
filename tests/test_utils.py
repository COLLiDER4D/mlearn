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

from micromlkit.utils.math import compute_kernel, linear_kernel, rbf_kernel, sigmoid, soft_threshold
from micromlkit.utils.validation import (
    check_is_fitted,
    ensure_1d_array,
    ensure_2d_float_array,
    ensure_same_n_samples,
    validate_feature_count,
    validate_non_negative_number,
    validate_positive_integer,
    validate_positive_number,
    validate_random_state,
)


class TestSigmoid(unittest.TestCase):
    def test_zero_input(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)

    def test_large_positive(self):
        self.assertAlmostEqual(sigmoid(100.0), 1.0, places=6)

    def test_large_negative(self):
        self.assertAlmostEqual(sigmoid(-100.0), 0.0, places=6)

    def test_array_input(self):
        result = sigmoid(np.array([-1.0, 0.0, 1.0]))
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(result > 0.0))
        self.assertTrue(np.all(result < 1.0))

    def test_numerically_stable_clipping(self):
        # Values well beyond +/-500 should not produce nan or inf.
        result = sigmoid(np.array([-1000.0, 1000.0]))
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))


class TestSoftThreshold(unittest.TestCase):
    def test_above_threshold(self):
        self.assertAlmostEqual(soft_threshold(5.0, 2.0), 3.0)

    def test_below_negative_threshold(self):
        self.assertAlmostEqual(soft_threshold(-5.0, 2.0), -3.0)

    def test_within_threshold_returns_zero(self):
        self.assertEqual(soft_threshold(1.0, 2.0), 0.0)
        self.assertEqual(soft_threshold(-1.0, 2.0), 0.0)
        self.assertEqual(soft_threshold(0.0, 0.5), 0.0)

    def test_at_boundary(self):
        self.assertEqual(soft_threshold(2.0, 2.0), 0.0)
        self.assertEqual(soft_threshold(-2.0, 2.0), 0.0)


class TestLinearKernel(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.Y = np.array([[1.0, 0.0], [0.0, 1.0]])

    def test_identity_matrix_input(self):
        K = linear_kernel(self.X, self.Y)
        np.testing.assert_array_almost_equal(K, np.eye(2))

    def test_output_shape(self):
        X = np.ones((3, 4))
        Y = np.ones((5, 4))
        K = linear_kernel(X, Y)
        self.assertEqual(K.shape, (3, 5))


class TestRbfKernel(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1.0, 0.0], [0.0, 1.0]])

    def test_diagonal_is_one(self):
        K = rbf_kernel(self.X, self.X, gamma=1.0)
        np.testing.assert_array_almost_equal(np.diag(K), np.ones(2))

    def test_output_shape(self):
        X = np.ones((3, 4))
        Y = np.ones((5, 4))
        K = rbf_kernel(X, Y, gamma=0.5)
        self.assertEqual(K.shape, (3, 5))

    def test_invalid_gamma_none_raises(self):
        with self.assertRaises((ValueError, TypeError)):
            rbf_kernel(self.X, self.X, gamma=None)

    def test_invalid_gamma_zero_raises(self):
        with self.assertRaises(ValueError):
            rbf_kernel(self.X, self.X, gamma=0.0)

    def test_invalid_gamma_negative_raises(self):
        with self.assertRaises(ValueError):
            rbf_kernel(self.X, self.X, gamma=-1.0)

    def test_invalid_gamma_bool_raises(self):
        with self.assertRaises(ValueError):
            rbf_kernel(self.X, self.X, gamma=True)


class TestComputeKernel(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_linear_kernel_dispatch(self):
        K_direct = linear_kernel(self.X, self.X)
        K_compute = compute_kernel(self.X, self.X, "linear", gamma=None)
        np.testing.assert_array_almost_equal(K_compute, K_direct)

    def test_rbf_kernel_dispatch(self):
        K_direct = rbf_kernel(self.X, self.X, gamma=0.5)
        K_compute = compute_kernel(self.X, self.X, "rbf", gamma=0.5)
        np.testing.assert_array_almost_equal(K_compute, K_direct)

    def test_invalid_kernel_raises(self):
        with self.assertRaises(ValueError):
            compute_kernel(self.X, self.X, "polynomial", gamma=1.0)

    def test_invalid_kernel_empty_string_raises(self):
        with self.assertRaises(ValueError):
            compute_kernel(self.X, self.X, "", gamma=1.0)


class TestEnsure2dFloatArray(unittest.TestCase):
    def test_valid_2d_array_returned(self):
        X = [[1, 2], [3, 4]]
        result = ensure_2d_float_array(X)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.dtype, float)

    def test_1d_array_raises(self):
        with self.assertRaises(ValueError):
            ensure_2d_float_array([1, 2, 3])

    def test_3d_array_raises(self):
        with self.assertRaises(ValueError):
            ensure_2d_float_array(np.ones((2, 2, 2)))

    def test_empty_array_allowed_by_default(self):
        X = np.empty((0, 3))
        result = ensure_2d_float_array(X)
        self.assertEqual(result.shape, (0, 3))

    def test_empty_array_raises_when_require_non_empty(self):
        with self.assertRaises(ValueError):
            ensure_2d_float_array(np.empty((0, 3)), require_non_empty=True)

    def test_empty_features_raises_when_require_non_empty(self):
        with self.assertRaises(ValueError):
            ensure_2d_float_array(np.empty((3, 0)), require_non_empty=True)

    def test_custom_name_in_error_message(self):
        with self.assertRaises(ValueError) as ctx:
            ensure_2d_float_array([1, 2, 3], name="X_train")
        self.assertIn("X_train", str(ctx.exception))


class TestValidatePositiveInteger(unittest.TestCase):
    def test_valid_integer(self):
        self.assertEqual(validate_positive_integer(5, "n"), 5)

    def test_numpy_integer(self):
        self.assertEqual(validate_positive_integer(np.int64(3), "n"), 3)

    def test_zero_raises(self):
        with self.assertRaises(ValueError):
            validate_positive_integer(0, "n")

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            validate_positive_integer(-1, "n")

    def test_float_raises(self):
        with self.assertRaises(ValueError):
            validate_positive_integer(1.5, "n")

    def test_bool_raises(self):
        with self.assertRaises(ValueError):
            validate_positive_integer(True, "n")

    def test_none_raises(self):
        with self.assertRaises(ValueError):
            validate_positive_integer(None, "n")


class TestValidatePositiveNumber(unittest.TestCase):
    def test_valid_float(self):
        self.assertAlmostEqual(validate_positive_number(2.5, "x"), 2.5)

    def test_valid_integer(self):
        self.assertAlmostEqual(validate_positive_number(3, "x"), 3.0)

    def test_zero_raises(self):
        with self.assertRaises(ValueError):
            validate_positive_number(0.0, "x")

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            validate_positive_number(-0.1, "x")

    def test_bool_raises(self):
        with self.assertRaises(ValueError):
            validate_positive_number(True, "x")

    def test_string_raises(self):
        with self.assertRaises(ValueError):
            validate_positive_number("1.0", "x")


class TestValidateNonNegativeNumber(unittest.TestCase):
    def test_zero_is_valid(self):
        self.assertAlmostEqual(validate_non_negative_number(0.0, "x"), 0.0)

    def test_positive_is_valid(self):
        self.assertAlmostEqual(validate_non_negative_number(5.0, "x"), 5.0)

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            validate_non_negative_number(-1.0, "x")

    def test_bool_raises(self):
        with self.assertRaises(ValueError):
            validate_non_negative_number(False, "x")


class TestValidateRandomState(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(validate_random_state(None))

    def test_valid_integer(self):
        self.assertEqual(validate_random_state(42), 42)

    def test_bool_raises(self):
        with self.assertRaises(ValueError):
            validate_random_state(True)

    def test_float_raises(self):
        with self.assertRaises(ValueError):
            validate_random_state(1.5)


class TestValidateFeatureCount(unittest.TestCase):
    def test_matching_features_passes(self):
        X = np.ones((5, 3))
        validate_feature_count(X, 3, "TestEstimator")  # should not raise

    def test_mismatched_features_raises(self):
        X = np.ones((5, 4))
        with self.assertRaises(ValueError) as ctx:
            validate_feature_count(X, 3, "TestEstimator")
        self.assertIn("TestEstimator", str(ctx.exception))


class TestCheckIsFitted(unittest.TestCase):
    def test_fitted_attribute_present(self):
        class FakeEstimator:
            coef_ = np.array([1.0])

        check_is_fitted(FakeEstimator(), "coef_")  # should not raise

    def test_missing_attribute_raises(self):
        class FakeEstimator:
            pass

        with self.assertRaises(ValueError):
            check_is_fitted(FakeEstimator(), "coef_")

    def test_multiple_attributes_all_present(self):
        class FakeEstimator:
            coef_ = 1
            intercept_ = 0

        check_is_fitted(FakeEstimator(), ["coef_", "intercept_"])

    def test_multiple_attributes_one_missing_raises(self):
        class FakeEstimator:
            coef_ = 1

        with self.assertRaises(ValueError):
            check_is_fitted(FakeEstimator(), ["coef_", "intercept_"])


if __name__ == "__main__":
    unittest.main()
