"""Utility math and validation helpers used across :mod:`micromlkit`.

Examples
--------
>>> from micromlkit.utils import sigmoid, validate_positive_integer
>>> round(float(sigmoid(0.0)), 1)
0.5
>>> validate_positive_integer(3, "n_neighbors")
3
"""

from .math import compute_kernel, linear_kernel, rbf_kernel, sigmoid, soft_threshold
from .validation import (
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

__all__ = [
	"check_is_fitted",
	"compute_kernel",
	"ensure_1d_array",
	"ensure_2d_float_array",
	"ensure_same_n_samples",
	"linear_kernel",
	"rbf_kernel",
	"sigmoid",
	"soft_threshold",
	"validate_feature_count",
	"validate_non_negative_number",
	"validate_positive_integer",
	"validate_positive_number",
	"validate_random_state",
]
