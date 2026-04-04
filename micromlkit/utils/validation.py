import numpy as np


def ensure_2d_float_array(X, name="X", require_non_empty=False):
	"""Validate and return X as a 2D float NumPy array."""
	X = np.asarray(X, dtype=float)
	if X.ndim != 2:
		raise ValueError(f"{name} must be a 2D array of shape (n_samples, n_features).")

	if require_non_empty and (X.shape[0] == 0 or X.shape[1] == 0):
		raise ValueError(f"{name} must contain at least one sample and one feature.")

	return X


def ensure_1d_array(y, name="y", dtype=None):
	"""Validate and return y as a 1D NumPy array."""
	if dtype is None:
		y = np.asarray(y)
	else:
		y = np.asarray(y, dtype=dtype)

	if y.ndim != 1:
		raise ValueError(f"{name} must be a 1D array of shape (n_samples,).")

	return y


def ensure_same_n_samples(X, y):
	"""Ensure X and y contain the same number of samples."""
	if X.shape[0] != y.shape[0]:
		raise ValueError("X and y must have the same number of samples.")


def validate_feature_count(X, n_features_in, estimator_name):
	"""Validate feature-count consistency between fit and inference."""
	if X.shape[1] != n_features_in:
		raise ValueError(
			f"X has {X.shape[1]} features, but {estimator_name} was fitted with "
			f"{n_features_in} features."
		)


def validate_positive_integer(value, name):
	"""Validate and return a positive integer."""
	if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
		raise ValueError(f"{name} must be a positive integer.")

	value = int(value)
	if value <= 0:
		raise ValueError(f"{name} must be a positive integer.")

	return value


def validate_positive_number(value, name):
	"""Validate and return a positive numeric value."""
	if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
		raise ValueError(f"{name} must be a positive number.")

	value = float(value)
	if value <= 0.0:
		raise ValueError(f"{name} must be a positive number.")

	return value


def validate_non_negative_number(value, name):
	"""Validate and return a non-negative numeric value."""
	if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
		raise ValueError(f"{name} must be a non-negative number.")

	value = float(value)
	if value < 0.0:
		raise ValueError(f"{name} must be a non-negative number.")

	return value


def validate_random_state(random_state):
	"""Validate and normalize random_state."""
	if random_state is None:
		return None

	if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
		raise ValueError("random_state must be an integer or None.")

	return int(random_state)


def check_is_fitted(estimator, attributes):
	"""Raise if estimator does not have all required fitted attributes."""
	if isinstance(attributes, str):
		attributes = (attributes,)

	if not all(hasattr(estimator, attr) for attr in attributes):
		raise ValueError(
			f"This {estimator.__class__.__name__} instance is not fitted yet. "
			"Call 'fit' first."
		)
