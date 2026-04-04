import numpy as np

from .validation import validate_positive_number


def sigmoid(z):
	"""Numerically stable sigmoid."""
	z = np.clip(z, -500.0, 500.0)
	return 1.0 / (1.0 + np.exp(-z))


def soft_threshold(value, threshold):
	"""Soft-thresholding operator used in L1 regularization."""
	if value > threshold:
		return value - threshold
	if value < -threshold:
		return value + threshold
	return 0.0


def linear_kernel(X, Y):
	"""Linear kernel matrix."""
	return X @ Y.T


def rbf_kernel(X, Y, gamma):
	"""RBF kernel matrix."""
	gamma = validate_positive_number(gamma, "gamma")
	X_norm = np.sum(X ** 2, axis=1, keepdims=True)
	Y_norm = np.sum(Y ** 2, axis=1, keepdims=True).T
	squared_distances = np.maximum(X_norm + Y_norm - 2.0 * (X @ Y.T), 0.0)
	return np.exp(-gamma * squared_distances)


def compute_kernel(X, Y, kernel, gamma):
	"""Dispatch kernel computation by kernel name."""
	if kernel == "linear":
		return linear_kernel(X, Y)
	if kernel == "rbf":
		return rbf_kernel(X, Y, gamma)

	raise ValueError("kernel must be one of {'linear', 'rbf'}.")
