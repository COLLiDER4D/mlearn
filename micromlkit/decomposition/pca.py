import numpy as np

from ..base import BaseTransformer
from ..utils.validation import check_is_fitted, ensure_2d_float_array, validate_feature_count


class PCA(BaseTransformer):
	"""Principal Component Analysis (PCA) using SVD.

	Parameters
	----------
	n_components : int or None, default=None
		Number of principal components to keep. If None, keep
		``min(n_samples, n_features)`` components.
	"""

	def __init__(self, n_components=None):
		self.n_components = n_components

	def _validate_X(self, X):
		return ensure_2d_float_array(X, require_non_empty=True)

	def _resolve_n_components(self, n_samples, n_features):
		max_components = min(n_samples, n_features)
		if self.n_components is None:
			return max_components

		if isinstance(self.n_components, bool):
			raise ValueError("n_components must be an integer or None.")

		if not isinstance(self.n_components, (int, np.integer)):
			raise ValueError("n_components must be an integer or None.")

		n_components = int(self.n_components)
		if n_components <= 0:
			raise ValueError("n_components must be greater than 0.")
		if n_components > max_components:
			raise ValueError(
				f"n_components={n_components} is invalid for X with shape "
				f"({n_samples}, {n_features}). Expected n_components <= {max_components}."
			)
		return n_components

	def fit(self, X, y=None):
		"""Fit PCA model with X.

		Computes principal axes and explained variance using SVD on centered data.
		"""
		X = self._validate_X(X)
		n_samples, n_features = X.shape

		self.n_features_in_ = n_features
		self.n_samples_seen_ = n_samples
		self.mean_ = np.mean(X, axis=0)

		n_components = self._resolve_n_components(n_samples, n_features)
		X_centered = X - self.mean_
		_, singular_values, vt = np.linalg.svd(X_centered, full_matrices=False)

		if n_samples > 1:
			full_explained_variance = (singular_values ** 2) / (n_samples - 1)
		else:
			full_explained_variance = np.zeros_like(singular_values)

		total_variance = np.sum(full_explained_variance)
		if total_variance > 0.0:
			full_explained_variance_ratio = full_explained_variance / total_variance
		else:
			full_explained_variance_ratio = np.zeros_like(full_explained_variance)

		self.components_ = vt[:n_components]
		self.explained_variance_ = full_explained_variance[:n_components]
		self.explained_variance_ratio_ = full_explained_variance_ratio[:n_components]

		return self

	def fit_transform(self, X, y=None):
		self.fit(X, y)
		return self.transform(X)

	def transform(self, X):
		"""Apply dimensionality reduction to X."""
		check_is_fitted(self, ("components_", "mean_"))

		X = self._validate_X(X)
		validate_feature_count(X, self.n_features_in_, "PCA")

		return (X - self.mean_) @ self.components_.T
