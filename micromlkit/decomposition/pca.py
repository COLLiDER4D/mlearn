import numpy as np

from ..base import BaseTransformer


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
		X = np.asarray(X, dtype=float)
		if X.ndim != 2:
			raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
		if X.shape[0] == 0 or X.shape[1] == 0:
			raise ValueError("X must contain at least one sample and one feature.")
		return X

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
		if not hasattr(self, "components_") or not hasattr(self, "mean_"):
			raise ValueError("This PCA instance is not fitted yet. Call 'fit' first.")

		X = self._validate_X(X)
		if X.shape[1] != self.n_features_in_:
			raise ValueError(
				f"X has {X.shape[1]} features, but PCA was fitted with "
				f"{self.n_features_in_} features."
			)

		return (X - self.mean_) @ self.components_.T
