import numpy as np

from ..base import BaseTransformer


class StandardScaler(BaseTransformer):
	"""Standardize features by removing the mean and scaling to unit variance."""

	def __init__(self, with_mean=True, with_std=True):
		self.with_mean = with_mean
		self.with_std = with_std

	def _validate_X(self, X):
		X = np.asarray(X, dtype=float)
		if X.ndim != 2:
			raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
		return X

	def fit(self, X, y=None):
		"""Compute scaling statistics from the training data."""
		X = self._validate_X(X)

		self.n_features_in_ = X.shape[1]
		self.mean_ = np.mean(X, axis=0) if self.with_mean else np.zeros(self.n_features_in_)

		if self.with_std:
			scale = np.std(X, axis=0)
			scale[scale == 0.0] = 1.0
			self.scale_ = scale
		else:
			self.scale_ = np.ones(self.n_features_in_)

		return self

	def fit_transform(self, X, y=None):
		self.fit(X, y)
		return self.transform(X)

	def transform(self, X):
		"""Scale features of X according to previously computed statistics."""
		if not hasattr(self, "mean_") or not hasattr(self, "scale_"):
			raise ValueError("This StandardScaler instance is not fitted yet. Call 'fit' first.")

		X = self._validate_X(X)
		if X.shape[1] != self.n_features_in_:
			raise ValueError(
				f"X has {X.shape[1]} features, but StandardScaler was fitted with "
				f"{self.n_features_in_} features."
			)

		return (X - self.mean_) / self.scale_
