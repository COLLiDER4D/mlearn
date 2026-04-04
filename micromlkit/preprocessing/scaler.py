import numpy as np

from ..base import BaseTransformer
from ..utils.validation import check_is_fitted, ensure_2d_float_array, validate_feature_count


class StandardScaler(BaseTransformer):
	"""Standardize features by removing the mean and scaling to unit variance."""

	def __init__(self, with_mean=True, with_std=True):
		self.with_mean = with_mean
		self.with_std = with_std

	def _validate_X(self, X):
		return ensure_2d_float_array(X)

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
		check_is_fitted(self, ("mean_", "scale_"))

		X = self._validate_X(X)
		validate_feature_count(X, self.n_features_in_, "StandardScaler")

		return (X - self.mean_) / self.scale_
