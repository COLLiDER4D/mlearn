import numpy as np

from ..base import BaseModel, RegressorMixin
from ..cluster._distance import (
	ensure_2d_float_array,
	pairwise_distances,
	validate_feature_count,
	validate_metric,
	validate_minkowski_p,
)


class KNNRegressor(BaseModel, RegressorMixin):
	"""K-Nearest Neighbors regressor."""

	def __init__(self, n_neighbors=5, metric="euclidean", p=2):
		self.n_neighbors = n_neighbors
		self.metric = metric
		self.p = p

	def _validate_params(self):
		if isinstance(self.n_neighbors, bool) or not isinstance(self.n_neighbors, (int, np.integer)):
			raise ValueError("n_neighbors must be a positive integer.")
		if self.n_neighbors <= 0:
			raise ValueError("n_neighbors must be a positive integer.")

		self.metric_ = validate_metric(self.metric)
		self.p_ = validate_minkowski_p(self.metric_, self.p)

	def _validate_X_y(self, X, y=None):
		X = ensure_2d_float_array(X)

		if y is None:
			return X, None

		y = np.asarray(y, dtype=float)
		if y.ndim != 1:
			raise ValueError("y must be a 1D array of shape (n_samples,).")
		if X.shape[0] != y.shape[0]:
			raise ValueError("X and y must have the same number of samples.")

		return X, y

	def fit(self, X, y):
		"""Fit the KNN regressor."""
		self._validate_params()
		X, y = self._validate_X_y(X, y)

		n_samples, n_features = X.shape
		if self.n_neighbors > n_samples:
			raise ValueError("n_neighbors must be less than or equal to the number of samples.")

		self.X_train_ = X
		self.y_train_ = y
		self.n_features_in_ = n_features
		return self

	def predict(self, X):
		"""Predict regression targets for samples in X."""
		if not hasattr(self, "X_train_") or not hasattr(self, "y_train_"):
			raise ValueError("This KNNRegressor instance is not fitted yet. Call 'fit' first.")

		X, _ = self._validate_X_y(X, None)
		validate_feature_count(X, self.n_features_in_, "KNNRegressor")

		distances = pairwise_distances(X, self.X_train_, metric=self.metric_, p=self.p_)
		neighbor_indices = np.argpartition(
			distances, kth=self.n_neighbors - 1, axis=1
		)[:, : self.n_neighbors]
		neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
		neighbor_order = np.argsort(neighbor_distances, axis=1)
		neighbor_indices = np.take_along_axis(neighbor_indices, neighbor_order, axis=1)
		neighbor_targets = self.y_train_[neighbor_indices]

		return np.mean(neighbor_targets, axis=1)
