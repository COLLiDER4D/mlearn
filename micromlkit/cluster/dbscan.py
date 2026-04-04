from collections import deque

import numpy as np

from ..base import BaseModel, ClusterMixin
from ._distance import (
	ensure_2d_float_array,
	pairwise_distances,
	validate_feature_count,
	validate_metric,
	validate_minkowski_p,
)


class DBSCAN(BaseModel, ClusterMixin):
	"""Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

	Parameters
	----------
	eps : float, default=0.5
		Maximum neighborhood radius.
	min_samples : int, default=5
		Minimum samples in an epsilon-neighborhood for a core point.
	metric : {'euclidean', 'manhattan', 'cosine', 'chebyshev', 'minkowski'}, default='euclidean'
		Distance metric used for neighborhood queries.
	p : float, default=2
		Power parameter for Minkowski distance when ``metric='minkowski'``.
	"""

	def __init__(self, eps=0.5, min_samples=5, metric="euclidean", p=2):
		self.eps = eps
		self.min_samples = min_samples
		self.metric = metric
		self.p = p

	def _validate_params(self):
		if isinstance(self.eps, bool) or not isinstance(self.eps, (int, float, np.integer, np.floating)):
			raise ValueError("eps must be a positive number.")
		if float(self.eps) <= 0.0:
			raise ValueError("eps must be a positive number.")

		if isinstance(self.min_samples, bool) or not isinstance(self.min_samples, (int, np.integer)):
			raise ValueError("min_samples must be an integer greater than or equal to 1.")
		if self.min_samples < 1:
			raise ValueError("min_samples must be an integer greater than or equal to 1.")

		self.metric_ = validate_metric(self.metric)
		self.p_ = validate_minkowski_p(self.metric_, self.p)

	def fit(self, X, y=None):
		"""Fit DBSCAN model."""
		self._validate_params()
		X = ensure_2d_float_array(X)
		n_samples, n_features = X.shape

		distances = pairwise_distances(X, X, metric=self.metric_, p=self.p_)
		neighborhoods = [np.flatnonzero(distances[i] <= float(self.eps)) for i in range(n_samples)]

		unassigned = -2
		labels = np.full(n_samples, unassigned, dtype=int)
		visited = np.zeros(n_samples, dtype=bool)

		cluster_id = 0
		for i in range(n_samples):
			if visited[i]:
				continue

			visited[i] = True
			neighbor_indices = neighborhoods[i]

			if neighbor_indices.size < int(self.min_samples):
				labels[i] = -1
				continue

			labels[i] = cluster_id
			queue = deque(neighbor_indices.tolist())
			in_queue = np.zeros(n_samples, dtype=bool)
			in_queue[neighbor_indices] = True

			while queue:
				point_index = int(queue.popleft())

				if not visited[point_index]:
					visited[point_index] = True
					point_neighbors = neighborhoods[point_index]

					if point_neighbors.size >= int(self.min_samples):
						for neighbor_index in point_neighbors:
							if not in_queue[neighbor_index]:
								queue.append(int(neighbor_index))
								in_queue[neighbor_index] = True

				if labels[point_index] in (unassigned, -1):
					labels[point_index] = cluster_id

			cluster_id += 1

		labels[labels == unassigned] = -1

		core_mask = np.array(
			[neighborhoods[i].size >= int(self.min_samples) and labels[i] != -1 for i in range(n_samples)],
			dtype=bool,
		)
		core_sample_indices = np.flatnonzero(core_mask)

		self.labels_ = labels
		self.core_sample_indices_ = core_sample_indices
		self.n_features_in_ = n_features
		self._X_fit_ = X
		self._core_samples_ = X[core_sample_indices] if core_sample_indices.size > 0 else np.empty((0, n_features))
		self._core_labels_ = labels[core_sample_indices] if core_sample_indices.size > 0 else np.empty((0,), dtype=int)

		return self

	def fit_predict(self, X, y=None):
		"""Fit model and return training-set cluster labels from the density expansion."""
		return self.fit(X, y).labels_

	def predict(self, X):
		"""Predict labels for unseen samples.

		A sample is assigned the label of its nearest core sample if the
		core-sample distance is <= eps; otherwise it is labeled as noise (-1).
		"""
		if not hasattr(self, "labels_"):
			raise ValueError("This DBSCAN instance is not fitted yet. Call 'fit' first.")

		X = ensure_2d_float_array(X)
		validate_feature_count(X, self.n_features_in_, "DBSCAN")

		if self._core_samples_.shape[0] == 0:
			return np.full(X.shape[0], -1, dtype=int)

		core_distances = pairwise_distances(X, self._core_samples_, metric=self.metric_, p=self.p_)
		nearest_core_idx = np.argmin(core_distances, axis=1)
		nearest_core_dist = core_distances[np.arange(X.shape[0]), nearest_core_idx]

		preds = np.full(X.shape[0], -1, dtype=int)
		inlier_mask = nearest_core_dist <= float(self.eps)
		preds[inlier_mask] = self._core_labels_[nearest_core_idx[inlier_mask]]
		return preds
