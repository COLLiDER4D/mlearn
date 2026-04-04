import numpy as np

from ..base import BaseModel, ClusterMixin
from ._distance import (
	ensure_2d_float_array,
	pairwise_distances,
	validate_feature_count,
	validate_metric,
	validate_minkowski_p,
)


class KMeans(BaseModel, ClusterMixin):
	"""K-Means clustering.

	Parameters
	----------
	n_clusters : int, default=8
		Number of clusters.
	max_iter : int, default=300
		Maximum number of optimization iterations.
	tol : float, default=1e-4
		Convergence tolerance based on center movement.
	random_state : int or None, default=None
		Random seed for center initialization.
	metric : {'euclidean', 'manhattan', 'cosine', 'chebyshev', 'minkowski'}, default='euclidean'
		Distance metric used for assignment.
	p : float, default=2
		Power parameter for Minkowski distance when ``metric='minkowski'``.
	"""

	def __init__(
		self,
		n_clusters=8,
		max_iter=300,
		tol=1e-4,
		random_state=None,
		metric="euclidean",
		p=2,
	):
		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.tol = tol
		self.random_state = random_state
		self.metric = metric
		self.p = p

	def _validate_params(self):
		if isinstance(self.n_clusters, bool) or not isinstance(self.n_clusters, (int, np.integer)):
			raise ValueError("n_clusters must be a positive integer.")
		if self.n_clusters <= 0:
			raise ValueError("n_clusters must be a positive integer.")

		if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, (int, np.integer)):
			raise ValueError("max_iter must be a positive integer.")
		if self.max_iter <= 0:
			raise ValueError("max_iter must be a positive integer.")

		if isinstance(self.tol, bool) or not isinstance(self.tol, (int, float, np.integer, np.floating)):
			raise ValueError("tol must be a positive number.")
		if float(self.tol) <= 0.0:
			raise ValueError("tol must be a positive number.")

		self.metric_ = validate_metric(self.metric)
		self.p_ = validate_minkowski_p(self.metric_, self.p)

	def fit(self, X, y=None):
		"""Fit K-Means model."""
		self._validate_params()
		X = ensure_2d_float_array(X, require_non_empty=True)
		n_samples, n_features = X.shape

		if self.n_clusters > n_samples:
			raise ValueError("n_clusters must be less than or equal to the number of samples.")

		rng = np.random.default_rng(self.random_state)
		center_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
		centers = X[center_indices].astype(float, copy=True)

		for iteration in range(int(self.max_iter)):
			distances = pairwise_distances(X, centers, metric=self.metric_, p=self.p_)
			labels = np.argmin(distances, axis=1).astype(int)
			min_distances = np.min(distances, axis=1)

			new_centers = centers.copy()
			used_replacements = np.zeros(n_samples, dtype=bool)

			for cluster_index in range(int(self.n_clusters)):
				mask = labels == cluster_index
				if np.any(mask):
					new_centers[cluster_index] = np.mean(X[mask], axis=0)
					continue

				# Empty cluster: re-seed with a far point to keep deterministic progress.
				candidate_order = np.argsort(-min_distances)
				replacement_index = None
				for idx in candidate_order:
					if not used_replacements[idx]:
						replacement_index = int(idx)
						used_replacements[idx] = True
						break

				if replacement_index is None:
					replacement_index = int(candidate_order[0])

				new_centers[cluster_index] = X[replacement_index]

			center_shift = np.linalg.norm(new_centers - centers)
			centers = new_centers
			if center_shift <= float(self.tol):
				break

		final_distances = pairwise_distances(X, centers, metric=self.metric_, p=self.p_)
		final_labels = np.argmin(final_distances, axis=1).astype(int)
		closest_distances = np.min(final_distances, axis=1)

		self.cluster_centers_ = centers
		self.labels_ = final_labels
		self.n_features_in_ = n_features
		self.n_iter_ = iteration + 1
		if self.metric_ == "euclidean":
			self.inertia_ = float(np.sum(closest_distances ** 2))
		else:
			self.inertia_ = float(np.sum(closest_distances))

		return self

	def predict(self, X):
		"""Predict cluster index for each sample in X."""
		if not hasattr(self, "cluster_centers_"):
			raise ValueError("This KMeans instance is not fitted yet. Call 'fit' first.")

		X = ensure_2d_float_array(X, require_non_empty=True)
		validate_feature_count(X, self.n_features_in_, "KMeans")

		distances = pairwise_distances(X, self.cluster_centers_, metric=self.metric_, p=self.p_)
		return np.argmin(distances, axis=1).astype(int)
