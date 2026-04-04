import numpy as np

from ..base import BaseModel, ClusterMixin
from ._distance import (
	ensure_2d_float_array,
	pairwise_distances,
	validate_feature_count,
	validate_metric,
	validate_minkowski_p,
)


class AgglomerativeClustering(BaseModel, ClusterMixin):
	"""Agglomerative hierarchical clustering.

	Parameters
	----------
	n_clusters : int, default=2
		Number of clusters to find.
	linkage : {'single', 'complete', 'average'}, default='single'
		Linkage criterion used during cluster merges.
	metric : {'euclidean', 'manhattan', 'cosine', 'chebyshev', 'minkowski'}, default='euclidean'
		Distance metric used for pairwise distances.
	p : float, default=2
		Power parameter for Minkowski distance when ``metric='minkowski'``.
	"""

	def __init__(self, n_clusters=2, linkage="single", metric="euclidean", p=2):
		self.n_clusters = n_clusters
		self.linkage = linkage
		self.metric = metric
		self.p = p

	def _validate_params(self):
		if isinstance(self.n_clusters, bool) or not isinstance(self.n_clusters, (int, np.integer)):
			raise ValueError("n_clusters must be a positive integer.")
		if self.n_clusters <= 0:
			raise ValueError("n_clusters must be a positive integer.")

		if not isinstance(self.linkage, str):
			raise ValueError("linkage must be one of {'single', 'complete', 'average'}.")
		self.linkage_ = self.linkage.lower()
		if self.linkage_ not in {"single", "complete", "average"}:
			raise ValueError("linkage must be one of {'single', 'complete', 'average'}.")

		self.metric_ = validate_metric(self.metric)
		self.p_ = validate_minkowski_p(self.metric_, self.p)

	def _linkage_distance(self, left_indices, right_indices, distance_matrix):
		pairwise = distance_matrix[np.ix_(left_indices, right_indices)]

		if self.linkage_ == "single":
			return float(np.min(pairwise))
		if self.linkage_ == "complete":
			return float(np.max(pairwise))
		return float(np.mean(pairwise))

	def fit(self, X, y=None):
		"""Fit agglomerative clustering model."""
		self._validate_params()
		X = ensure_2d_float_array(X)
		n_samples, n_features = X.shape

		if self.n_clusters > n_samples:
			raise ValueError("n_clusters must be less than or equal to the number of samples.")

		distance_matrix = pairwise_distances(X, X, metric=self.metric_, p=self.p_)

		clusters = {i: [i] for i in range(n_samples)}
		active_cluster_ids = list(clusters.keys())
		next_cluster_id = n_samples

		while len(active_cluster_ids) > int(self.n_clusters):
			best_pair = None
			best_distance = np.inf

			for i in range(len(active_cluster_ids)):
				for j in range(i + 1, len(active_cluster_ids)):
					left_id = active_cluster_ids[i]
					right_id = active_cluster_ids[j]
					dist = self._linkage_distance(
						clusters[left_id],
						clusters[right_id],
						distance_matrix,
					)
					if dist < best_distance:
						best_distance = dist
						best_pair = (left_id, right_id)

			left_id, right_id = best_pair
			merged_members = clusters[left_id] + clusters[right_id]
			clusters[next_cluster_id] = merged_members

			active_cluster_ids = [cid for cid in active_cluster_ids if cid not in best_pair]
			active_cluster_ids.append(next_cluster_id)

			del clusters[left_id]
			del clusters[right_id]
			next_cluster_id += 1

		final_clusters = sorted(active_cluster_ids)
		labels = np.full(n_samples, -1, dtype=int)
		representative_indices = []

		for label_idx, cluster_id in enumerate(final_clusters):
			member_indices = np.asarray(clusters[cluster_id], dtype=int)
			labels[member_indices] = label_idx

			if member_indices.size == 1:
				representative_indices.append(int(member_indices[0]))
			else:
				member_distances = distance_matrix[np.ix_(member_indices, member_indices)]
				medoid_local_idx = int(np.argmin(np.sum(member_distances, axis=1)))
				representative_indices.append(int(member_indices[medoid_local_idx]))

		self.labels_ = labels
		self.n_features_in_ = n_features
		self._X_fit_ = X
		self._cluster_representatives_ = X[np.asarray(representative_indices, dtype=int)]
		self._representative_labels_ = np.arange(len(representative_indices), dtype=int)

		return self

	def predict(self, X):
		"""Predict cluster labels for unseen samples via nearest representative."""
		if not hasattr(self, "labels_"):
			raise ValueError(
				"This AgglomerativeClustering instance is not fitted yet. Call 'fit' first."
			)

		X = ensure_2d_float_array(X)
		validate_feature_count(X, self.n_features_in_, "AgglomerativeClustering")

		distances = pairwise_distances(
			X,
			self._cluster_representatives_,
			metric=self.metric_,
			p=self.p_,
		)
		nearest_idx = np.argmin(distances, axis=1)
		return self._representative_labels_[nearest_idx]
