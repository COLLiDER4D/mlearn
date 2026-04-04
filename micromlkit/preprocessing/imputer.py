import numpy as np

from ..base import BaseTransformer


class SimpleImputer(BaseTransformer):
	"""Impute missing values using simple column-wise statistics."""

	def __init__(self, strategy="mean", fill_value=0):
		self.strategy = strategy
		self.fill_value = fill_value

	def _is_missing(self, value):
		return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))

	def _validate_X_object(self, X):
		X_obj = np.asarray(X, dtype=object)
		if X_obj.ndim != 2:
			raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

		missing_mask = np.vectorize(self._is_missing)(X_obj)
		return X_obj, missing_mask

	def fit(self, X, y=None):
		"""Compute per-feature imputation values."""
		valid_strategies = {"mean", "median", "most_frequent", "constant"}
		if self.strategy not in valid_strategies:
			raise ValueError(
				"Invalid strategy. Supported strategies are: "
				"'mean', 'median', 'most_frequent', 'constant'."
			)

		X_obj, missing_mask = self._validate_X_object(X)
		self.n_features_in_ = X_obj.shape[1]

		stats = []
		for j in range(self.n_features_in_):
			col = X_obj[:, j]
			observed = col[~missing_mask[:, j]]

			if self.strategy == "constant":
				stats.append(self.fill_value)
				continue

			if observed.size == 0:
				raise ValueError(
					f"Cannot compute statistic for column {j} because it contains only missing values."
				)

			if self.strategy in {"mean", "median"}:
				try:
					values = observed.astype(float)
				except (TypeError, ValueError) as exc:
					raise ValueError(
						f"SimpleImputer with strategy='{self.strategy}' supports only numeric data."
					) from exc

				stat = float(np.mean(values)) if self.strategy == "mean" else float(np.median(values))
				stats.append(stat)
			else:
				uniques, counts = np.unique(observed, return_counts=True)
				stats.append(uniques[np.argmax(counts)])

		self.statistics_ = np.asarray(stats, dtype=object)
		return self

	def fit_transform(self, X, y=None):
		self.fit(X, y)
		return self.transform(X)

	def transform(self, X):
		"""Replace missing values using statistics computed during fit."""
		if not hasattr(self, "statistics_"):
			raise ValueError("This SimpleImputer instance is not fitted yet. Call 'fit' first.")

		X_obj, missing_mask = self._validate_X_object(X)
		if X_obj.shape[1] != self.n_features_in_:
			raise ValueError(
				f"X has {X_obj.shape[1]} features, but SimpleImputer was fitted with "
				f"{self.n_features_in_} features."
			)

		X_out = X_obj.copy()
		for j in range(self.n_features_in_):
			X_out[missing_mask[:, j], j] = self.statistics_[j]

		if self.strategy in {"mean", "median"}:
			return X_out.astype(float)

		return X_out
