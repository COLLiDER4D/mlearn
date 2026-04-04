import numpy as np

from ..base import BaseTransformer


class LabelEncoder(BaseTransformer):
	"""Encode target labels with value between 0 and n_classes-1."""

	def _validate_y(self, y):
		y = np.asarray(y, dtype=object)
		if y.ndim != 1:
			raise ValueError("y must be a 1D array of shape (n_samples,).")
		return y

	def fit(self, X, y=None):
		"""Fit label encoder using the input label vector."""
		y_arr = self._validate_y(X)

		self.classes_ = np.unique(y_arr)
		self.class_to_index_ = {label: idx for idx, label in enumerate(self.classes_)}
		return self

	def fit_transform(self, X, y=None):
		self.fit(X, y)
		return self.transform(X)

	def transform(self, X):
		"""Transform labels to normalized integer encoding."""
		if not hasattr(self, "classes_"):
			raise ValueError("This LabelEncoder instance is not fitted yet. Call 'fit' first.")

		y_arr = self._validate_y(X)

		unseen = sorted({label for label in y_arr if label not in self.class_to_index_})
		if unseen:
			raise ValueError(f"y contains previously unseen labels: {unseen}")

		return np.asarray([self.class_to_index_[label] for label in y_arr], dtype=int)

	def inverse_transform(self, X):
		"""Transform integer labels back to original encoding."""
		if not hasattr(self, "classes_"):
			raise ValueError("This LabelEncoder instance is not fitted yet. Call 'fit' first.")

		indices = np.asarray(X)
		if indices.ndim != 1:
			raise ValueError("X must be a 1D array of shape (n_samples,).")

		if not np.issubdtype(indices.dtype, np.integer):
			try:
				float_indices = indices.astype(float)
			except (TypeError, ValueError):
				raise ValueError("Encoded labels must be numeric integers.")

			if np.any(float_indices != np.floor(float_indices)):
				raise ValueError("Encoded labels must be integers.")
			indices = float_indices.astype(int)

		if np.any(indices < 0) or np.any(indices >= len(self.classes_)):
			raise ValueError("Encoded labels are out of range.")

		return self.classes_[indices]
