import numpy as np
from ..base import BaseModel, ClassifierMixin
from ..utils.math import sigmoid
from ..utils.validation import (
	check_is_fitted,
	ensure_1d_array,
	ensure_2d_float_array,
	ensure_same_n_samples,
	validate_feature_count,
)


class LogisticRegression(BaseModel, ClassifierMixin):
	"""Binary Logistic Regression classifier trained with gradient descent."""

	def __init__(self, learning_rate=0.1, n_iters=1000, tol=1e-6):
		self.learning_rate = learning_rate
		self.n_iters = n_iters
		self.tol = tol

	def _validate_X_y(self, X, y=None):
		X = ensure_2d_float_array(X)

		if y is None:
			return X, None

		y = ensure_1d_array(y, dtype=float)
		ensure_same_n_samples(X, y)

		unique_labels = np.unique(y)
		if not np.all(np.isin(unique_labels, [0.0, 1.0])):
			raise ValueError("y must contain binary labels encoded as 0 and 1.")

		return X, y

	def fit(self, X, y):
		"""Fit the logistic regression model to the data.

		Parameters:
		- X: array-like of shape (n_samples, n_features)
			The input samples.
		- y: array-like of shape (n_samples,)
			Binary target values encoded as 0 and 1.
		"""
		X, y = self._validate_X_y(X, y)

		n_samples, n_features = X.shape
		self.n_features_in_ = n_features

		self.coef_ = np.zeros(n_features, dtype=float)
		self.intercept_ = 0.0

		for _ in range(self.n_iters):
			linear = X @ self.coef_ + self.intercept_
			y_pred = sigmoid(linear)

			error = y_pred - y
			grad_w = (X.T @ error) / n_samples
			grad_b = np.sum(error) / n_samples

			new_coef = self.coef_ - self.learning_rate * grad_w
			new_intercept = self.intercept_ - self.learning_rate * grad_b

			max_step = max(
				np.max(np.abs(new_coef - self.coef_)),
				abs(new_intercept - self.intercept_),
			)

			self.coef_ = new_coef
			self.intercept_ = float(new_intercept)

			if max_step < self.tol:
				break

		return self

	def predict_proba(self, X):
		"""Estimate class probabilities for each input sample.

		Parameters:
		- X: array-like of shape (n_samples, n_features)
			The input samples.

		Returns:
		- proba: array-like of shape (n_samples, 2)
			Class probabilities in the order [P(class=0), P(class=1)].
		"""
		check_is_fitted(self, ("coef_", "intercept_"))

		X, _ = self._validate_X_y(X, None)
		validate_feature_count(X, self.n_features_in_, "LogisticRegression")

		proba_pos = sigmoid(X @ self.coef_ + self.intercept_)
		proba_neg = 1.0 - proba_pos
		return np.column_stack((proba_neg, proba_pos))

	def predict(self, X):
		"""Predict class labels for samples in X.

		Parameters:
		- X: array-like of shape (n_samples, n_features)
			The input samples.

		Returns:
		- y_pred: array-like of shape (n_samples,)
			Predicted binary class labels (0 or 1).
		"""
		proba = self.predict_proba(X)[:, 1]
		return (proba >= 0.5).astype(int)
