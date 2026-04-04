import numpy as np
from ..base import BaseModel, RegressorMixin
from ..utils.validation import (
	check_is_fitted,
	ensure_1d_array,
	ensure_2d_float_array,
	ensure_same_n_samples,
	validate_feature_count,
)


class Ridge(BaseModel, RegressorMixin):
	"""Ridge Regression model (L2-regularized linear regression)."""

	def __init__(self, alpha=1.0):
		self.alpha = alpha

	def _validate_X_y(self, X, y=None):
		X = ensure_2d_float_array(X)

		if y is None:
			return X, None

		y = ensure_1d_array(y, dtype=float)
		ensure_same_n_samples(X, y)

		return X, y

	def fit(self, X, y):
		"""Fit the ridge regression model to the data.

		Parameters:
		- X: array-like of shape (n_samples, n_features)
			The input samples.
		- y: array-like of shape (n_samples,)
			The target values.
		"""
		if self.alpha < 0:
			raise ValueError("alpha must be non-negative.")

		X, y = self._validate_X_y(X, y)

		n_samples, n_features = X.shape
		self.n_features_in_ = n_features
		X_intercept = np.hstack((np.ones((n_samples, 1)), X))

		# Ridge normal equation:
		# beta = (X^T X + alpha * I)^(-1) X^T y
		# Do not regularize intercept term.
		penalty = self.alpha * np.eye(n_features + 1)
		penalty[0, 0] = 0.0

		beta = np.linalg.pinv(X_intercept.T @ X_intercept + penalty) @ X_intercept.T @ y
		self.intercept_ = float(beta[0])
		self.coef_ = beta[1:]
		return self

	def predict(self, X):
		"""Predict using the ridge regression model.

		Parameters:
		- X: array-like of shape (n_samples, n_features)
			The input samples.

		Returns:
		- y_pred: array-like of shape (n_samples,)
			The predicted values.
		"""
		check_is_fitted(self, ("coef_", "intercept_"))

		X, _ = self._validate_X_y(X, None)
		validate_feature_count(X, self.n_features_in_, "Ridge")

		y_pred = X @ self.coef_ + self.intercept_
		return y_pred
