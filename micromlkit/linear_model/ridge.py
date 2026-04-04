import numpy as np
from ..base import BaseModel, RegressorMixin


class Ridge(BaseModel, RegressorMixin):
	"""Ridge Regression model (L2-regularized linear regression)."""

	def __init__(self, alpha=1.0):
		self.alpha = alpha

	def _validate_X_y(self, X, y=None):
		X = np.asarray(X, dtype=float)
		if X.ndim != 2:
			raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

		if y is None:
			return X, None

		y = np.asarray(y, dtype=float)
		if y.ndim != 1:
			raise ValueError("y must be a 1D array of shape (n_samples,).")
		if X.shape[0] != y.shape[0]:
			raise ValueError("X and y must have the same number of samples.")

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
		if not hasattr(self, "coef_") or not hasattr(self, "intercept_"):
			raise ValueError("This Ridge instance is not fitted yet. Call 'fit' first.")

		X, _ = self._validate_X_y(X, None)
		if X.shape[1] != self.n_features_in_:
			raise ValueError(
				f"X has {X.shape[1]} features, but Ridge was fitted with "
				f"{self.n_features_in_} features."
			)

		y_pred = X @ self.coef_ + self.intercept_
		return y_pred
