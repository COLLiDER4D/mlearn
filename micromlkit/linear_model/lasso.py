import numpy as np
from ..base import BaseModel, RegressorMixin


class Lasso(BaseModel, RegressorMixin):
	"""Lasso Regression model (L1-regularized linear regression)."""

	def __init__(self, alpha=1.0, max_iter=1000, tol=1e-6):
		self.alpha = alpha
		self.max_iter = max_iter
		self.tol = tol

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

	def _soft_threshold(self, value, threshold):
		if value > threshold:
			return value - threshold
		if value < -threshold:
			return value + threshold
		return 0.0

	def fit(self, X, y):
		"""Fit the lasso regression model to the data.

		Parameters:
		- X: array-like of shape (n_samples, n_features)
			The input samples.
		- y: array-like of shape (n_samples,)
			The target values.
		"""
		if self.alpha < 0:
			raise ValueError("alpha must be non-negative.")
		if self.max_iter <= 0:
			raise ValueError("max_iter must be a positive integer.")
		if self.tol < 0:
			raise ValueError("tol must be non-negative.")

		X, y = self._validate_X_y(X, y)

		n_samples, n_features = X.shape
		self.n_features_in_ = n_features

		# Center X and y so the intercept is not regularized.
		X_mean = np.mean(X, axis=0)
		y_mean = np.mean(y)
		X_centered = X - X_mean
		y_centered = y - y_mean

		coef = np.zeros(n_features, dtype=float)

		for _ in range(self.max_iter):
			coef_old = coef.copy()

			for j in range(n_features):
				# Partial residual excluding feature j
				residual = y_centered - (X_centered @ coef) + X_centered[:, j] * coef[j]
				rho = X_centered[:, j] @ residual
				z = np.sum(X_centered[:, j] ** 2)

				if z == 0.0:
					coef[j] = 0.0
					continue

				coef[j] = self._soft_threshold(rho, self.alpha * n_samples) / z

			max_delta = np.max(np.abs(coef - coef_old))
			if max_delta < self.tol:
				break

		self.coef_ = coef
		self.intercept_ = float(y_mean - X_mean @ self.coef_)
		return self

	def predict(self, X):
		"""Predict using the lasso regression model.

		Parameters:
		- X: array-like of shape (n_samples, n_features)
			The input samples.

		Returns:
		- y_pred: array-like of shape (n_samples,)
			The predicted values.
		"""
		if not hasattr(self, "coef_") or not hasattr(self, "intercept_"):
			raise ValueError("This Lasso instance is not fitted yet. Call 'fit' first.")

		X, _ = self._validate_X_y(X, None)
		if X.shape[1] != self.n_features_in_:
			raise ValueError(
				f"X has {X.shape[1]} features, but Lasso was fitted with "
				f"{self.n_features_in_} features."
			)

		y_pred = X @ self.coef_ + self.intercept_
		return y_pred
