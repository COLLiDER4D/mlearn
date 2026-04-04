import numpy as np
from ..base import BaseModel, RegressorMixin
from ..utils.math import soft_threshold
from ..utils.validation import (
	check_is_fitted,
	ensure_1d_array,
	ensure_2d_float_array,
	ensure_same_n_samples,
	validate_feature_count,
)


class Lasso(BaseModel, RegressorMixin):
	"""Lasso Regression model (L1-regularized linear regression)."""

	def __init__(self, alpha=1.0, max_iter=1000, tol=1e-6):
		self.alpha = alpha
		self.max_iter = max_iter
		self.tol = tol

	def _validate_X_y(self, X, y=None):
		X = ensure_2d_float_array(X)

		if y is None:
			return X, None

		y = ensure_1d_array(y, dtype=float)
		ensure_same_n_samples(X, y)

		return X, y

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

				coef[j] = soft_threshold(rho, self.alpha * n_samples) / z

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
		check_is_fitted(self, ("coef_", "intercept_"))

		X, _ = self._validate_X_y(X, None)
		validate_feature_count(X, self.n_features_in_, "Lasso")

		y_pred = X @ self.coef_ + self.intercept_
		return y_pred
