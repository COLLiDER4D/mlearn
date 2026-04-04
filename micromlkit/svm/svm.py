import numpy as np

from ..base import BaseModel, ClassifierMixin, RegressorMixin
from ..utils.math import compute_kernel as _compute_kernel
from ..utils.validation import (
	validate_non_negative_number as _validate_non_negative_number,
	validate_positive_integer as _validate_positive_integer,
	validate_positive_number as _validate_positive_number,
	validate_random_state as _validate_random_state,
)


_ALLOWED_KERNELS = {"linear", "rbf"}
_ALLOWED_GAMMA_STRINGS = {"scale", "auto"}


def _resolve_gamma(gamma, X):
	if isinstance(gamma, str):
		gamma = gamma.lower()
		if gamma not in _ALLOWED_GAMMA_STRINGS:
			raise ValueError("gamma must be one of {'scale', 'auto'} or a positive number.")

		n_features = X.shape[1]
		if gamma == "auto":
			return 1.0 / n_features

		variance = float(np.var(X))
		if variance <= 0.0:
			return 1.0
		return 1.0 / (n_features * variance)

	if isinstance(gamma, bool) or not isinstance(gamma, (int, float, np.integer, np.floating)):
		raise ValueError("gamma must be one of {'scale', 'auto'} or a positive number.")

	gamma = float(gamma)
	if gamma <= 0.0:
		raise ValueError("gamma must be one of {'scale', 'auto'} or a positive number.")
	return gamma


def _fit_binary_kernel_classifier(X, y_binary, C, kernel, gamma, tol, K=None):
	if K is None:
		K = _compute_kernel(X, X, kernel=kernel, gamma=gamma)
	regularization = 1.0 / C
	system = K + regularization * np.eye(K.shape[0])
	beta = np.linalg.solve(system, y_binary)
	bias = float(np.mean(y_binary - K @ beta))

	support_indices = np.flatnonzero(np.abs(beta) > tol)
	if support_indices.size == 0:
		support_indices = np.arange(X.shape[0], dtype=int)

	model = {
		"X_train": X,
		"beta": beta,
		"bias": bias,
		"support_indices": support_indices,
		"kernel": kernel,
		"gamma": gamma,
	}

	if kernel == "linear":
		model["coef"] = X.T @ beta

	return model


def _fit_epsilon_insensitive_kernel_regressor(X, y, C, epsilon, kernel, gamma, tol, max_iter=100):
	regularization = 1.0 / C
	n_samples = X.shape[0]

	if kernel == "linear":
		X_aug = np.hstack([X, np.ones((n_samples, 1))])
		reg_matrix = np.eye(X_aug.shape[1])
		reg_matrix[-1, -1] = 0.0

		initial_system = X_aug.T @ X_aug + regularization * reg_matrix
		initial_solution = np.linalg.solve(initial_system, X_aug.T @ y)
		coef = initial_solution[:-1]
		bias = float(initial_solution[-1])

		previous_active = None
		for _ in range(max_iter):
			train_predictions = X @ coef + bias
			signed_residual = y - train_predictions
			active = np.abs(signed_residual) > epsilon + tol

			if not np.any(active):
				break

			sign = np.sign(signed_residual[active])
			adjusted_targets = y[active] - epsilon * sign
			X_active = X[active]
			X_active_aug = np.hstack([X_active, np.ones((X_active.shape[0], 1))])

			system = X_active_aug.T @ X_active_aug + regularization * reg_matrix
			solution = np.linalg.solve(system, X_active_aug.T @ adjusted_targets)
			new_coef = solution[:-1]
			new_bias = float(solution[-1])

			if (
				previous_active is not None
				and np.array_equal(active, previous_active)
				and np.allclose(new_coef, coef, atol=tol, rtol=0.0)
				and abs(new_bias - bias) <= tol
			):
				coef = new_coef
				bias = new_bias
				break

			coef = new_coef
			bias = new_bias
			previous_active = active.copy()

		# Recover a sample-space coefficient vector for consistent SVM-style attributes.
		beta = np.linalg.lstsq(X.T, coef, rcond=None)[0]
		return {
			"X_train": X,
			"beta": beta,
			"bias": bias,
			"kernel": kernel,
			"gamma": gamma,
			"coef": coef,
		}

	K = _compute_kernel(X, X, kernel=kernel, gamma=gamma)
	initial_system = K + regularization * np.eye(n_samples)
	beta = np.linalg.solve(initial_system, y)
	bias = float(np.mean(y - K @ beta))

	previous_active = None
	for _ in range(max_iter):
		train_predictions = K @ beta + bias
		signed_residual = y - train_predictions
		active = np.abs(signed_residual) > epsilon + tol

		if not np.any(active):
			break

		sign = np.sign(signed_residual[active])
		adjusted_targets = y[active] - epsilon * sign
		K_active = K[active, :]
		ones_active = np.ones((K_active.shape[0], 1))

		top_left = K_active.T @ K_active + regularization * K
		top_right = K_active.T @ ones_active
		bottom_left = top_right.T
		bottom_right = np.array([[float(K_active.shape[0])]])
		system = np.block([
			[top_left, top_right],
			[bottom_left, bottom_right],
		])
		rhs = np.concatenate([K_active.T @ adjusted_targets, np.array([np.sum(adjusted_targets)])])
		solution = np.linalg.solve(system, rhs)
		new_beta = solution[:-1]
		new_bias = float(solution[-1])

		if (
			previous_active is not None
			and np.array_equal(active, previous_active)
			and np.allclose(new_beta, beta, atol=tol, rtol=0.0)
			and abs(new_bias - bias) <= tol
		):
			beta = new_beta
			bias = new_bias
			break

		beta = new_beta
		bias = new_bias
		previous_active = active.copy()

	return {
		"X_train": X,
		"beta": beta,
		"bias": bias,
		"kernel": kernel,
		"gamma": gamma,
	}


def _fit_kernel_regressor(X, y, C, epsilon, kernel, gamma, tol, max_iter=100):
	model = _fit_epsilon_insensitive_kernel_regressor(
		X,
		y,
		C,
		epsilon,
		kernel,
		gamma,
		tol,
		max_iter=max_iter,
	)

	train_predictions = _decision_function(model, X)
	residual = np.abs(y - train_predictions)
	support_mask = (np.abs(model["beta"]) > tol) | (residual >= max(epsilon - tol, 0.0))
	support_indices = np.flatnonzero(support_mask)
	if support_indices.size == 0:
		support_indices = np.arange(X.shape[0], dtype=int)

	model["support_indices"] = support_indices
	return model


def _decision_function(model, X):
	K = _compute_kernel(X, model["X_train"], kernel=model["kernel"], gamma=model["gamma"])
	return K @ model["beta"] + model["bias"]


class SVC(BaseModel, ClassifierMixin):
	"""Support Vector Classifier with linear and RBF kernels."""

	def __init__(self, C=1.0, kernel="rbf", gamma="scale", tol=1e-3, max_iter=1000, random_state=None):
		self.C = C
		self.kernel = kernel
		self.gamma = gamma
		self.tol = tol
		self.max_iter = max_iter
		self.random_state = random_state

	def _validate_params(self):
		self.C_ = _validate_positive_number(self.C, "C")

		if not isinstance(self.kernel, str):
			raise ValueError("kernel must be one of {'linear', 'rbf'}.")
		kernel = self.kernel.lower()
		if kernel not in _ALLOWED_KERNELS:
			raise ValueError("kernel must be one of {'linear', 'rbf'}.")
		self.kernel_ = kernel

		if isinstance(self.gamma, str):
			gamma = self.gamma.lower()
			if gamma not in _ALLOWED_GAMMA_STRINGS:
				raise ValueError("gamma must be one of {'scale', 'auto'} or a positive number.")
			self.gamma_ = gamma
		else:
			self.gamma_ = _validate_positive_number(self.gamma, "gamma")

		self.tol_ = _validate_positive_number(self.tol, "tol")
		# Validate for API compatibility; these parameters are not currently used
		# by the underlying training implementation.
		_validate_positive_integer(self.max_iter, "max_iter")
		_validate_random_state(self.random_state)

	def _validate_X_y(self, X, y=None):
		X = np.asarray(X, dtype=float)
		if X.ndim != 2:
			raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

		if y is None:
			return X, None

		y = np.asarray(y)
		if y.ndim != 1:
			raise ValueError("y must be a 1D array of shape (n_samples,).")
		if X.shape[0] != y.shape[0]:
			raise ValueError("X and y must have the same number of samples.")

		return X, y

	def fit(self, X, y):
		"""Fit the SVC model."""
		self._validate_params()
		X, y = self._validate_X_y(X, y)

		if X.shape[0] == 0:
			raise ValueError("X must contain at least one sample.")

		self.n_features_in_ = X.shape[1]
		self.classes_ = np.unique(y)
		if self.classes_.size < 2:
			raise ValueError("y must contain at least two classes.")

		gamma_value = _resolve_gamma(self.gamma_, X)

		if self.classes_.size == 2:
			positive_class = self.classes_[1]
			y_binary = np.where(y == positive_class, 1.0, -1.0)
			model = _fit_binary_kernel_classifier(
				X,
				y_binary,
				C=self.C_,
				kernel=self.kernel_,
				gamma=gamma_value,
				tol=self.tol_,
			)
			self.models_ = [model]
		else:
			models = []
			K_precomputed = _compute_kernel(X, X, kernel=self.kernel_, gamma=gamma_value)
			for cls in self.classes_:
				y_binary = np.where(y == cls, 1.0, -1.0)
				model = _fit_binary_kernel_classifier(
					X,
					y_binary,
					C=self.C_,
					kernel=self.kernel_,
					gamma=gamma_value,
					tol=self.tol_,
					K=K_precomputed,
				)
				models.append(model)
			self.models_ = models

		support_union = np.unique(np.concatenate([m["support_indices"] for m in self.models_]))
		self.support_ = support_union.astype(int)
		self.support_vectors_ = X[self.support_]
		self.gamma_value_ = gamma_value

		if self.classes_.size == 2:
			self.intercept_ = float(self.models_[0]["bias"])
			self.dual_coef_ = self.models_[0]["beta"][self.support_].reshape(1, -1)
			if self.kernel_ == "linear":
				self.coef_ = np.asarray(self.models_[0]["coef"], dtype=float)
		else:
			self.intercept_ = np.asarray([m["bias"] for m in self.models_], dtype=float)
			if self.kernel_ == "linear":
				self.coef_ = np.vstack([m["coef"] for m in self.models_])

		return self

	def decision_function(self, X):
		"""Compute signed decision scores for samples in X."""
		if not hasattr(self, "models_"):
			raise ValueError("This SVC instance is not fitted yet. Call 'fit' first.")

		X, _ = self._validate_X_y(X, None)
		if X.shape[1] != self.n_features_in_:
			raise ValueError(
				f"X has {X.shape[1]} features, but SVC was fitted with "
				f"{self.n_features_in_} features."
			)

		if self.classes_.size == 2:
			return _decision_function(self.models_[0], X)

		# Precompute the kernel matrix once and reuse it across all OvR models.
		ref = self.models_[0]
		K = _compute_kernel(X, ref["X_train"], kernel=ref["kernel"], gamma=ref["gamma"])
		betas = np.column_stack([m["beta"] for m in self.models_])
		biases = np.array([m["bias"] for m in self.models_], dtype=float)
		return K @ betas + biases

	def predict(self, X):
		"""Predict class labels for samples in X."""
		scores = self.decision_function(X)

		if self.classes_.size == 2:
			pred = np.where(scores >= 0.0, self.classes_[1], self.classes_[0])
			return np.asarray(pred, dtype=self.classes_.dtype)

		class_indices = np.argmax(scores, axis=1)
		return self.classes_[class_indices]


class SVR(BaseModel, RegressorMixin):
	"""Support Vector Regressor with linear and RBF kernels."""

	def __init__(
		self,
		C=1.0,
		epsilon=0.1,
		kernel="rbf",
		gamma="scale",
		tol=1e-3,
		max_iter=1000,
		random_state=None,
	):
		self.C = C
		self.epsilon = epsilon
		self.kernel = kernel
		self.gamma = gamma
		self.tol = tol
		self.max_iter = max_iter
		self.random_state = random_state

	def _validate_params(self):
		self.C_ = _validate_positive_number(self.C, "C")
		self.epsilon_ = _validate_non_negative_number(self.epsilon, "epsilon")

		if not isinstance(self.kernel, str):
			raise ValueError("kernel must be one of {'linear', 'rbf'}.")
		kernel = self.kernel.lower()
		if kernel not in _ALLOWED_KERNELS:
			raise ValueError("kernel must be one of {'linear', 'rbf'}.")
		self.kernel_ = kernel

		if isinstance(self.gamma, str):
			gamma = self.gamma.lower()
			if gamma not in _ALLOWED_GAMMA_STRINGS:
				raise ValueError("gamma must be one of {'scale', 'auto'} or a positive number.")
			self.gamma_ = gamma
		else:
			self.gamma_ = _validate_positive_number(self.gamma, "gamma")

		self.tol_ = _validate_positive_number(self.tol, "tol")
		self.max_iter_ = _validate_positive_integer(self.max_iter, "max_iter")
		# Validate for API compatibility; random_state is not currently used
		# by the underlying training implementation.
		_validate_random_state(self.random_state)

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
		"""Fit the SVR model."""
		self._validate_params()
		X, y = self._validate_X_y(X, y)

		if X.shape[0] == 0:
			raise ValueError("X must contain at least one sample.")

		self.n_features_in_ = X.shape[1]
		gamma_value = _resolve_gamma(self.gamma_, X)

		model = _fit_kernel_regressor(
			X,
			y,
			C=self.C_,
			epsilon=self.epsilon_,
			kernel=self.kernel_,
			gamma=gamma_value,
			tol=self.tol_,
			max_iter=self.max_iter_,
		)

		self.X_train_ = model["X_train"]
		self._beta_ = model["beta"]
		self.intercept_ = float(model["bias"])
		self.support_ = model["support_indices"].astype(int)
		self.support_vectors_ = self.X_train_[self.support_]
		self.dual_coef_ = self._beta_[self.support_].reshape(1, -1)
		self.gamma_value_ = gamma_value

		if self.kernel_ == "linear":
			self.coef_ = np.asarray(model["coef"], dtype=float)

		return self

	def predict(self, X):
		"""Predict regression targets for samples in X."""
		if not hasattr(self, "_beta_") or not hasattr(self, "X_train_"):
			raise ValueError("This SVR instance is not fitted yet. Call 'fit' first.")

		X, _ = self._validate_X_y(X, None)
		if X.shape[1] != self.n_features_in_:
			raise ValueError(
				f"X has {X.shape[1]} features, but SVR was fitted with "
				f"{self.n_features_in_} features."
			)

		K = _compute_kernel(X, self.X_train_, kernel=self.kernel_, gamma=self.gamma_value_)
		return K @ self._beta_ + self.intercept_
