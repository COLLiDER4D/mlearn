import numpy as np

from ..base import BaseModel, ClassifierMixin, RegressorMixin


_ALLOWED_KERNELS = {"linear", "rbf"}
_ALLOWED_GAMMA_STRINGS = {"scale", "auto"}


def _validate_positive_number(value, name):
	if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
		raise ValueError(f"{name} must be a positive number.")
	value = float(value)
	if value <= 0.0:
		raise ValueError(f"{name} must be a positive number.")
	return value


def _validate_non_negative_number(value, name):
	if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
		raise ValueError(f"{name} must be a non-negative number.")
	value = float(value)
	if value < 0.0:
		raise ValueError(f"{name} must be a non-negative number.")
	return value


def _validate_positive_integer(value, name):
	if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
		raise ValueError(f"{name} must be a positive integer.")
	value = int(value)
	if value <= 0:
		raise ValueError(f"{name} must be a positive integer.")
	return value


def _validate_random_state(random_state):
	if random_state is None:
		return None
	if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
		raise ValueError("random_state must be an integer or None.")
	return int(random_state)


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


def _linear_kernel(X, Y):
	return X @ Y.T


def _rbf_kernel(X, Y, gamma):
	X_norm = np.sum(X ** 2, axis=1, keepdims=True)
	Y_norm = np.sum(Y ** 2, axis=1, keepdims=True).T
	squared_distances = np.maximum(X_norm + Y_norm - 2.0 * (X @ Y.T), 0.0)
	return np.exp(-gamma * squared_distances)


def _compute_kernel(X, Y, kernel, gamma):
	if kernel == "linear":
		return _linear_kernel(X, Y)
	return _rbf_kernel(X, Y, gamma)


def _fit_binary_kernel_classifier(X, y_binary, C, kernel, gamma, tol):
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


def _fit_kernel_regressor(X, y, C, epsilon, kernel, gamma, tol):
	if kernel == "linear":
		regularization = 1.0 / C
		X_mean = np.mean(X, axis=0)
		y_mean = float(np.mean(y))
		X_centered = X - X_mean
		y_centered = y - y_mean

		system = X_centered.T @ X_centered + regularization * np.eye(X.shape[1])
		coef = np.linalg.solve(system, X_centered.T @ y_centered)
		bias = float(y_mean - X_mean @ coef)

		# Recover a sample-space coefficient vector for consistent SVM-style attributes.
		beta = np.linalg.lstsq(X.T, coef, rcond=None)[0]

		train_predictions = X @ coef + bias
		residual = np.abs(y - train_predictions)
		support_mask = (np.abs(beta) > tol) | (residual >= max(epsilon - tol, 0.0))
		support_indices = np.flatnonzero(support_mask)
		if support_indices.size == 0:
			support_indices = np.arange(X.shape[0], dtype=int)

		return {
			"X_train": X,
			"beta": beta,
			"bias": bias,
			"support_indices": support_indices,
			"kernel": kernel,
			"gamma": gamma,
			"coef": coef,
		}

	K = _compute_kernel(X, X, kernel=kernel, gamma=gamma)
	regularization = 1.0 / C
	system = K + regularization * np.eye(K.shape[0])
	beta = np.linalg.solve(system, y)
	bias = float(np.mean(y - K @ beta))

	train_predictions = K @ beta + bias
	residual = np.abs(y - train_predictions)
	support_mask = (np.abs(beta) > tol) | (residual >= max(epsilon - tol, 0.0))
	support_indices = np.flatnonzero(support_mask)
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
		self.max_iter_ = _validate_positive_integer(self.max_iter, "max_iter")
		self.random_state_ = _validate_random_state(self.random_state)

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
			for cls in self.classes_:
				y_binary = np.where(y == cls, 1.0, -1.0)
				model = _fit_binary_kernel_classifier(
					X,
					y_binary,
					C=self.C_,
					kernel=self.kernel_,
					gamma=gamma_value,
					tol=self.tol_,
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

		scores = np.column_stack([_decision_function(model, X) for model in self.models_])
		return scores

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
		self.random_state_ = _validate_random_state(self.random_state)

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
