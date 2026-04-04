import numpy as np

from ..base import BaseModel, ClassifierMixin, RegressorMixin
from ..tree import DecisionTreeRegressor


class GradientBoostingRegressor(BaseModel, RegressorMixin):
	"""A simple gradient boosting regressor with decision tree base learners."""

	def __init__(
		self,
		n_estimators=100,
		learning_rate=0.1,
		max_depth=3,
		min_samples_split=2,
		min_samples_leaf=1,
		random_state=None,
	):
		self.n_estimators = n_estimators
		self.learning_rate = learning_rate
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.random_state = random_state

	def _validate_params(self):
		if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, (int, np.integer)):
			raise ValueError("n_estimators must be a positive integer.")
		if self.n_estimators <= 0:
			raise ValueError("n_estimators must be a positive integer.")

		if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float, np.floating)):
			raise ValueError("learning_rate must be a positive number.")
		if float(self.learning_rate) <= 0.0:
			raise ValueError("learning_rate must be a positive number.")

		if self.max_depth is not None:
			if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, (int, np.integer)):
				raise ValueError("max_depth must be a positive integer or None.")
			if self.max_depth <= 0:
				raise ValueError("max_depth must be a positive integer or None.")

		if isinstance(self.min_samples_split, bool) or not isinstance(self.min_samples_split, (int, np.integer)):
			raise ValueError("min_samples_split must be an integer greater than or equal to 2.")
		if self.min_samples_split < 2:
			raise ValueError("min_samples_split must be an integer greater than or equal to 2.")

		if isinstance(self.min_samples_leaf, bool) or not isinstance(self.min_samples_leaf, (int, np.integer)):
			raise ValueError("min_samples_leaf must be a positive integer.")
		if self.min_samples_leaf < 1:
			raise ValueError("min_samples_leaf must be a positive integer.")

		if self.random_state is not None:
			if isinstance(self.random_state, bool) or not isinstance(self.random_state, (int, np.integer)):
				raise ValueError("random_state must be an integer or None.")

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
		"""Fit the gradient boosting regressor."""
		self._validate_params()
		X, y = self._validate_X_y(X, y)

		n_samples, n_features = X.shape
		if n_samples == 0:
			raise ValueError("X must contain at least one sample.")

		self.n_features_in_ = n_features
		self.init_pred_ = float(np.mean(y))
		self.estimators_ = []

		rng = np.random.default_rng(self.random_state)
		current_pred = np.full(n_samples, self.init_pred_, dtype=float)

		for _ in range(self.n_estimators):
			residual = y - current_pred
			tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))
			estimator = DecisionTreeRegressor(
				max_depth=self.max_depth,
				criterion="mse",
				min_samples_split=self.min_samples_split,
				min_samples_leaf=self.min_samples_leaf,
				random_state=tree_seed,
			)
			estimator.fit(X, residual)
			update = estimator.predict(X)
			current_pred += float(self.learning_rate) * update
			self.estimators_.append(estimator)

		return self

	def predict(self, X):
		"""Predict target values for samples in X."""
		if not hasattr(self, "estimators_"):
			raise ValueError("This GradientBoostingRegressor instance is not fitted yet. Call 'fit' first.")

		X, _ = self._validate_X_y(X, None)
		if X.shape[1] != self.n_features_in_:
			raise ValueError(
				f"X has {X.shape[1]} features, but GradientBoostingRegressor was fitted with "
				f"{self.n_features_in_} features."
			)

		predictions = np.full(X.shape[0], self.init_pred_, dtype=float)
		for estimator in self.estimators_:
			predictions += float(self.learning_rate) * estimator.predict(X)

		return predictions


class GradientBoostingClassifier(BaseModel, ClassifierMixin):
	"""A simple binary gradient boosting classifier with logistic loss."""

	def __init__(
		self,
		n_estimators=100,
		learning_rate=0.1,
		max_depth=3,
		min_samples_split=2,
		min_samples_leaf=1,
		random_state=None,
	):
		self.n_estimators = n_estimators
		self.learning_rate = learning_rate
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.random_state = random_state

	def _validate_params(self):
		if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, (int, np.integer)):
			raise ValueError("n_estimators must be a positive integer.")
		if self.n_estimators <= 0:
			raise ValueError("n_estimators must be a positive integer.")

		if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float, np.floating)):
			raise ValueError("learning_rate must be a positive number.")
		if float(self.learning_rate) <= 0.0:
			raise ValueError("learning_rate must be a positive number.")

		if self.max_depth is not None:
			if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, (int, np.integer)):
				raise ValueError("max_depth must be a positive integer or None.")
			if self.max_depth <= 0:
				raise ValueError("max_depth must be a positive integer or None.")

		if isinstance(self.min_samples_split, bool) or not isinstance(self.min_samples_split, (int, np.integer)):
			raise ValueError("min_samples_split must be an integer greater than or equal to 2.")
		if self.min_samples_split < 2:
			raise ValueError("min_samples_split must be an integer greater than or equal to 2.")

		if isinstance(self.min_samples_leaf, bool) or not isinstance(self.min_samples_leaf, (int, np.integer)):
			raise ValueError("min_samples_leaf must be a positive integer.")
		if self.min_samples_leaf < 1:
			raise ValueError("min_samples_leaf must be a positive integer.")

		if self.random_state is not None:
			if isinstance(self.random_state, bool) or not isinstance(self.random_state, (int, np.integer)):
				raise ValueError("random_state must be an integer or None.")

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

	def _sigmoid(self, z):
		z = np.clip(z, -500.0, 500.0)
		return 1.0 / (1.0 + np.exp(-z))

	def fit(self, X, y):
		"""Fit the gradient boosting classifier (binary classification only)."""
		self._validate_params()
		X, y = self._validate_X_y(X, y)

		n_samples, n_features = X.shape
		if n_samples == 0:
			raise ValueError("X must contain at least one sample.")

		self.classes_ = np.unique(y)
		if self.classes_.shape[0] != 2:
			raise ValueError("GradientBoostingClassifier supports binary classification with exactly 2 classes.")

		y_binary = (y == self.classes_[1]).astype(float)
		positive_rate = float(np.clip(np.mean(y_binary), 1e-12, 1.0 - 1e-12))

		self.n_features_in_ = n_features
		self.init_pred_ = float(np.log(positive_rate / (1.0 - positive_rate)))
		self.estimators_ = []

		rng = np.random.default_rng(self.random_state)
		logits = np.full(n_samples, self.init_pred_, dtype=float)

		for _ in range(self.n_estimators):
			probabilities = self._sigmoid(logits)
			residual = y_binary - probabilities

			tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))
			estimator = DecisionTreeRegressor(
				max_depth=self.max_depth,
				criterion="mse",
				min_samples_split=self.min_samples_split,
				min_samples_leaf=self.min_samples_leaf,
				random_state=tree_seed,
			)
			estimator.fit(X, residual)
			update = estimator.predict(X)
			logits += float(self.learning_rate) * update
			self.estimators_.append(estimator)

		return self

	def decision_function(self, X):
		"""Compute raw additive scores before sigmoid transformation."""
		if not hasattr(self, "estimators_"):
			raise ValueError("This GradientBoostingClassifier instance is not fitted yet. Call 'fit' first.")

		X, _ = self._validate_X_y(X, None)
		if X.shape[1] != self.n_features_in_:
			raise ValueError(
				f"X has {X.shape[1]} features, but GradientBoostingClassifier was fitted with "
				f"{self.n_features_in_} features."
			)

		scores = np.full(X.shape[0], self.init_pred_, dtype=float)
		for estimator in self.estimators_:
			scores += float(self.learning_rate) * estimator.predict(X)

		return scores

	def predict_proba(self, X):
		"""Predict class probabilities for samples in X."""
		scores = self.decision_function(X)
		positive_prob = self._sigmoid(scores)
		negative_prob = 1.0 - positive_prob
		return np.column_stack((negative_prob, positive_prob))

	def predict(self, X):
		"""Predict class labels for samples in X."""
		probabilities = self.predict_proba(X)[:, 1]
		predicted_positive = probabilities >= 0.5
		return np.where(predicted_positive, self.classes_[1], self.classes_[0])
