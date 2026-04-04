import numpy as np

from ..base import BaseModel, ClassifierMixin, RegressorMixin
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier(BaseModel, ClassifierMixin):
	"""A simple random forest classifier using decision trees as base estimators."""

	def __init__(
		self,
		n_estimators=100,
		max_depth=None,
		min_samples_split=2,
		min_samples_leaf=1,
		random_state=None,
		bootstrap=True,
		max_features="sqrt",
	):
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.random_state = random_state
		self.bootstrap = bootstrap
		self.max_features = max_features

	def _validate_params(self):
		if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, (int, np.integer)):
			raise ValueError("n_estimators must be a positive integer.")
		if self.n_estimators <= 0:
			raise ValueError("n_estimators must be a positive integer.")

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

		if not isinstance(self.bootstrap, bool):
			raise ValueError("bootstrap must be a boolean.")

		if self.max_features is not None:
			if isinstance(self.max_features, bool):
				raise ValueError(
					"max_features must be one of {'sqrt', 'log2', 'auto'}, a positive integer, "
					"a float in (0, 1], or None."
				)
			if isinstance(self.max_features, str):
				if self.max_features.lower() not in {"sqrt", "log2", "auto"}:
					raise ValueError(
						"max_features must be one of {'sqrt', 'log2', 'auto'}, a positive integer, "
						"a float in (0, 1], or None."
					)
			elif isinstance(self.max_features, (int, np.integer)):
				if self.max_features <= 0:
					raise ValueError("max_features must be a positive integer when provided as an integer.")
			elif isinstance(self.max_features, (float, np.floating)):
				if not (0.0 < float(self.max_features) <= 1.0):
					raise ValueError("max_features must be in (0, 1] when provided as a float.")
			else:
				raise ValueError(
					"max_features must be one of {'sqrt', 'log2', 'auto'}, a positive integer, "
					"a float in (0, 1], or None."
				)

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

	def _resolve_max_features(self, n_features):
		if self.max_features is None:
			return n_features

		if isinstance(self.max_features, str):
			key = self.max_features.lower()
			if key in {"sqrt", "auto"}:
				return max(1, int(np.sqrt(n_features)))
			return max(1, int(np.log2(n_features)))

		if isinstance(self.max_features, (int, np.integer)):
			if self.max_features > n_features:
				raise ValueError("max_features must be less than or equal to the number of features.")
			return int(self.max_features)

		return max(1, int(np.ceil(float(self.max_features) * n_features)))

	def fit(self, X, y):
		"""Fit the random forest classifier."""
		self._validate_params()
		X, y = self._validate_X_y(X, y)

		n_samples, n_features = X.shape
		if n_samples == 0:
			raise ValueError("X must contain at least one sample.")
		if n_features == 0:
			raise ValueError("X must contain at least one feature.")

		self.n_features_in_ = n_features
		self.classes_ = np.unique(y)
		self.max_features_ = self._resolve_max_features(n_features)

		rng = np.random.default_rng(self.random_state)
		self.estimators_ = []
		self.feature_indices_ = []

		for _ in range(self.n_estimators):
			feature_indices = rng.choice(n_features, size=self.max_features_, replace=False)
			if self.bootstrap:
				bootstrap_indices = rng.integers(0, n_samples, size=n_samples)
			else:
				bootstrap_indices = np.arange(n_samples)

			tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))
			estimator = DecisionTreeClassifier(
				max_depth=self.max_depth,
				criterion="gini",
				min_samples_split=self.min_samples_split,
				min_samples_leaf=self.min_samples_leaf,
				random_state=tree_seed,
			)
			estimator.fit(X[bootstrap_indices][:, feature_indices], y[bootstrap_indices])

			self.estimators_.append(estimator)
			self.feature_indices_.append(feature_indices)

		return self

	def predict(self, X):
		"""Predict class labels for samples in X."""
		if not hasattr(self, "estimators_"):
			raise ValueError("This RandomForestClassifier instance is not fitted yet. Call 'fit' first.")

		X, _ = self._validate_X_y(X, None)
		if X.shape[1] != self.n_features_in_:
			raise ValueError(
				f"X has {X.shape[1]} features, but RandomForestClassifier was fitted with "
				f"{self.n_features_in_} features."
			)

		n_samples = X.shape[0]
		n_classes = self.classes_.shape[0]
		votes = np.zeros((n_samples, n_classes), dtype=int)

		for estimator, feature_indices in zip(self.estimators_, self.feature_indices_):
			estimator_predictions = estimator.predict(X[:, feature_indices])
			encoded = np.searchsorted(self.classes_, estimator_predictions)
			votes[np.arange(n_samples), encoded] += 1

		winner_indices = np.argmax(votes, axis=1)
		return self.classes_[winner_indices]


class RandomForestRegressor(BaseModel, RegressorMixin):
	"""A simple random forest regressor using decision trees as base estimators."""

	def __init__(
		self,
		n_estimators=100,
		max_depth=None,
		min_samples_split=2,
		min_samples_leaf=1,
		random_state=None,
		bootstrap=True,
		max_features=1.0,
	):
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.random_state = random_state
		self.bootstrap = bootstrap
		self.max_features = max_features

	def _validate_params(self):
		if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, (int, np.integer)):
			raise ValueError("n_estimators must be a positive integer.")
		if self.n_estimators <= 0:
			raise ValueError("n_estimators must be a positive integer.")

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

		if not isinstance(self.bootstrap, bool):
			raise ValueError("bootstrap must be a boolean.")

		if self.max_features is not None:
			if isinstance(self.max_features, bool):
				raise ValueError(
					"max_features must be one of {'sqrt', 'log2', 'auto'}, a positive integer, "
					"a float in (0, 1], or None."
				)
			if isinstance(self.max_features, str):
				if self.max_features.lower() not in {"sqrt", "log2", "auto"}:
					raise ValueError(
						"max_features must be one of {'sqrt', 'log2', 'auto'}, a positive integer, "
						"a float in (0, 1], or None."
					)
			elif isinstance(self.max_features, (int, np.integer)):
				if self.max_features <= 0:
					raise ValueError("max_features must be a positive integer when provided as an integer.")
			elif isinstance(self.max_features, (float, np.floating)):
				if not (0.0 < float(self.max_features) <= 1.0):
					raise ValueError("max_features must be in (0, 1] when provided as a float.")
			else:
				raise ValueError(
					"max_features must be one of {'sqrt', 'log2', 'auto'}, a positive integer, "
					"a float in (0, 1], or None."
				)

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

	def _resolve_max_features(self, n_features):
		if self.max_features is None:
			return n_features

		if isinstance(self.max_features, str):
			key = self.max_features.lower()
			if key == "sqrt":
				return max(1, int(np.sqrt(n_features)))
			if key == "log2":
				return max(1, int(np.log2(n_features)))
			return n_features

		if isinstance(self.max_features, (int, np.integer)):
			if self.max_features > n_features:
				raise ValueError("max_features must be less than or equal to the number of features.")
			return int(self.max_features)

		return max(1, int(np.ceil(float(self.max_features) * n_features)))

	def fit(self, X, y):
		"""Fit the random forest regressor."""
		self._validate_params()
		X, y = self._validate_X_y(X, y)

		n_samples, n_features = X.shape
		if n_samples == 0:
			raise ValueError("X must contain at least one sample.")
		if n_features == 0:
			raise ValueError("X must contain at least one feature.")

		self.n_features_in_ = n_features
		self.max_features_ = self._resolve_max_features(n_features)

		rng = np.random.default_rng(self.random_state)
		self.estimators_ = []
		self.feature_indices_ = []

		for _ in range(self.n_estimators):
			feature_indices = rng.choice(n_features, size=self.max_features_, replace=False)
			if self.bootstrap:
				bootstrap_indices = rng.integers(0, n_samples, size=n_samples)
			else:
				bootstrap_indices = np.arange(n_samples)

			tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))
			estimator = DecisionTreeRegressor(
				max_depth=self.max_depth,
				criterion="mse",
				min_samples_split=self.min_samples_split,
				min_samples_leaf=self.min_samples_leaf,
				random_state=tree_seed,
			)
			estimator.fit(X[bootstrap_indices][:, feature_indices], y[bootstrap_indices])

			self.estimators_.append(estimator)
			self.feature_indices_.append(feature_indices)

		return self

	def predict(self, X):
		"""Predict target values for samples in X."""
		if not hasattr(self, "estimators_"):
			raise ValueError("This RandomForestRegressor instance is not fitted yet. Call 'fit' first.")

		X, _ = self._validate_X_y(X, None)
		if X.shape[1] != self.n_features_in_:
			raise ValueError(
				f"X has {X.shape[1]} features, but RandomForestRegressor was fitted with "
				f"{self.n_features_in_} features."
			)

		all_predictions = []
		for estimator, feature_indices in zip(self.estimators_, self.feature_indices_):
			all_predictions.append(estimator.predict(X[:, feature_indices]))

		return np.mean(np.asarray(all_predictions, dtype=float), axis=0)
