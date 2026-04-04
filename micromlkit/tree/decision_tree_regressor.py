import numpy as np

from ..base import BaseModel, RegressorMixin


class DecisionTreeRegressor(BaseModel, RegressorMixin):
	"""A simple CART-style decision tree regressor."""

	def __init__(
		self,
		max_depth=None,
		criterion="mse",
		min_samples_split=2,
		min_samples_leaf=1,
		random_state=None,
	):
		self.max_depth = max_depth
		self.criterion = criterion
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.random_state = random_state
		self._rng = None

	def _validate_params(self):
		if self.max_depth is not None:
			if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, (int, np.integer)):
				raise ValueError("max_depth must be a positive integer or None.")
			if self.max_depth <= 0:
				raise ValueError("max_depth must be a positive integer or None.")

		if isinstance(self.criterion, str):
			criterion = self.criterion.lower()
		else:
			raise ValueError("criterion must be one of {'mse', 'mae'}.")
		if criterion not in {"mse", "mae"}:
			raise ValueError("criterion must be one of {'mse', 'mae'}.")
		self.criterion_ = criterion

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

	def _impurity(self, y):
		if self.criterion_ == "mse":
			center = float(np.mean(y))
			return float(np.mean((y - center) ** 2))

		center = float(np.median(y))
		return float(np.mean(np.abs(y - center)))

	def _leaf_value(self, y):
		if self.criterion_ == "mse":
			return float(np.mean(y))
		return float(np.median(y))

	def _best_split(self, X, y, parent_impurity):
		n_samples, n_features = X.shape
		best_gain = 0.0
		best_feature = None
		best_threshold = None
		best_left_mask = None

		feature_indices = self._rng.permutation(n_features) if self.random_state is not None else range(n_features)

		for feature_index in feature_indices:
			feature_values = X[:, feature_index]
			unique_values = np.unique(feature_values)
			if unique_values.size <= 1:
				continue

			thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
			for threshold in thresholds:
				left_mask = feature_values <= threshold
				n_left = int(np.sum(left_mask))
				n_right = n_samples - n_left

				if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
					continue

				y_left = y[left_mask]
				y_right = y[~left_mask]

				impurity_left = self._impurity(y_left)
				impurity_right = self._impurity(y_right)
				weighted_impurity = (n_left / n_samples) * impurity_left + (n_right / n_samples) * impurity_right
				gain = parent_impurity - weighted_impurity

				if gain > best_gain + 1e-12:
					best_gain = gain
					best_feature = feature_index
					best_threshold = float(threshold)
					best_left_mask = left_mask
				elif abs(gain - best_gain) <= 1e-12 and best_feature is not None:
					if feature_index < best_feature or (
						feature_index == best_feature and float(threshold) < float(best_threshold)
					):
						best_feature = feature_index
						best_threshold = float(threshold)
						best_left_mask = left_mask

		return best_feature, best_threshold, best_left_mask

	def _build_tree(self, X, y, depth):
		root = {}
		node_count = 0
		max_depth_seen = depth
		stack = [(root, X, y, depth)]

		while stack:
			node, X_node, y_node, node_depth = stack.pop()
			n_samples = X_node.shape[0]
			node_count += 1
			max_depth_seen = max(max_depth_seen, node_depth)

			should_stop = (
				(self.max_depth is not None and node_depth >= self.max_depth)
				or (n_samples < self.min_samples_split)
				or np.allclose(y_node, y_node[0])
			)

			if should_stop:
				node["is_leaf"] = True
				node["value"] = self._leaf_value(y_node)
				continue

			parent_impurity = self._impurity(y_node)
			feature_index, threshold, left_mask = self._best_split(X_node, y_node, parent_impurity)

			if feature_index is None:
				node["is_leaf"] = True
				node["value"] = self._leaf_value(y_node)
				continue

			left_node = {}
			right_node = {}
			node["is_leaf"] = False
			node["feature_index"] = int(feature_index)
			node["threshold"] = float(threshold)
			node["left"] = left_node
			node["right"] = right_node

			stack.append((right_node, X_node[~left_mask], y_node[~left_mask], node_depth + 1))
			stack.append((left_node, X_node[left_mask], y_node[left_mask], node_depth + 1))

		return root, node_count, max_depth_seen

	def fit(self, X, y):
		"""Fit the decision tree regressor."""
		self._validate_params()
		X, y = self._validate_X_y(X, y)

		if X.shape[0] == 0:
			raise ValueError("X must contain at least one sample.")

		self._rng = np.random.default_rng(self.random_state)
		self.n_features_in_ = X.shape[1]
		self.tree_, self.n_nodes_, self.depth_ = self._build_tree(X, y, depth=0)
		return self

	def _predict_one(self, x, node):
		current = node
		while not current["is_leaf"]:
			if x[current["feature_index"]] <= current["threshold"]:
				current = current["left"]
			else:
				current = current["right"]
		return float(current["value"])

	def predict(self, X):
		"""Predict target values for samples in X."""
		if not hasattr(self, "tree_"):
			raise ValueError("This DecisionTreeRegressor instance is not fitted yet. Call 'fit' first.")

		X, _ = self._validate_X_y(X, None)
		if X.shape[1] != self.n_features_in_:
			raise ValueError(
				f"X has {X.shape[1]} features, but DecisionTreeRegressor was fitted with "
				f"{self.n_features_in_} features."
			)

		predictions = [self._predict_one(sample, self.tree_) for sample in X]
		return np.asarray(predictions, dtype=float)
