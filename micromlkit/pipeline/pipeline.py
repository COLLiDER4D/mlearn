from ..base import BaseEstimator


class Pipeline(BaseEstimator):
	"""A simple transformer-estimator pipeline.

	Parameters
	----------
	steps : list[tuple[str, object]]
		Ordered sequence of ``(name, step)`` pairs.
		All intermediate steps must provide ``fit_transform`` and ``transform``.
		The final step must provide ``fit`` and ``predict``.
	"""

	def __init__(self, steps):
		validated_steps = self._validate_steps(steps)
		self.steps = validated_steps
		self._refresh_named_steps()

	def _refresh_named_steps(self):
		self.named_steps = {name: step for name, step in self.steps}

	def _validate_steps(self, steps):
		if not isinstance(steps, (list, tuple)):
			raise ValueError("'steps' must be a list or tuple of (name, step) pairs.")
		if len(steps) == 0:
			raise ValueError("'steps' must contain at least one step.")

		validated = []
		seen_names = set()

		for idx, item in enumerate(steps):
			if not isinstance(item, tuple) or len(item) != 2:
				raise ValueError(
					"Each step must be a (name, step) tuple with exactly 2 elements."
				)

			name, step = item
			if not isinstance(name, str) or not name:
				raise ValueError("Each step name must be a non-empty string.")
			if name == "steps":
				raise ValueError(
					"Step name 'steps' is reserved by the Pipeline and cannot be used."
				)
			if "__" in name:
				raise ValueError(
					f"Step name '{name}' must not contain '__', as it conflicts with "
					"the nested parameter convention (name__param)."
				)
			if name in seen_names:
				raise ValueError(f"Step names must be unique. Duplicate name: '{name}'.")
			seen_names.add(name)

			if idx < len(steps) - 1:
				self._require_methods(step, ("fit_transform", "transform"), name)
			else:
				self._require_methods(step, ("fit", "predict"), name)

			validated.append((name, step))

		return list(validated)

	def _require_methods(self, step, methods, step_name):
		for method in methods:
			attr = getattr(step, method, None)
			if attr is None or not callable(attr):
				raise ValueError(
					f"Step '{step_name}' must define a callable '{method}' method."
				)

	def _transform_intermediate(self, X):
		for _, step in self.steps[:-1]:
			X = step.transform(X)
		return X

	def fit(self, X, y=None):
		"""Fit all transformers, then fit the final estimator."""
		for _, step in self.steps[:-1]:
			X = step.fit_transform(X, y)

		self.steps[-1][1].fit(X, y)
		self.fitted_ = True
		return self

	def predict(self, X):
		"""Transform X through intermediate steps and predict with the final step."""
		if not hasattr(self, "fitted_"):
			raise ValueError("This Pipeline instance is not fitted yet. Call 'fit' first.")

		X = self._transform_intermediate(X)
		return self.steps[-1][1].predict(X)

	def transform(self, X):
		"""Transform X through the full pipeline when final step supports transform."""
		if not hasattr(self, "fitted_"):
			raise ValueError("This Pipeline instance is not fitted yet. Call 'fit' first.")

		X = self._transform_intermediate(X)
		final_step = self.steps[-1][1]
		if not hasattr(final_step, "transform") or not callable(getattr(final_step, "transform")):
			raise ValueError("Final step does not support 'transform'.")
		return final_step.transform(X)

	def fit_transform(self, X, y=None):
		"""Fit the pipeline and return transformed output if supported."""
		for _, step in self.steps[:-1]:
			X = step.fit_transform(X, y)

		final_step = self.steps[-1][1]
		if hasattr(final_step, "fit_transform") and callable(getattr(final_step, "fit_transform")):
			X_out = final_step.fit_transform(X, y)
			self.fitted_ = True
			return X_out

		final_step.fit(X, y)
		self.fitted_ = True
		if hasattr(final_step, "transform") and callable(getattr(final_step, "transform")):
			return final_step.transform(X)

		raise ValueError(
			"Final step does not support 'fit_transform' or 'transform'. "
			"Use 'fit' followed by 'predict' instead."
		)

	def get_params(self, deep=True):
		"""Return parameters for this pipeline.

		When ``deep=True``, include nested step parameters under ``name__param`` keys.
		"""
		params = {"steps": self.steps}
		if not deep:
			return params

		for name, step in self.steps:
			params[name] = step
			if hasattr(step, "get_params") and callable(getattr(step, "get_params")):
				for key, value in step.get_params().items():
					params[f"{name}__{key}"] = value

		return params

	def set_params(self, **params):
		"""Set parameters for this pipeline and nested step estimators."""
		if not params:
			return self

		step_names = [name for name, _ in self.steps]

		if "steps" in params:
			new_steps = self._validate_steps(params.pop("steps"))
			self.steps = new_steps
			self._refresh_named_steps()
			step_names = [name for name, _ in self.steps]

		for key, value in params.items():
			if "__" in key:
				step_name, sub_key = key.split("__", 1)
				if step_name not in self.named_steps:
					raise ValueError(
						f"Invalid step '{step_name}' for Pipeline. "
						f"Valid steps are: {step_names}."
					)
				step = self.named_steps[step_name]
				if not hasattr(step, "set_params") or not callable(getattr(step, "set_params")):
					raise ValueError(
						f"Step '{step_name}' does not support nested parameter setting."
					)
				step.set_params(**{sub_key: value})
				continue

			if key in self.named_steps:
				for idx, (name, _) in enumerate(self.steps):
					if name == key:
						self.steps[idx] = (name, value)
						break
				self.steps = self._validate_steps(self.steps)
				self._refresh_named_steps()
				continue

			raise ValueError(
				f"Invalid parameter '{key}' for Pipeline. "
				f"Valid parameters are: {sorted(self.get_params(deep=True).keys())}."
			)

		return self
