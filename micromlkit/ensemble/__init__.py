"""Ensemble estimators in :mod:`micromlkit.ensemble`.

Examples
--------
>>> import numpy as np
>>> from micromlkit.ensemble import RandomForestClassifier
>>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
>>> y = np.array([0, 0, 1, 1])
>>> clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
>>> clf.predict(np.array([[1.5]])).shape
(1,)
"""

from .gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor
from .random_forest import RandomForestClassifier, RandomForestRegressor

__all__ = [
	"RandomForestClassifier",
	"RandomForestRegressor",
	"GradientBoostingClassifier",
	"GradientBoostingRegressor",
]
