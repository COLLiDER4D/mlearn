"""Decision-tree estimators in :mod:`micromlkit.tree`.

Examples
--------
>>> import numpy as np
>>> from micromlkit.tree import DecisionTreeRegressor
>>> X = np.array([[0.0], [1.0], [2.0]])
>>> y = np.array([0.0, 1.0, 2.0])
>>> reg = DecisionTreeRegressor(max_depth=2).fit(X, y)
>>> reg.predict(np.array([[1.5]])).shape
(1,)
"""

from .decision_tree_classifier import DecisionTreeClassifier
from .decision_tree_regressor import DecisionTreeRegressor

__all__ = [
	"DecisionTreeClassifier",
	"DecisionTreeRegressor",
]
