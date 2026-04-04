"""Nearest-neighbor estimators in :mod:`micromlkit.neighbors`.

Examples
--------
>>> import numpy as np
>>> from micromlkit.neighbors import KNNClassifier
>>> X = np.array([[0.0], [1.0], [2.0]])
>>> y = np.array([0, 0, 1])
>>> clf = KNNClassifier(n_neighbors=1).fit(X, y)
>>> clf.predict(np.array([[1.5]])).shape
(1,)
"""

from .knn_classifier import KNNClassifier
from .knn_regressor import KNNRegressor

__all__ = [
	"KNNClassifier",
	"KNNRegressor",
]
