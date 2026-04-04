"""Linear models available in :mod:`micromlkit.linear_model`.

Examples
--------
>>> import numpy as np
>>> from micromlkit.linear_model import LinearRegression
>>> X = np.array([[1.0], [2.0], [3.0]])
>>> y = np.array([2.0, 4.0, 6.0])
>>> model = LinearRegression().fit(X, y)
>>> model.predict(np.array([[4.0]])).shape
(1,)
"""

from .linear_regression import LinearRegression
from .ridge import Ridge
from .lasso import Lasso
from .logistic_regression import LogisticRegression
__all__ = [
	"LinearRegression",
	"Ridge",
	"Lasso",
	"LogisticRegression"
]
