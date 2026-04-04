"""Pipeline utilities for chaining transformers and estimators.

Examples
--------
>>> import numpy as np
>>> from micromlkit.pipeline import Pipeline
>>> from micromlkit.preprocessing import StandardScaler
>>> from micromlkit.linear_model import LinearRegression
>>> pipe = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
>>> X = np.array([[1.0], [2.0], [3.0]])
>>> y = np.array([2.0, 4.0, 6.0])
>>> pipe.fit(X, y).predict(np.array([[4.0]])).shape
(1,)
"""

from .pipeline import Pipeline

__all__ = ["Pipeline"]
