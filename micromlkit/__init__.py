"""Top-level package for :mod:`micromlkit`.

This package exposes educational machine-learning modules built with NumPy and
a scikit-learn-inspired API.

Examples
--------
>>> from micromlkit import linear_model
>>> model = linear_model.LinearRegression()
>>> isinstance(model.get_params(), dict)
True
"""

from . import linear_model
from . import preprocessing
from . import decomposition
from . import cluster
from . import neighbors
from . import pipeline
from . import tree
from . import svm
from . import ensemble

__all__ = [
	"linear_model",
	"preprocessing",
	"decomposition",
	"cluster",
	"neighbors",
	"pipeline",
	"tree",
	"svm",
	"ensemble",
]
