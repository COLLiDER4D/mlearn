"""Preprocessing transformers in :mod:`micromlkit.preprocessing`.

Examples
--------
>>> import numpy as np
>>> from micromlkit.preprocessing import StandardScaler
>>> scaler = StandardScaler().fit(np.array([[1.0], [3.0], [5.0]]))
>>> scaler.transform(np.array([[3.0]])).shape
(1, 1)
"""

from .encoder import LabelEncoder
from .imputer import SimpleImputer
from .scaler import StandardScaler

__all__ = [
	"StandardScaler",
	"SimpleImputer",
	"LabelEncoder",
]
