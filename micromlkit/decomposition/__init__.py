"""Dimensionality-reduction tools in :mod:`micromlkit.decomposition`.

Examples
--------
>>> import numpy as np
>>> from micromlkit.decomposition import PCA
>>> X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
>>> pca = PCA(n_components=1).fit(X)
>>> pca.transform(np.array([[7.0, 8.0]])).shape
(1, 1)
"""

from .pca import PCA

__all__ = [
	"PCA",
]
