"""Clustering estimators available in :mod:`micromlkit.cluster`.

Examples
--------
>>> import numpy as np
>>> from micromlkit.cluster import KMeans
>>> X = np.array([[0.0, 0.0], [0.0, 1.0], [3.0, 3.0], [3.0, 4.0]])
>>> labels = KMeans(n_clusters=2, random_state=0).fit_predict(X)
>>> labels.shape
(4,)
"""

from .agglomerative import AgglomerativeClustering
from .dbscan import DBSCAN
from .kmeans import KMeans

__all__ = [
	"KMeans",
	"DBSCAN",
	"AgglomerativeClustering",
]
