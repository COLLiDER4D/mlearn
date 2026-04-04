"""Support vector machine estimators in :mod:`micromlkit.svm`.

Examples
--------
>>> import numpy as np
>>> from micromlkit.svm import SVC
>>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
>>> y = np.array([0, 0, 1, 1])
>>> clf = SVC(kernel="linear", random_state=0).fit(X, y)
>>> clf.predict(np.array([[1.5]])).shape
(1,)
"""

from .svm import SVC, SVR

__all__ = [
	"SVC",
	"SVR",
]
