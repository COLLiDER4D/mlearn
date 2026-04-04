"""Classification and regression metrics for model evaluation.

Examples
--------
>>> from micromlkit.metrics import accuracy_score, mean_squared_error
>>> accuracy_score([0, 1, 1], [0, 1, 0])
0.6666666666666666
>>> mean_squared_error([1.0, 2.0], [1.5, 2.5])
0.25
"""

from .classification import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from .regression import mean_absolute_error, mean_squared_error, r2_score


__all__ = [
    "accuracy_score",
    "confusion_matrix",
    "f1_score",
    "precision_score",
    "recall_score",
    "mean_absolute_error",
    "mean_squared_error",
    "r2_score",
]