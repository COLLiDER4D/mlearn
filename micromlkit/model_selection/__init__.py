"""Model-selection utilities such as splitters and grid search.

Examples
--------
>>> from micromlkit.model_selection import ParameterGrid
>>> grid = list(ParameterGrid({"alpha": [0.1, 1.0]}))
>>> len(grid)
2
"""

from .split import train_test_split, KFold, StratifiedKFold
from .cross_validation import cross_val_score
from .search import ParameterGrid, GridSearchCV

__all__ = [
	"train_test_split",
	"KFold",
	"StratifiedKFold",
	"cross_val_score",
	"ParameterGrid",
	"GridSearchCV",
]
