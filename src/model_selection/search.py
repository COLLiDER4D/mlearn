import numpy as np
import math
from model_selection.cross_validation import cross_val_score


class ParameterGrid:
    """ParameterGrid is a utility class that generates all combinations of parameters from a given parameter grid.
    It is used in hyperparameter tuning to create a grid of parameters for exhaustive search.
    
    Parameters
    ----------
    param_grid : dict of lists
        The parameter grid to generate combinations for.
    """
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def __iter__(self):
        """Generates all combinations of parameters from the parameter grid."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        for combination in self._product(*values):
            yield dict(zip(keys, combination))

    def _product(self, *args):
        """Generates the Cartesian product of input iterables."""
        pools = [tuple(pool) for pool in args]
        result = [[]]
        for pool in pools:
            result = [x + [y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)

class GridSearchCV:
    """GridSearchCV is a class that performs exhaustive search over specified parameter values for an estimator.
    It uses cross-validation to evaluate the performance of each combination of parameters.
    
    Parameters
    ----------
    estimator : object
        The estimator for which the hyperparameters are to be tuned.
    param_grid : dict of lists
        The parameter grid to search over.
    cv : int, default=5
        The number of folds in cross-validation.
    """
    def __init__(self, estimator, param_grid, cv=5):
        self.estimator = estimator
        self.param_grid = ParameterGrid(param_grid)
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = -math.inf

    def fit(self, X, y):
        """Performs grid search to find the best parameters for the estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        """
        for params in self.param_grid:
            self.estimator.set_params(**params)
            scores = cross_val_score(self.estimator, X, y, cv=self.cv)
            mean_score = np.mean(scores)
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
        self.estimator.set_params(**self.best_params_)