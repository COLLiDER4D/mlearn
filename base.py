from abc import ABC, ABC, abstractmethod


class BaseEstimator():
    """Simple base class for all estimators in the mlearn library."""
    # TODO: Add support for nested parameters (e.g., for pipelines) if needed.
    def get_params(self):
        """Return model hyperparameters as a dict.

        Note:
        - Any attribute ending with '_' is treated as learned state and skipped.
        - ``deep`` is kept for API compatibility but not used in this simple version.
        """
        # Keep only user-set hyperparameters, not learned attributes like coef_.
        return {k: v for k, v in self.__dict__.items() if not k.endswith("_")}

    def set_params(self, **params):
        """Set model hyperparameters dynamically."""
        if not params:
            return self

        # Validate only against current hyperparameter names.
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{key}' for estimator "
                    f"{self.__class__.__name__}. "
                    f"Valid parameters are: {sorted(valid_params.keys())}."
                )

            setattr(self, key, value)

        return self

class TransformerMixin():
    """Mixin class for transformers in the mlearn library."""
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The data to fit and transform.
        - y: array-like of shape (n_samples,), default=None
            Target values (ignored by default).

        Returns:
        - X_transformed: array-like of shape (n_samples, n_transformed_features)
            The transformed data.
        """
        self.fit(X, y)
        return self.transform(X)

class RegressorMixin():
    """Mixin class for regressors in the mlearn library."""

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            Test samples.
        - y: array-like of shape (n_samples,)
            True values for X.

        Returns:
        - score: float
            R^2 score.
        """
        from .metrics.regression import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
class ClassifierMixin():
    """Mixin class for classifiers in the mlearn library."""

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            Test samples.
        - y: array-like of shape (n_samples,)
            True labels for X.

        Returns:
        - score: float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from .metrics.classification import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

class ClusterMixin():
    """Mixin class for clusterers in the mlearn library."""

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The input samples.
        - y: Ignored

        Returns:
        - labels: array-like of shape (n_samples,)
            Cluster labels for each point in the dataset.
        """
        self.fit(X, y)
        return self.predict(X)
    
class BaseModel(BaseEstimator, ABC):
    """Base class for all models in the mlearn library."""
    
    @abstractmethod
    def fit(self, X, y=None):
        """Fit the model to data.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The training input samples.
        - y: array-like of shape (n_samples,), default=None
            The target values (class labels in classification, real numbers in regression).
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Predict using the model.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        - y_pred: array-like of shape (n_samples,)
            The predicted values.
        """
        pass


class BaseTransformer(BaseEstimator, TransformerMixin, ABC):
    """Base class for all transformers in the mlearn library."""
    
    @abstractmethod
    def fit_transform(self, X, y=None):
        pass
    
    @abstractmethod
    def transform(self, X):
        """Transform the data.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        - X_transformed: array-like of shape (n_samples, n_transformed_features)
            The transformed data.
        """
        pass

class BasePipeline(BaseEstimator, ABC):
    """Simple base class for pipelines in the mlearn library.

     TODO:
    - Implement full pipeline logic
    - Support named steps
    - Support fit/transform chaining
    - Add caching
    """

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        """Fit the pipeline to data.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The training input samples.
        - y: array-like of shape (n_samples,), default=None
            The target values (class labels in classification, real numbers in regression).
        """        
        for name, step in self.steps[:-1]:
            X = step.fit_transform(X, y)
        
        self .steps[-1][1].fit(X, y)
        return self
    
    def predict(self, X):
        """Predict using the pipeline.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        - y_pred: array-like of shape (n_samples,)
            The predicted values.
        """
        for name, step in self.steps[:-1]:
            X = step.transform(X)
        
        return self.steps[-1][1].predict(X)