import numpy as np

from model_selection.split import KFold


def cross_val_score(estimator, X, y, cv, scoring):
    """Evaluate a score by cross-validation.
    <p>It performs K-Fold cross-validation on the given estimator and returns an array of scores for each fold.</p>
    Parameters
    ----------
    estimator : object
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.
    cv : int or cross-validation generator
        Determines the cross-validation splitting strategy. If int, it is the number of folds. If a cross-validation generator is passed, it should provide train/test indices to split the data.
    scoring : callable
        A callable that takes (y_true, y_pred) and returns a score.

    Returns
    -------
    scores : array of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.
    """
    if scoring is None:
        scoring = estimator.score

    if isinstance(cv, int):
        cv = KFold(n_splits=cv)
    scores = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = np.asarray(X)[train_index], np.asarray(X)[test_index]
        y_train, y_test = np.asarray(y)[train_index], np.asarray(y)[test_index]

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        score = scoring(y_test, y_pred)
        scores.append(score)
    return np.array(scores)
