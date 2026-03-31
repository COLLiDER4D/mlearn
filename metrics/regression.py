import numpy as np


def r2_score(y_true, y_pred):
    """
    R^2 (coefficient of determination) regression score function. 
    
    <p>It is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y_true, disregarding the input features, would get a R^2 score of 0.0.</p>
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    score : float
        R^2 score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def mean_squared_error(y_true, y_pred):
    """
    Mean squared error regression loss.

    <p>It is the average of the squared differences between the predicted and actual values. The lower the mean squared error, the better the model's performance.</p>

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    mse : float
        Mean squared error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Root mean squared error regression loss.

    <p>It is the square root of the average of the squared differences between the predicted and actual values. The lower the root mean squared error, the better the model's performance.</p>

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    rmse : float
        Root mean squared error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    """
    Mean absolute error regression loss.

    <p>It is the average of the absolute differences between the predicted and actual values. The lower the mean absolute error, the better the model's performance.</p>

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    mae : float
        Mean absolute error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_true - y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Mean absolute percentage error regression loss.

    <p>It is the average of the absolute percentage differences between the predicted and actual values. The lower the mean absolute percentage error, the better the model's performance.</p>

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    mape : float
        Mean absolute percentage error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if np.any(y_true == 0):
        raise ValueError("Mean absolute percentage error is undefined when y_true contains zero values.")

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100