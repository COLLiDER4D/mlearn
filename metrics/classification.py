import numpy as np

def accuracy_score(y_true, y_pred):
    """Calculate the accuracy of predictions.
    It is defined as the ratio of correctly predicted samples to the total number of samples.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels.
    - y_pred: array-like of shape (n_samples,)
        Predicted labels.

    Returns:
    - accuracy: float
        The accuracy of the predictions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples.")
    
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = y_true.shape[0]
    
    return correct_predictions / total_predictions

def precision_score(y_true, y_pred):
    """Calculate the precision of predictions.
    It is defined as the ratio of true positives to the total predicted positives.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels.
    - y_pred: array-like of shape (n_samples,)
        Predicted labels.

    Returns:
    - precision: float
        The precision of the predictions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples.")
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    
    if predicted_positives == 0:
        return 0.0
    
    return true_positives / predicted_positives

def recall_score(y_true, y_pred):
    """Calculate the recall of predictions.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels.
    - y_pred: array-like of shape (n_samples,)
        Predicted labels.

    Returns:
    - recall: float
        The recall of the predictions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples.")
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    
    if actual_positives == 0:
        return 0.0
    
    return true_positives / actual_positives

def f1_score(y_true, y_pred):
    """Calculate the F1 score of predictions.
    It is the harmonic mean of precision and recall.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels.
    - y_pred: array-like of shape (n_samples,)
        Predicted labels.

    Returns:
    - f1: float
        The F1 score of the predictions.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def roc_auc_score(y_true, y_scores):
    """Calculate the ROC AUC score of predictions.
    It is the area under the receiver operating characteristic curve.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True binary labels.
    - y_scores: array-like of shape (n_samples,)
        Target scores, can either be probability estimates of the positive class or confidence values.

    Returns:
    - auc: float
        The ROC AUC score of the predictions.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    if y_true.shape[0] != y_scores.shape[0]:
        raise ValueError("y_true and y_scores must have the same number of samples.")
    # Sort indices by scores in descending order
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]
    
    # Calculate true positives and false positives
    total_positives = np.sum(y_true == 1)
    total_negatives = np.sum(y_true == 0)
    
    # Cumulative sums
    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)
    
    # Calculate AUC using trapezoidal rule
    auc = np.sum((fp[1:] - fp[:-1]) * (tp[1:] + tp[:-1])) / 2.0
    
    return auc / (total_positives * total_negatives)


def log_loss(y_true, y_pred_proba, eps=1e-15):
    """Calculate the log loss of predictions.
    It is defined as the negative log-likelihood of the true labels given the predicted probabilities.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True binary labels.
    - y_pred_proba: array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    - eps: float, default=1e-15
        Small value to avoid log(0).

    Returns:
    - log_loss: float
        The log loss of the predictions.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if y_true.shape[0] != y_pred_proba.shape[0]:
        raise ValueError("y_true and y_pred_proba must have the same number of samples.")
    
    # Clip predicted probabilities to avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    
    # Calculate log loss
    loss = -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
    
    return loss

