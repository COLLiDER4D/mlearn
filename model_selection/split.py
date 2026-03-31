import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    """Splits arrays or matrices into random train and test subsets.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    X_train : array-like of shape (n_train_samples, n_features)
        The training input samples.
    X_test : array-like of shape (n_test_samples, n_features)
        The testing input samples.
    y_train : array-like of shape (n_train_samples,)
        The training target values.
    y_test : array-like of shape (n_test_samples,)
        The testing target values.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_size = int(n_samples * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def KFold(n_splits=5, shuffle=False, random_state=None):
    """K-Folds cross-validator.
    
    Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the shuffling when shuffle is True.

    Returns
    -------
    generator of tuples (train_indices, test_indices)
        The generator yields tuples of train and test indices for each fold.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    
    def split(X):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)

        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[:n_samples % n_splits] += 1
        current = 0
        
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate((indices[:start], indices[stop:]))
            yield train_indices, test_indices
            current = stop

    return split


def StratifiedKFold(n_splits=5, shuffle=False, random_state=None):
    """Stratified K-Folds cross-validator.
    
    Provides train/test indices to split data in train/test sets. This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the shuffling when shuffle is True.

    Returns
    -------
    generator of tuples (train_indices, test_indices)
        The generator yields tuples of train and test indices for each fold.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    
    def split(X, y):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)

        # Group samples by class
        classes, y_indices = np.unique(y, return_inverse=True)
        class_counts = np.bincount(y_indices)
        
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[:n_samples % n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = []
            for cls in classes:
                cls_indices = indices[y_indices == cls]
                cls_fold_size = int(fold_size * class_counts[cls] / n_samples)
                test_indices.extend(cls_indices[:cls_fold_size])
                indices = np.setdiff1d(indices, cls_indices[:cls_fold_size])
            train_indices = np.setdiff1d(indices, test_indices)
            yield train_indices, test_indices
            current = stop

    return split
