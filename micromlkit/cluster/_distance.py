import numpy as np


_ALLOWED_METRICS = {
    "euclidean",
    "manhattan",
    "cosine",
    "chebyshev",
    "minkowski",
}


def ensure_2d_float_array(X, name="X"):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape (n_samples, n_features).")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and one feature.")
    return X


def validate_metric(metric):
    if not isinstance(metric, str):
        raise ValueError(
            "metric must be one of {'euclidean', 'manhattan', 'cosine', 'chebyshev', 'minkowski'}."
        )

    metric = metric.lower()
    if metric not in _ALLOWED_METRICS:
        raise ValueError(
            "metric must be one of {'euclidean', 'manhattan', 'cosine', 'chebyshev', 'minkowski'}."
        )
    return metric


def validate_minkowski_p(metric, p):
    if metric != "minkowski":
        return 2.0

    if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)):
        raise ValueError("p must be a numeric value greater than or equal to 1 for minkowski metric.")

    p = float(p)
    if p < 1.0:
        raise ValueError("p must be greater than or equal to 1 for minkowski metric.")

    return p


def validate_feature_count(X, n_features_in, estimator_name):
    if X.shape[1] != n_features_in:
        raise ValueError(
            f"X has {X.shape[1]} features, but {estimator_name} was fitted with "
            f"{n_features_in} features."
        )


def pairwise_distances(X, Y, metric="euclidean", p=2.0):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays of shape (n_samples, n_features).")

    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features.")

    metric = validate_metric(metric)
    p = validate_minkowski_p(metric, p)

    if metric != "cosine":
        diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]

        if metric == "euclidean":
            return np.sqrt(np.sum(diff ** 2, axis=2))

        if metric == "manhattan":
            return np.sum(np.abs(diff), axis=2)

        if metric == "chebyshev":
            return np.max(np.abs(diff), axis=2)

        if metric == "minkowski":
            return np.sum(np.abs(diff) ** p, axis=2) ** (1.0 / p)

    # cosine distance
    dot_products = X @ Y.T
    x_norm = np.linalg.norm(X, axis=1, keepdims=True)
    y_norm = np.linalg.norm(Y, axis=1, keepdims=True).T
    denom = x_norm * y_norm

    cosine_similarity = np.divide(
        dot_products,
        denom,
        out=np.zeros_like(dot_products, dtype=float),
        where=denom > 0,
    )

    # Treat two zero vectors as maximally similar to avoid unstable behavior.
    x_zero = np.isclose(x_norm, 0.0)
    y_zero = np.isclose(y_norm, 0.0)
    both_zero = x_zero & y_zero
    cosine_similarity = np.where(both_zero, 1.0, cosine_similarity)
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

    return 1.0 - cosine_similarity