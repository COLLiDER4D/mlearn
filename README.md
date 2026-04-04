# microMlKit

`microMlKit` is a small, educational machine learning toolkit built with NumPy. It provides a lightweight sklearn-style API with estimators, mixins, metrics, preprocessing helpers, model selection utilities, and a simple pipeline base.

## Overview

This project is designed for learning and experimentation rather than production use.

### Verified implemented pieces

- [x] Package metadata for `microMlKit` with Python `>=3.9`
- [x] Core estimator and mixin base classes in `microMlKit/base.py`
- [x] Regression metrics in `microMlKit/metrics/regression.py`
- [x] Train/test splitting and cross-validation helpers in `microMlKit/model_selection/`
- [x] Grid search utilities in `microMlKit/model_selection/search.py`
- [x] Source modules present for linear models, tree models, clustering, ensembles, neighbors, SVM, preprocessing, decomposition, and tests

### Verified incomplete or not yet implemented

- [ ] Nested parameter support in `BaseEstimator`
- [ ] Full pipeline logic in `BasePipeline`
- [ ] Named pipeline steps
- [ ] Fit/transform chaining improvements
- [ ] Caching support for pipelines
- [ ] A complete pipeline implementation in `microMlKit/pipeline/pipeline.py` (currently empty)

## Installation

From the project root:

```bash
pip install -e .
```

If you only want the runtime dependencies for local development:

```bash
pip install -r requirements.txt
```

## Quick Start

### Linear regression example

<pre><code class="language-python">import numpy as np
from src.linear_model.linear_regression import LinearRegression
from src.metrics.regression import mean_squared_error, r2_score
from src.model_selection.split import train_test_split

# Sample data
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([2.0, 4.1, 6.1, 8.2, 10.1])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Fit and predict
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate
print("MSE:", mean_squared_error(y_test, predictions))
print("R2:", r2_score(y_test, predictions))</code></pre>

### Pipeline usage

`BasePipeline` provides a simple fit/predict chain for a list of steps. Each intermediate step should support `fit_transform` during training and `transform` during prediction.

## Module Highlights

### `microMlKit/base.py`

Core abstractions and mixins:

- `BaseEstimator`
- `BaseModel`
- `BaseTransformer`
- `TransformerMixin`
- `RegressorMixin`
- `ClassifierMixin`
- `ClusterMixin`
- `BasePipeline`

### `microMlKit/linear_model/`

- `LinearRegression`
- `Lasso`
- `Ridge`
- `LogisticRegression`

### `microMlKit/tree/`

- Decision tree classifiers and regressors

### `microMlKit/cluster/`

- `KMeans`
- `DBSCAN`
- `Agglomerative` clustering

### `microMlKit/ensemble/`

- `RandomForest`
- `GradientBoosting`

### `microMlKit/neighbors/`

- K-nearest neighbors classifier and regressor

### `microMlKit/svm/`

- Support vector machine utilities and models

### `microMlKit/preprocessing/`

- Encoders
- Imputers
- Scalers

### `microMlKit/metrics/`

- Regression metrics such as `r2_score`, `mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, and `mean_absolute_percentage_error`

### `microMlKit/model_selection/`

- `train_test_split`
- `KFold`
- `StratifiedKFold`
- Search helpers

### `microMlKit/decomposition/`

- `PCA`

## Testing

The repository includes tests under `tests/`. To run them:

```bash
pytest
```

## Project Structure

```text
microMlKit/
├── base.py
├── cluster/
├── decomposition/
├── ensemble/
├── linear_model/
├── metrics/
├── model_selection/
├── neighbors/
├── pipeline/
├── preprocessing/
├── svm/
├── tree/
└── tests/
```

## Notes

- The codebase follows a simple sklearn-inspired interface, but it is intentionally minimal.
- Some modules are still evolving, so behavior may differ from full-featured machine learning libraries.
- If you extend the library, keep the estimator-style `fit` / `predict` / `transform` conventions consistent.


