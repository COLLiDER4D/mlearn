# microMlKit

`microMlKit` is a lightweight, educational machine learning toolkit built with NumPy and a scikit-learn-inspired API.

> Package name for imports and installation metadata: `micromlkit` (all lowercase).

## What this project is

This library is intended for learning, experimentation, and understanding ML internals—not production workloads.

It includes:

- Estimator and mixin base classes
- Linear models
- Clustering, tree, ensemble, neighbors, SVM, and decomposition modules
- Preprocessing helpers
- Classification and regression metrics
- Model selection helpers (`train_test_split`, `KFold`, `StratifiedKFold`, cross-validation, grid search)

## Installation

Install from PyPI:

```bash
pip install micromlkit
```

## Quick start

### Linear regression example

```python
import numpy as np

from micromlkit.linear_model.linear_regression import LinearRegression
from micromlkit.metrics.regression import mean_squared_error, r2_score
from micromlkit.model_selection.split import train_test_split

X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([2.0, 4.1, 6.1, 8.2, 10.1])

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.4, random_state=42
)

model = LinearRegression().fit(X_train, y_train)
predictions = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, predictions))
print("R2:", r2_score(y_test, predictions))
```

### Base pipeline example

`BasePipeline` (from `micromlkit.base`) supports a simple transform-then-estimate flow.

```python
import numpy as np

from micromlkit.base import BasePipeline


class DoubleTransformer:
	def fit_transform(self, X, y=None):
		return np.asarray(X) * 2

	def transform(self, X):
		return np.asarray(X) * 2


class FirstColumnModel:
	def fit(self, X, y=None):
		return self

	def predict(self, X):
		X = np.asarray(X)
		return X[:, 0]


pipeline = BasePipeline(
	steps=[("double", DoubleTransformer()), ("model", FirstColumnModel())]
)

X = np.array([[1, 2], [3, 4]])
y = np.array([10, 20])

pipeline.fit(X, y)
print(pipeline.predict(X))
```

## Module highlights

- `micromlkit.base`: `BaseEstimator`, `BaseModel`, `BaseTransformer`, mixins, `BasePipeline`
- `micromlkit.linear_model`: `LinearRegression`, `Lasso`, `Ridge`, `LogisticRegression`
- `micromlkit.metrics`: classification + regression metrics
- `micromlkit.model_selection`: splitters, cross-validation, search utilities
- `micromlkit.preprocessing`: encoder, imputer, scaler helpers
- `micromlkit.cluster`, `tree`, `ensemble`, `neighbors`, `svm`, `decomposition`

## Testing

Run the test suite:

```bash
pytest
```

## Current status

- API is intentionally minimal and may evolve.
- Some advanced conveniences (for example, richer nested-parameter handling and expanded pipeline ergonomics) are still limited.

## Implementation checklist

Status below is based on current source files in `micromlkit/`.

### Implemented (non-empty)

- [x] `micromlkit/__init__.py`
- [x] `micromlkit/base.py`
- [x] `micromlkit/linear_model/linear_regression.py`
- [x] `micromlkit/linear_model/lasso.py`
- [x] `micromlkit/linear_model/ridge.py`
- [x] `micromlkit/linear_model/logistic_regression.py`
- [x] `micromlkit/linear_model/__init__.py`
- [x] `micromlkit/metrics/classification.py`
- [x] `micromlkit/metrics/regression.py`
- [x] `micromlkit/metrics/__init__.py`
- [x] `micromlkit/model_selection/split.py`
- [x] `micromlkit/model_selection/cross_validation.py`
- [x] `micromlkit/model_selection/search.py`
- [x] `micromlkit/model_selection/__init__.py`
- [x] `micromlkit/preprocessing/encoder.py`
- [x] `micromlkit/preprocessing/imputer.py`
- [x] `micromlkit/preprocessing/scaler.py`
- [x] `micromlkit/preprocessing/__init__.py`
- [x] `micromlkit/decomposition/pca.py`
- [x] `micromlkit/decomposition/__init__.py`
- [x] `micromlkit/pipeline/pipeline.py`
- [x] `micromlkit/pipeline/__init__.py`

### Pending implementation (empty files)

#### Cluster
- [ ] `micromlkit/cluster/agglomerative.py`
- [ ] `micromlkit/cluster/dbscan.py`
- [ ] `micromlkit/cluster/kmeans.py`
- [ ] `micromlkit/cluster/__init__.py` (exports)

#### Ensemble
- [ ] `micromlkit/ensemble/random_forest.py`
- [ ] `micromlkit/ensemble/gradient_boosting.py`
- [ ] `micromlkit/ensemble/__init__.py` (exports)

#### Neighbors
- [ ] `micromlkit/neighbors/knn_classifier.py`
- [ ] `micromlkit/neighbors/knn_regressor.py`
- [ ] `micromlkit/neighbors/__init__.py` (exports)

#### SVM
- [ ] `micromlkit/svm/svm.py`
- [ ] `micromlkit/svm/__init__.py` (exports)

#### Tree
- [ ] `micromlkit/tree/decision_tree_classifier.py`
- [ ] `micromlkit/tree/decision_tree_regressor.py`
- [ ] `micromlkit/tree/__init__.py` (exports)

#### Utils
- [ ] `micromlkit/utils/math.py`
- [ ] `micromlkit/utils/validation.py`
- [ ] `micromlkit/utils/__init__.py` (exports)

## Project layout

```text
micromlkit/
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
└── tree/
```


