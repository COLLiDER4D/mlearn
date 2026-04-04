# microMlKit

`microMlKit` is a lightweight, educational machine learning library built with NumPy and a scikit-learn-inspired API.

> Package/import name: `micromlkit` (lowercase)

## Overview

`micromlkit` is designed for learning, experimentation, and understanding core ML algorithms through readable implementations.

It provides:

- A familiar estimator workflow (`fit`, `predict`, `transform`)
- Reusable base classes and mixins
- Core algorithms across regression, classification, clustering, decomposition, trees, SVM, ensemble, and neighbors
- Preprocessing, metrics, model selection, and pipeline utilities

## Installation

### Install from PyPI

```bash
pip install micromlkit
```

### Install from source (development)

```bash
git clone <repository-url>
cd microMlKit
pip install -e .[dev]
```

## Quick Start

### Regression with preprocessing + pipeline

```python
import numpy as np

from micromlkit.linear_model import LinearRegression
from micromlkit.metrics import mean_squared_error, r2_score
from micromlkit.model_selection import train_test_split
from micromlkit.pipeline import Pipeline
from micromlkit.preprocessing import StandardScaler

X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([2.0, 4.1, 6.1, 8.0, 10.2])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression()),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
```

### Classification example

```python
import numpy as np

from micromlkit.linear_model import LogisticRegression
from micromlkit.metrics import accuracy_score

X = np.array([[0.1], [0.3], [0.6], [0.8]])
y = np.array([0, 0, 1, 1])

clf = LogisticRegression(learning_rate=0.1, n_iterations=2000)
clf.fit(X, y)
pred = clf.predict(X)

print("Accuracy:", accuracy_score(y, pred))
```

## Implemented Modules

| Module | Key Components |
|---|---|
| `micromlkit.base` | `BaseEstimator`, `BaseModel`, `BaseTransformer`, mixins, `BasePipeline` |
| `micromlkit.linear_model` | `LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression` |
| `micromlkit.cluster` | `KMeans`, `DBSCAN`, `AgglomerativeClustering` |
| `micromlkit.tree` | `DecisionTreeClassifier`, `DecisionTreeRegressor` |
| `micromlkit.ensemble` | `RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor` |
| `micromlkit.neighbors` | `KNNClassifier`, `KNNRegressor` |
| `micromlkit.svm` | `SVC`, `SVR` |
| `micromlkit.decomposition` | `PCA` |
| `micromlkit.preprocessing` | `StandardScaler`, `SimpleImputer`, `LabelEncoder` |
| `micromlkit.metrics` | classification + regression metrics |
| `micromlkit.model_selection` | `train_test_split`, `KFold`, `StratifiedKFold`, `cross_val_score`, `ParameterGrid`, `GridSearchCV` |
| `micromlkit.pipeline` | `Pipeline` |
| `micromlkit.utils` | kernel/math and validation helpers |

## Project Scope

- **Goal:** educational readability and practical experimentation
- **Runtime dependency:** NumPy
- **Python version:** `>=3.9`
- **Production note:** this project is intended for learning and small-scale experimentation rather than production-grade ML workloads

## Testing

Run the test suite:

```bash
pytest -ra
```

Latest local verification in this repository: **157 passed**.

## Repository Structure

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
├── tree/
└── utils/
```

## License

See the [`LICENSE`](./LICENSE) file in the repository root.
