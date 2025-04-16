# Hydrology Preprocessing Framework

This module provides a robust framework for preprocessing hydrological time series data, with support for catchment-specific transformations, grouped operations, and data quality checking.

## Core Concepts

- **BaseTransformer**: All transformers extend a common base class with consistent interfaces
- **Grouped Transformations**: Apply different transformations for each catchment
- **Pipelines**: Combine transformers in sequence for complex preprocessing
- **Inverse Transformations**: Revert data to original scale for interpretable predictions

## Custom Transformer Implementation

To create a custom transformer, follow these steps:

### 1. Extend the Base Class

```python
from preprocessing.base import HydroTransformer

class MyCustomTransformer(HydroTransformer):
    def __init__(self, columns=None, param1=1.0):
        super().__init__(columns=columns)
        self.param1 = param1
        self._fitted_state["custom_values"] = {}  # Store fitted values here
```

### 2. Implement Required Methods

#### `fit` Method

```python
def fit(self, X, y=None):
    # Call parent fit for validation
    super().fit(X, y)
    
    # Process each feature
    feature_cols = self._get_feature_columns(X)
    
    for col in feature_cols:
        # Calculate and store parameters needed for transform/inverse_transform
        if isinstance(X, pd.DataFrame):
            value = X[col].mean()  # Example computation
        else:
            col_idx = col if isinstance(col, int) else list(X.columns).index(col)
            value = np.mean(X[:, col_idx])
            
        self._fitted_state["custom_values"][col] = value
        
    return self
```

#### `transform` Method

```python
def transform(self, X):
    # Call parent transform for validation
    super().transform(X)
    
    feature_cols = self._get_feature_columns(X)
    
    # Create copy to avoid modifying original data
    if isinstance(X, pd.DataFrame):
        X_transformed = X.copy()
        for col in feature_cols:
            value = self._fitted_state["custom_values"].get(col, 0)
            X_transformed[col] = X[col] * value  # Apply transformation
        return X_transformed
    else:
        X_transformed = X.copy()
        for col_idx, col in enumerate(feature_cols):
            value = self._fitted_state["custom_values"].get(col, 0)
            X_transformed[:, col_idx] = X[:, col_idx] * value
        return X_transformed
```

#### `inverse_transform` Method

```python
def inverse_transform(self, X):
    # Call parent inverse_transform for validation
    super().inverse_transform(X)
    
    feature_cols = self._get_feature_columns(X)
    
    # Create copy to avoid modifying original data
    if isinstance(X, pd.DataFrame):
        X_inverse = X.copy()
        for col in feature_cols:
            value = self._fitted_state["custom_values"].get(col, 0)
            X_inverse[col] = X[col] / value if value != 0 else X[col]
        return X_inverse
    else:
        X_inverse = X.copy()
        for col_idx, col in enumerate(feature_cols):
            value = self._fitted_state["custom_values"].get(col, 0) 
            X_inverse[:, col_idx] = X[:, col_idx] / value if value != 0 else X[:, col_idx]
        return X_inverse
```

## Handling Different Data Types

The framework supports both pandas DataFrames and numpy arrays. Your transformer should handle both formats:

- Use `isinstance(X, pd.DataFrame)` to check the input type
- For DataFrames, access columns by name: `X[col]`
- For arrays, use indices: `X[:, col_idx]`

## Using Grouped Transformers

To apply different transformations for each catchment:

```python
from preprocessing.grouped import GroupedTransformer
from sklearn.pipeline import Pipeline
from preprocessing.standard_scale import StandardScaleTransformer
from preprocessing.log_scale import LogTransformer

# Create a pipeline
pipeline = Pipeline([
    ('log', LogTransformer()),
    ('scale', StandardScaleTransformer())
])

# Create a grouped transformer
grouped_transformer = GroupedTransformer(
    pipeline=pipeline,
    columns=['streamflow', 'precipitation'],
    group_identifier='gauge_id'
)

# Fit and transform
grouped_transformer.fit(data)
transformed_data = grouped_transformer.transform(data)
```

## Creating Preprocessing Pipelines

Combine multiple transformers in a pipeline:

```python
preprocessing_config = {
    "features": {
        "pipeline": Pipeline([
            ("log", LogTransformer(columns=["precipitation"])),
            ("scale", StandardScaleTransformer(columns=["precipitation", "temperature"]))
        ])
    },
    "target": {
        "pipeline": GroupedTransformer(
            pipeline=Pipeline([
                ("log", LogTransformer()),
                ("scale", StandardScaleTransformer())
            ]),
            columns=["streamflow"],
            group_identifier="gauge_id"
        )
    }
}
```

## Examples of Available Transformers

- **StandardScaleTransformer**: Standardize features by removing mean and scaling to unit variance
- **LogTransformer**: Apply log transformation to features, with handling for negative values
- **GroupedTransformer**: Apply a transformer separately to each group (e.g., catchment)

## Best Practices

1. Always store fitted parameters in `self._fitted_state` dictionary
2. Support both pandas DataFrames and numpy arrays
3. Make transformers compatible with sklearn's Pipeline interface
4. Implement proper validation in fit/transform methods
5. Ensure inverse_transform correctly reverses transformations
6. Handle edge cases (e.g., zero or negative values for log transforms)
