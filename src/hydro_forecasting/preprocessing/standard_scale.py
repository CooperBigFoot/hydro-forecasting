import numpy as np
import pandas as pd

from .base import HydroTransformer


class StandardScaleTransformer(HydroTransformer):
    """Standardizes features by removing the mean and scaling to unit variance."""

    def __init__(self, columns=None):
        super().__init__(columns=columns)
        self._fitted_state["mean"] = {}
        self._fitted_state["std"] = {}

    def fit(self, X, y=None):
        super().fit(X, y)

        feature_cols = self._get_feature_columns(X)

        for col in feature_cols:
            if isinstance(X, pd.DataFrame):
                self._fitted_state["mean"][col] = X[col].mean()
                self._fitted_state["std"][col] = X[col].std()
            else:
                col_idx = col if isinstance(col, int) else list(X.columns).index(col)
                self._fitted_state["mean"][col] = np.mean(X[:, col_idx])
                self._fitted_state["std"][col] = np.std(X[:, col_idx])

            # Handle zero std (constant features)
            if self._fitted_state["std"][col] == 0:
                self._fitted_state["std"][col] = 1.0

        return self

    def transform(self, X):
        super().transform(X)

        feature_cols = self._get_feature_columns(X)

        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for col in feature_cols:
                mean = self._fitted_state["mean"].get(col, 0)
                std = self._fitted_state["std"].get(col, 1)
                X_transformed[col] = (X[col] - mean) / std
            return X_transformed
        else:
            X_transformed = X.copy()
            for col in feature_cols:
                col_idx = col if isinstance(col, int) else list(X.columns).index(col)
                mean = self._fitted_state["mean"].get(col, 0)
                std = self._fitted_state["std"].get(col, 1)
                X_transformed[:, col_idx] = (X[:, col_idx] - mean) / std
            return X_transformed

    def inverse_transform(self, X):
        super().inverse_transform(X)

        feature_cols = self._get_feature_columns(X)

        if isinstance(X, pd.DataFrame):
            X_inverse = X.copy()
            for col in feature_cols:
                mean = self._fitted_state["mean"].get(col, 0)
                std = self._fitted_state["std"].get(col, 1)
                X_inverse[col] = X[col] * std + mean
            return X_inverse
        else:
            X_inverse = X.copy()
            for col in feature_cols:
                col_idx = col if isinstance(col, int) else list(X.columns).index(col)
                mean = self._fitted_state["mean"].get(col, 0)
                std = self._fitted_state["std"].get(col, 1)
                X_inverse[:, col_idx] = X[:, col_idx] * std + mean
            return X_inverse
