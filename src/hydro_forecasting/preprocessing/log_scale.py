import numpy as np
import pandas as pd

from .base import HydroTransformer


class LogTransformer(HydroTransformer):
    """Transformer for applying log transformation to data.

    This transformer applies log1p transformation to data, handling negative values
    by adding an offset. The offset is stored during fitting and applied during
    inverse transformation.

    Attributes:
        epsilon: Small value added to ensure strictly positive values
        _fitted_state: Dictionary storing offsets for each column
    """

    def __init__(self, columns: list[str] | None = None, epsilon: float = 1e-3):
        """Initialize LogTransformer.

        Args:
            columns: List of column names to transform. If None, transforms all columns.
            epsilon: Small constant added to ensure strictly positive values
        """
        super().__init__(columns=columns)
        self.epsilon = epsilon
        self._fitted_state["offsets"] = {}

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | None = None) -> "LogTransformer":
        """Fit transformer by computing offsets for negative values.

        Args:
            X: Input data
            y: Target variable (unused)

        Returns:
            Self for method chaining
        """
        super().fit(X, y)

        feature_cols = self._get_feature_columns(X)
        self._fitted_state["offsets"] = {}

        if isinstance(X, pd.DataFrame):
            for col_idx, col in enumerate(feature_cols):
                min_val = X[col].min()
                self._fitted_state["offsets"][col] = abs(min_val) + self.epsilon if min_val < 0 else 0
        else:
            for col_idx, col in enumerate(feature_cols):
                min_val = X[:, col].min()
                self._fitted_state["offsets"][col] = abs(min_val) + self.epsilon if min_val < 0 else 0

        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Apply log1p transformation to data.

        Args:
            X: Input data to transform

        Returns:
            Log-transformed data
        """
        super().transform(X)  # For validation

        feature_cols = self._get_feature_columns(X)

        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for col in feature_cols:
                offset = self._fitted_state["offsets"].get(col, 0)
                X_transformed[col] = np.log1p(X[col] + offset)
            return X_transformed
        else:
            X_transformed = X.copy()
            for col_idx, col in enumerate(feature_cols):
                offset = self._fitted_state["offsets"].get(col, 0)
                X_transformed[:, col] = np.log1p(X[:, col] + offset)
            return X_transformed

    def inverse_transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Apply inverse log1p transformation to data.

        Args:
            X: Input data to inverse transform

        Returns:
            Inverse log-transformed data
        """
        super().inverse_transform(X)  # For validation

        feature_cols = self._get_feature_columns(X)

        if isinstance(X, pd.DataFrame):
            X_inverse = X.copy()
            for col in feature_cols:
                offset = self._fitted_state["offsets"].get(col, 0)
                X_inverse[col] = np.expm1(X[col]) - offset
            return X_inverse
        else:
            X_inverse = X.copy()
            for col_idx, col in enumerate(feature_cols):
                offset = self._fitted_state["offsets"].get(col, 0)
                X_inverse[:, col] = np.expm1(X[:, col]) - offset
            return X_inverse
