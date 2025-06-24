import pandas as pd
from sklearn.preprocessing import PowerTransformer

from .base import HydroTransformer, register_transformer


@register_transformer("normalize")
class NormalizeTransformer(HydroTransformer):
    """Transformer that attempts to transform features toward a normal distribution.

    Uses the Yeo-Johnson transformation which works with both positive and negative values.
    This transformer is useful for making data more Gaussian-like, which can improve
    the performance of many machine learning algorithms that assume normally
    distributed features.

    Attributes:
        columns: List of column names to transform
        copy: Whether to make a copy of the input data
        _fitted_state: Dictionary storing lambdas and other parameters for each column
    """

    def __init__(self, columns=None, copy=True):
        """Initialize NormalizeTransformer.

        Args:
            columns: List of column names to transform. If None, transforms all columns.
            copy: Whether to make a copy of the input data
        """
        super().__init__(columns=columns)
        self.copy = copy
        self._fitted_state["lambdas"] = {}
        self._transformer = None

    def fit(self, X, y=None):
        """Fit transformer by estimating the optimal parameters for normalization.

        Args:
            X: Input data
            y: Target variable (unused)

        Returns:
            Self for method chaining
        """
        # Call parent fit for validation
        super().fit(X, y)

        feature_cols = self._get_feature_columns(X)

        # Create and fit the sklearn PowerTransformer
        self._transformer = PowerTransformer(
            method="yeo-johnson",
            standardize=False,  # We don't want to standardize as we have a separate transformer for that
            copy=self.copy,
        )

        # Handle pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X_subset = X[feature_cols].copy()
            self._transformer.fit(X_subset)

            # Store lambdas for each column
            for i, col in enumerate(feature_cols):
                self._fitted_state["lambdas"][col] = self._transformer.lambdas_[i]

        # Handle numpy array
        else:
            # Extract columns to transform
            cols_idx = [col if isinstance(col, int) else list(X.columns).index(col) for col in feature_cols]
            X_subset = X[:, cols_idx].copy()
            self._transformer.fit(X_subset)

            # Store lambdas for each column
            for i, col in enumerate(feature_cols):
                self._fitted_state["lambdas"][col] = self._transformer.lambdas_[i]

        return self

    def transform(self, X):
        """Apply normalization transformation to data.

        Args:
            X: Input data to transform

        Returns:
            Normalized data
        """
        # Call parent transform for validation
        super().transform(X)

        feature_cols = self._get_feature_columns(X)

        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy() if self.copy else X
            X_subset = X[feature_cols].copy()

            # Apply transformation
            X_subset_transformed = self._transformer.transform(X_subset)

            # Update values in the output DataFrame
            for i, col in enumerate(feature_cols):
                X_transformed[col] = X_subset_transformed[:, i]

            return X_transformed
        else:
            X_transformed = X.copy() if self.copy else X

            # Extract columns to transform
            cols_idx = [col if isinstance(col, int) else list(X.columns).index(col) for col in feature_cols]
            X_subset = X[:, cols_idx].copy()

            # Apply transformation
            X_subset_transformed = self._transformer.transform(X_subset)

            # Update values in the output array
            for i, idx in enumerate(cols_idx):
                X_transformed[:, idx] = X_subset_transformed[:, i]

            return X_transformed

    def inverse_transform(self, X):
        """Apply inverse normalization transformation to data.

        Args:
            X: Input data to inverse transform

        Returns:
            Inverse transformed data
        """
        # Call parent inverse_transform for validation
        super().inverse_transform(X)

        feature_cols = self._get_feature_columns(X)

        if isinstance(X, pd.DataFrame):
            X_inverse = X.copy() if self.copy else X
            X_subset = X[feature_cols].copy()

            # Apply inverse transformation
            X_subset_inverse = self._transformer.inverse_transform(X_subset)

            # Update values in the output DataFrame
            for i, col in enumerate(feature_cols):
                X_inverse[col] = X_subset_inverse[:, i]

            return X_inverse
        else:
            X_inverse = X.copy() if self.copy else X

            # Extract columns to transform
            cols_idx = [col if isinstance(col, int) else list(X.columns).index(col) for col in feature_cols]
            X_subset = X[:, cols_idx].copy()

            # Apply inverse transformation
            X_subset_inverse = self._transformer.inverse_transform(X_subset)

            # Update values in the output array
            for i, idx in enumerate(cols_idx):
                X_inverse[:, idx] = X_subset_inverse[:, i]

            return X_inverse
