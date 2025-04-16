from typing import List, Optional, Union, Dict, Any
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class HydroTransformer(BaseEstimator, TransformerMixin):
    """Base class for all hydrological transformers.

    This class serves as a foundation for transformers that process hydrological data,
    providing consistent interface and utilities for handling both pandas DataFrames
    and numpy arrays.

    Attributes:
        columns: List of column names to transform
        _fitted: Boolean indicating if transformer has been fitted
        _fitted_state: Dictionary storing state required for inverse_transform
        _feature_names: List of feature names after transformation
    """

    def __init__(self, columns: Optional[List[str]] = None):
        """Initialize transformer.

        Args:
            columns: List of column names to transform. If None, transforms all columns.
        """
        self.columns = columns
        self._fitted = False
        self._fitted_state: Dict[str, Any] = {}
        self._feature_names = None

    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """Validate input data.

        Args:
            X: Input data to validate

        Raises:
            ValueError: If input validation fails
        """
        if isinstance(X, pd.DataFrame) and self.columns is not None:
            missing_cols = [col for col in self.columns if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in data: {missing_cols}")

    def _get_feature_columns(self, X: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        """Get list of feature columns to transform.

        Args:
            X: Input data

        Returns:
            List of column names/indices to transform
        """
        if isinstance(X, pd.DataFrame):
            if self.columns is None:
                return X.columns.tolist()
            return self.columns
        else:
            if self.columns is None:
                return list(range(X.shape[1]))
            return self.columns

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None
    ) -> "HydroTransformer":
        """Fit transformer to data.

        Args:
            X: Input data
            y: Target variable (optional, unused in most transformers)

        Returns:
            Self for method chaining
        """
        self._validate_input(X)

        # Store sklearn-specific attributes
        if isinstance(X, pd.DataFrame):
            self.n_features_in_ = X.shape[1]
            # Use numpy array for sklearn compatibility
            self.feature_names_in_ = np.array(X.columns.tolist())
        else:
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = np.array(
                [f"feature_{i}" for i in range(X.shape[1])]
            )

        # Original code
        if isinstance(X, pd.DataFrame):
            all_columns = X.columns.tolist()
        else:
            all_columns = [f"feature_{i}" for i in range(X.shape[1])]

        self._feature_names = all_columns
        self._fitted = True
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Transform data.

        Args:
            X: Input data to transform

        Returns:
            Transformed data in same format as input

        Raises:
            ValueError: If transformer hasn't been fitted
        """
        if not self._fitted:
            raise ValueError("Transformer must be fitted before calling transform")

        self._validate_input(X)

        # Default implementation: return data unchanged
        return X

    def inverse_transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Inverse transform data.

        Args:
            X: Input data to inverse transform

        Returns:
            Inverse transformed data in same format as input

        Raises:
            ValueError: If transformer hasn't been fitted
        """
        if not self._fitted:
            raise ValueError(
                "Transformer must be fitted before calling inverse_transform"
            )

        self._validate_input(X)

        # Default implementation: return data unchanged
        return X

    def get_feature_names_out(self) -> List[str]:
        """Get output feature names.

        Returns:
            List of output feature names

        Raises:
            ValueError: If transformer hasn't been fitted
        """
        if not self._fitted:
            raise ValueError("Transformer must be fitted before getting feature names")

        return self._feature_names
