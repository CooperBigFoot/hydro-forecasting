import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, clone


class UnifiedPipeline(BaseEstimator, TransformerMixin):
    """Unified pipeline that applies a single pipeline to all data.

    This transformer wraps a standard sklearn Pipeline and applies it uniformly
    to all input data, regardless of group membership. Unlike GroupedPipeline,
    it fits a single pipeline on the entirety of data it receives.

    Attributes:
        pipeline: sklearn Pipeline to apply to all data
        columns: Columns to transform
        fitted_pipeline: The fitted pipeline instance
    """

    def __init__(self, pipeline: Pipeline, columns: list[str]):
        """Initialize UnifiedPipeline.

        Args:
            pipeline: sklearn Pipeline to apply
            columns: Columns to transform
        """
        self.pipeline = pipeline
        self.columns = columns
        self.fitted_pipeline: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "UnifiedPipeline":
        """Fit the pipeline on the entire dataset.

        Args:
            X: Input data
            y: Target variable (optional, passed to pipeline fit method)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If columns not found in data
        """
        # Validate columns
        missing_cols = [col for col in self.columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
        
        # Handle empty data
        if len(X) == 0 and len(self.columns) > 0:
            # Create dummy data with one row of zeros for fitting
            dummy_data = pd.DataFrame(0, index=[0], columns=self.columns)
            self.fitted_pipeline = clone(self.pipeline)
            self.fitted_pipeline.fit(dummy_data, None)
        elif len(self.columns) == 0:
            # Handle empty columns list - skip pipeline fitting entirely
            self.fitted_pipeline = clone(self.pipeline)
        else:
            # Normal case
            self.fitted_pipeline = clone(self.pipeline)
            self.fitted_pipeline.fit(X[self.columns], y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted pipeline.

        Args:
            X: Input data

        Returns:
            Transformed data

        Raises:
            ValueError: If pipeline is not fitted or columns not found
        """
        if self.fitted_pipeline is None:
            raise ValueError("Pipeline must be fitted before transform")

        # Validate columns
        missing_cols = [col for col in self.columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")

        X_transformed = X.copy()
        
        # If no columns to transform, return original data
        if len(self.columns) == 0:
            return X_transformed
            
        transformed_data = self.fitted_pipeline.transform(X[self.columns])

        # Handle the case where pipeline returns ndarray instead of DataFrame
        if isinstance(transformed_data, pd.DataFrame):
            X_transformed[self.columns] = transformed_data
        else:
            # Assume numpy array
            for i, col in enumerate(self.columns):
                X_transformed[col] = transformed_data[:, i]

        return X_transformed

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform data using the fitted pipeline.

        Args:
            X: Input data

        Returns:
            Inverse transformed data

        Raises:
            ValueError: If pipeline is not fitted or doesn't support inverse_transform
        """
        if self.fitted_pipeline is None:
            raise ValueError("Pipeline must be fitted before inverse_transform")

        if not hasattr(self.fitted_pipeline, "inverse_transform"):
            raise ValueError("Pipeline does not support inverse_transform")

        # Validate columns
        missing_cols = [col for col in self.columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")

        X_inverse = X.copy()
        
        # If no columns to transform, return original data
        if len(self.columns) == 0:
            return X_inverse
            
        inverse_data = self.fitted_pipeline.inverse_transform(X[self.columns])

        # Handle the case where pipeline returns ndarray instead of DataFrame
        if isinstance(inverse_data, pd.DataFrame):
            X_inverse[self.columns] = inverse_data
        else:
            # Assume numpy array
            for i, col in enumerate(self.columns):
                X_inverse[col] = inverse_data[:, i]

        return X_inverse
