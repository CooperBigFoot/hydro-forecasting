"""
Comprehensive unit tests for UnifiedPipeline.

Tests cover:
- Pipeline initialization and configuration
- Fitting single pipeline on all data
- Transform functionality
- Inverse transform functionality
- Column processing and validation
- Error handling and edge cases
- Integration with sklearn pipelines
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from hydro_forecasting.preprocessing.unified import UnifiedPipeline


class MockTransformer(BaseEstimator, TransformerMixin):
    """Mock transformer for testing that tracks calls."""

    def __init__(self, multiply_factor=2):
        self.multiply_factor = multiply_factor
        self.fit_called = False
        self.transform_called = False

    def fit(self, X, y=None):
        self.fit_called = True
        return self

    def transform(self, X):
        self.transform_called = True
        return X * self.multiply_factor

    def inverse_transform(self, X):
        return X / self.multiply_factor


class TestUnifiedPipelineInitialization:
    """Test UnifiedPipeline initialization and configuration."""

    def test_unified_pipeline_init_basic(self):
        """Test basic initialization of UnifiedPipeline."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        columns = ["temperature", "precipitation"]

        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=columns)

        assert unified_pipeline.pipeline == pipeline
        assert unified_pipeline.columns == columns
        assert unified_pipeline.fitted_pipeline is None

    def test_unified_pipeline_init_empty_columns(self):
        """Test initialization with empty columns list."""
        pipeline = Pipeline([("scaler", StandardScaler())])

        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=[])

        assert unified_pipeline.columns == []

    def test_unified_pipeline_init_single_column(self):
        """Test initialization with single column."""
        pipeline = Pipeline([("scaler", StandardScaler())])

        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature"])

        assert unified_pipeline.columns == ["temperature"]
        assert len(unified_pipeline.columns) == 1


class TestUnifiedPipelineFitting:
    """Test UnifiedPipeline fitting functionality."""

    def test_unified_pipeline_fit_basic(self, synthetic_clean_data):
        """Test basic fitting functionality."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature", "precipitation"])

        unified_pipeline.fit(synthetic_clean_data)

        assert unified_pipeline.fitted_pipeline is not None
        assert hasattr(unified_pipeline.fitted_pipeline["scaler"], "mean_")
        assert hasattr(unified_pipeline.fitted_pipeline["scaler"], "scale_")

    def test_unified_pipeline_fit_with_target(self, synthetic_clean_data):
        """Test fitting with target variable."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature", "precipitation"])

        # Use streamflow as target
        y = synthetic_clean_data["streamflow"]

        unified_pipeline.fit(synthetic_clean_data, y)

        assert unified_pipeline.fitted_pipeline is not None
        # Target should be passed to pipeline fit method

    def test_unified_pipeline_fit_single_column(self, synthetic_clean_data):
        """Test fitting with single column."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature"])

        unified_pipeline.fit(synthetic_clean_data)

        assert unified_pipeline.fitted_pipeline is not None
        # Should fit on just the temperature column

    def test_unified_pipeline_fit_missing_columns(self, synthetic_clean_data):
        """Test error when required columns are missing."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["nonexistent_column"])

        with pytest.raises(ValueError, match="Columns not found in data"):
            unified_pipeline.fit(synthetic_clean_data)

    def test_unified_pipeline_fit_subset_of_data(self, synthetic_clean_data):
        """Test fitting on subset of data."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature", "precipitation"])

        # Fit on subset of basins
        subset_data = synthetic_clean_data[synthetic_clean_data["gauge_id"].isin(["basin_001", "basin_002"])]
        unified_pipeline.fit(subset_data)

        assert unified_pipeline.fitted_pipeline is not None

        # Should be able to transform full dataset
        transformed = unified_pipeline.transform(synthetic_clean_data)
        assert transformed.shape[0] == synthetic_clean_data.shape[0]

    def test_unified_pipeline_fit_with_mock_transformer(self, synthetic_clean_data):
        """Test fitting with mock transformer to verify calls."""
        mock_transformer = MockTransformer(multiply_factor=3)
        pipeline = Pipeline([("mock", mock_transformer)])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature"])

        unified_pipeline.fit(synthetic_clean_data)

        # Check that fit was called on the fitted pipeline, not the original mock
        assert unified_pipeline.fitted_pipeline is not None
        fitted_mock = unified_pipeline.fitted_pipeline["mock"]
        assert fitted_mock.fit_called


class TestUnifiedPipelineTransform:
    """Test UnifiedPipeline transform functionality."""

    def test_unified_pipeline_transform_basic(self, synthetic_clean_data):
        """Test basic transform functionality."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature", "precipitation"])

        unified_pipeline.fit(synthetic_clean_data)
        transformed = unified_pipeline.transform(synthetic_clean_data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == synthetic_clean_data.shape

        # Verify transformation occurred (values should be different)
        original_temp = synthetic_clean_data["temperature"].values
        transformed_temp = transformed["temperature"].values
        assert not np.array_equal(original_temp, transformed_temp)

        # Transformed data should have approximately zero mean (StandardScaler)
        assert abs(transformed["temperature"].mean()) < 1e-10

    def test_unified_pipeline_transform_column_processing(self, synthetic_clean_data):
        """Test column-specific processing in unified strategy."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature", "precipitation"])

        unified_pipeline.fit(synthetic_clean_data)
        transformed = unified_pipeline.transform(synthetic_clean_data)

        # Only specified columns should be transformed
        assert "temperature" in transformed.columns
        assert "precipitation" in transformed.columns
        assert "streamflow" in transformed.columns  # Should remain unchanged
        assert "gauge_id" in transformed.columns  # Should remain unchanged

        # Streamflow should be unchanged
        np.testing.assert_array_equal(transformed["streamflow"].values, synthetic_clean_data["streamflow"].values)

    def test_unified_pipeline_transform_not_fitted(self, synthetic_clean_data):
        """Test error when transform called before fitting."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature", "precipitation"])

        with pytest.raises(ValueError, match="Pipeline must be fitted before transform"):
            unified_pipeline.transform(synthetic_clean_data)

    def test_unified_pipeline_transform_missing_columns(self, synthetic_clean_data):
        """Test error when required columns missing in transform data."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature", "precipitation"])

        unified_pipeline.fit(synthetic_clean_data)

        # Remove required column
        data_missing_col = synthetic_clean_data.drop(columns=["temperature"])

        with pytest.raises(ValueError, match="Columns not found in data"):
            unified_pipeline.transform(data_missing_col)

    def test_unified_pipeline_transform_ndarray_output(self, synthetic_clean_data):
        """Test handling when pipeline returns ndarray instead of DataFrame."""
        mock_transformer = MockTransformer(multiply_factor=5)
        pipeline = Pipeline([("mock", mock_transformer)])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature", "precipitation"])

        unified_pipeline.fit(synthetic_clean_data)
        transformed = unified_pipeline.transform(synthetic_clean_data)

        assert isinstance(transformed, pd.DataFrame)

        # Verify transformation occurred (values should be multiplied by 5)
        expected_temp = synthetic_clean_data["temperature"] * 5
        np.testing.assert_array_almost_equal(transformed["temperature"].values, expected_temp.values)

    def test_unified_pipeline_transform_preserves_non_target_columns(self, synthetic_clean_data):
        """Test that non-target columns are preserved unchanged."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(
            pipeline=pipeline,
            columns=["temperature"],  # Only transform temperature
        )

        unified_pipeline.fit(synthetic_clean_data)
        transformed = unified_pipeline.transform(synthetic_clean_data)

        # Temperature should be transformed
        assert not np.array_equal(transformed["temperature"].values, synthetic_clean_data["temperature"].values)

        # Other columns should be unchanged
        np.testing.assert_array_equal(transformed["precipitation"].values, synthetic_clean_data["precipitation"].values)
        np.testing.assert_array_equal(transformed["streamflow"].values, synthetic_clean_data["streamflow"].values)

    def test_unified_pipeline_transform_single_column(self, synthetic_clean_data):
        """Test transform with single column."""
        pipeline = Pipeline([("scaler", MinMaxScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature"])

        unified_pipeline.fit(synthetic_clean_data)
        transformed = unified_pipeline.transform(synthetic_clean_data)

        # Temperature should be scaled to [0, 1] range
        assert transformed["temperature"].min() >= 0
        assert transformed["temperature"].max() <= 1.0 + 1e-10


class TestUnifiedPipelineInverseTransform:
    """Test UnifiedPipeline inverse transform functionality."""

    def test_unified_pipeline_inverse_transform_basic(self, synthetic_clean_data):
        """Test basic inverse transform functionality."""
        pipeline = Pipeline([("scaler", MinMaxScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature", "precipitation"])

        unified_pipeline.fit(synthetic_clean_data)
        transformed = unified_pipeline.transform(synthetic_clean_data)
        inverse_transformed = unified_pipeline.inverse_transform(transformed)

        # Should recover original values (within numerical precision)
        np.testing.assert_allclose(
            inverse_transformed["temperature"].values, synthetic_clean_data["temperature"].values, rtol=1e-10
        )
        np.testing.assert_allclose(
            inverse_transformed["precipitation"].values, synthetic_clean_data["precipitation"].values, rtol=1e-10
        )

    def test_unified_pipeline_inverse_transform_not_fitted(self, synthetic_clean_data):
        """Test error when inverse_transform called before fitting."""
        pipeline = Pipeline([("scaler", MinMaxScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature", "precipitation"])

        with pytest.raises(ValueError, match="Pipeline must be fitted before inverse_transform"):
            unified_pipeline.inverse_transform(synthetic_clean_data)

    def test_unified_pipeline_inverse_transform_not_supported(self, synthetic_clean_data):
        """Test error when pipeline doesn't support inverse_transform."""

        # Create a transformer without inverse_transform
        class NoInverseTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X

        pipeline = Pipeline([("no_inverse", NoInverseTransformer())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature", "precipitation"])

        unified_pipeline.fit(synthetic_clean_data)

        with pytest.raises(ValueError, match="Pipeline does not support inverse_transform"):
            unified_pipeline.inverse_transform(synthetic_clean_data)

    def test_unified_pipeline_inverse_transform_missing_columns(self, synthetic_clean_data):
        """Test error when required columns missing in inverse transform data."""
        pipeline = Pipeline([("scaler", MinMaxScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature", "precipitation"])

        unified_pipeline.fit(synthetic_clean_data)

        # Remove required column
        data_missing_col = synthetic_clean_data.drop(columns=["temperature"])

        with pytest.raises(ValueError, match="Columns not found in data"):
            unified_pipeline.inverse_transform(data_missing_col)

    def test_unified_pipeline_inverse_transform_ndarray_output(self, synthetic_clean_data):
        """Test inverse transform handling ndarray output."""
        mock_transformer = MockTransformer(multiply_factor=3)
        pipeline = Pipeline([("mock", mock_transformer)])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature", "precipitation"])

        unified_pipeline.fit(synthetic_clean_data)
        transformed = unified_pipeline.transform(synthetic_clean_data)
        inverse_transformed = unified_pipeline.inverse_transform(transformed)

        # Should recover original values
        np.testing.assert_allclose(
            inverse_transformed["temperature"].values, synthetic_clean_data["temperature"].values, rtol=1e-10
        )

    def test_unified_pipeline_inverse_transform_single_column(self, synthetic_clean_data):
        """Test inverse transform with single column."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature"])

        unified_pipeline.fit(synthetic_clean_data)
        transformed = unified_pipeline.transform(synthetic_clean_data)
        inverse_transformed = unified_pipeline.inverse_transform(transformed)

        # Should recover original temperature values
        np.testing.assert_allclose(
            inverse_transformed["temperature"].values, synthetic_clean_data["temperature"].values, rtol=1e-10
        )

        # Other columns should be unchanged
        np.testing.assert_array_equal(
            inverse_transformed["precipitation"].values, synthetic_clean_data["precipitation"].values
        )


class TestUnifiedPipelineEdgeCases:
    """Test edge cases and error scenarios."""

    def test_unified_pipeline_empty_data(self):
        """Test behavior with empty DataFrame."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature"])

        empty_df = pd.DataFrame(columns=["temperature"])

        # Should handle gracefully
        unified_pipeline.fit(empty_df)
        assert unified_pipeline.fitted_pipeline is not None

    def test_unified_pipeline_single_row(self):
        """Test behavior with single row."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature"])

        single_row_data = pd.DataFrame({"temperature": [10.0]})

        # StandardScaler might have issues with single samples
        try:
            unified_pipeline.fit(single_row_data)
            transformed = unified_pipeline.transform(single_row_data)
            assert isinstance(transformed, pd.DataFrame)
        except (ValueError, Warning):
            # Some transformers may not work with single samples, which is expected
            pass

    def test_unified_pipeline_nan_values(self, synthetic_clean_data):
        """Test behavior with NaN values."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature", "precipitation"])

        # Add some NaN values
        data_with_nans = synthetic_clean_data.copy()
        data_with_nans.loc[:10, "temperature"] = np.nan

        # StandardScaler should handle NaNs by propagating them
        unified_pipeline.fit(data_with_nans)
        transformed = unified_pipeline.transform(data_with_nans)

        # NaN values should still be NaN
        assert pd.isna(transformed.loc[:10, "temperature"]).all()

    def test_unified_pipeline_all_same_values(self):
        """Test behavior when all values in a column are the same."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature"])

        constant_data = pd.DataFrame({"temperature": [15.0] * 100})

        # StandardScaler should handle constant values
        unified_pipeline.fit(constant_data)
        transformed = unified_pipeline.transform(constant_data)

        # All transformed values should be 0 (or close to 0)
        assert np.allclose(transformed["temperature"], 0, atol=1e-10)

    def test_unified_pipeline_very_large_values(self):
        """Test behavior with very large values."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=["temperature"])

        large_values_data = pd.DataFrame({"temperature": np.random.normal(1e6, 1e5, 1000)})

        unified_pipeline.fit(large_values_data)
        transformed = unified_pipeline.transform(large_values_data)

        # Should handle large values and normalize them
        assert np.isfinite(transformed["temperature"]).all()
        assert abs(transformed["temperature"].mean()) < 1e-10

    def test_unified_pipeline_empty_columns_list(self, synthetic_clean_data):
        """Test behavior with empty columns list."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(pipeline=pipeline, columns=[])

        # Should fit without error (though not very useful)
        unified_pipeline.fit(synthetic_clean_data)
        transformed = unified_pipeline.transform(synthetic_clean_data)

        # Should return unchanged data
        pd.testing.assert_frame_equal(transformed, synthetic_clean_data)

    def test_unified_pipeline_duplicate_columns(self):
        """Test behavior with duplicate column names."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(
            pipeline=pipeline,
            columns=["temperature", "temperature"],  # Duplicate
        )

        data = pd.DataFrame({"temperature": [1, 2, 3, 4, 5]})

        # Should handle gracefully (pandas will handle duplicate column selection)
        unified_pipeline.fit(data)
        transformed = unified_pipeline.transform(data)

        assert isinstance(transformed, pd.DataFrame)

    def test_unified_pipeline_column_order_preservation(self, synthetic_clean_data):
        """Test that column order is preserved in output."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        unified_pipeline = UnifiedPipeline(
            pipeline=pipeline,
            columns=["precipitation", "temperature"],  # Different order
        )

        unified_pipeline.fit(synthetic_clean_data)
        transformed = unified_pipeline.transform(synthetic_clean_data)

        # Column order should be preserved from input DataFrame
        assert list(transformed.columns) == list(synthetic_clean_data.columns)
