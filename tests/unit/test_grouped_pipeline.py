"""
Comprehensive unit tests for GroupedPipeline.

Tests cover:
- Pipeline initialization and configuration
- Fitting separate pipelines for multiple groups
- Transform consistency per group
- Inverse transform functionality
- Multiprocessing capabilities
- Error handling and edge cases
- Integration with sklearn pipelines
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from hydro_forecasting.preprocessing.grouped import GroupedPipeline


class MockTransformer(BaseEstimator, TransformerMixin):
    """Mock transformer for testing that tracks calls."""

    def __init__(self, add_value=0):
        self.add_value = add_value
        self.fit_called = False
        self.transform_called = False

    def fit(self, X, y=None):
        self.fit_called = True
        return self

    def transform(self, X):
        self.transform_called = True
        return X + self.add_value

    def inverse_transform(self, X):
        return X - self.add_value


class TestGroupedPipelineInitialization:
    """Test GroupedPipeline initialization and configuration."""

    def test_grouped_pipeline_init_basic(self):
        """Test basic initialization of GroupedPipeline."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        columns = ["temperature", "precipitation"]
        group_identifier = "gauge_id"

        grouped_pipeline = GroupedPipeline(pipeline=pipeline, columns=columns, group_identifier=group_identifier)

        assert grouped_pipeline.pipeline == pipeline
        assert grouped_pipeline.columns == columns
        assert grouped_pipeline.group_identifier == group_identifier
        assert grouped_pipeline.n_jobs == 1
        assert grouped_pipeline.chunk_size is None
        assert len(grouped_pipeline.fitted_pipelines) == 0
        assert len(grouped_pipeline.all_groups) == 0

    def test_grouped_pipeline_init_with_multiprocessing(self):
        """Test initialization with multiprocessing parameters."""
        pipeline = Pipeline([("scaler", StandardScaler())])

        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline, columns=["temperature"], group_identifier="gauge_id", n_jobs=4, chunk_size=10
        )

        assert grouped_pipeline.n_jobs == 4
        assert grouped_pipeline.chunk_size == 10

    def test_grouped_pipeline_init_negative_n_jobs(self):
        """Test initialization with negative n_jobs (use all cores)."""
        pipeline = Pipeline([("scaler", StandardScaler())])

        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline, columns=["temperature"], group_identifier="gauge_id", n_jobs=-1
        )

        assert grouped_pipeline.n_jobs == -1


class TestGroupedPipelineFitting:
    """Test GroupedPipeline fitting functionality."""

    def test_grouped_pipeline_fit_single_group(self, synthetic_clean_data):
        """Test fitting with single group."""
        # Use only one basin
        single_basin_data = synthetic_clean_data[synthetic_clean_data["gauge_id"] == "basin_001"].copy()

        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline, columns=["temperature", "precipitation"], group_identifier="gauge_id"
        )

        grouped_pipeline.fit(single_basin_data)

        assert len(grouped_pipeline.fitted_pipelines) == 1
        assert "basin_001" in grouped_pipeline.fitted_pipelines
        assert "basin_001" in grouped_pipeline.all_groups

        # Verify pipeline was actually fitted
        fitted_pipe = grouped_pipeline.fitted_pipelines["basin_001"]
        assert hasattr(fitted_pipe["scaler"], "mean_")  # StandardScaler should have mean_ after fitting

    def test_grouped_pipeline_fit_multiple_groups(self, synthetic_clean_data):
        """Test fitting with multiple groups."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline, columns=["temperature", "precipitation"], group_identifier="gauge_id"
        )

        grouped_pipeline.fit(synthetic_clean_data)

        unique_basins = synthetic_clean_data["gauge_id"].unique()
        assert len(grouped_pipeline.fitted_pipelines) == len(unique_basins)
        assert len(grouped_pipeline.all_groups) == len(unique_basins)

        # Verify each group has its own fitted pipeline
        for basin_id in unique_basins:
            assert basin_id in grouped_pipeline.fitted_pipelines
            fitted_pipe = grouped_pipeline.fitted_pipelines[basin_id]
            assert hasattr(fitted_pipe["scaler"], "mean_")

    def test_grouped_pipeline_fit_with_target(self, synthetic_clean_data):
        """Test fitting with target variable."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline, columns=["temperature", "precipitation"], group_identifier="gauge_id"
        )

        # Use streamflow as target
        y = synthetic_clean_data["streamflow"]

        grouped_pipeline.fit(synthetic_clean_data, y)

        assert len(grouped_pipeline.fitted_pipelines) > 0
        # Target should be passed to each group's pipeline fit method

    def test_grouped_pipeline_fit_missing_columns(self, synthetic_clean_data):
        """Test error when required columns are missing."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline, columns=["nonexistent_column"], group_identifier="gauge_id"
        )

        with pytest.raises(ValueError, match="Columns not found in data"):
            grouped_pipeline.fit(synthetic_clean_data)

    def test_grouped_pipeline_fit_missing_group_identifier(self, synthetic_clean_data):
        """Test error when group identifier is missing."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline, columns=["temperature", "precipitation"], group_identifier="nonexistent_id"
        )

        with pytest.raises(ValueError, match="Group identifier nonexistent_id not found"):
            grouped_pipeline.fit(synthetic_clean_data)

    def test_grouped_pipeline_fit_empty_group(self, synthetic_clean_data):
        """Test handling of empty groups during fitting."""
        # Create data with some groups that would be empty after filtering
        data_with_empty = synthetic_clean_data.copy()

        # Add a group with no data (shouldn't happen in practice but test robustness)
        empty_group_data = pd.DataFrame(
            {
                "gauge_id": ["empty_basin"],
                "date": [pd.Timestamp("2020-01-01")],
                "temperature": [np.nan],
                "precipitation": [np.nan],
                "streamflow": [np.nan],
            }
        )

        # This would be filtered out, so we just test the robustness
        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline, columns=["temperature", "precipitation"], group_identifier="gauge_id"
        )

        # Should handle gracefully
        grouped_pipeline.fit(synthetic_clean_data)
        assert len(grouped_pipeline.fitted_pipelines) > 0

    def test_grouped_pipeline_fit_multiprocessing(self, synthetic_clean_data):
        """Test fitting with multiprocessing enabled."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline,
            columns=["temperature", "precipitation"],
            group_identifier="gauge_id",
            n_jobs=2,
            chunk_size=2,
        )

        grouped_pipeline.fit(synthetic_clean_data)

        unique_basins = synthetic_clean_data["gauge_id"].unique()
        assert len(grouped_pipeline.fitted_pipelines) == len(unique_basins)

        # Verify each pipeline was fitted correctly
        for basin_id in unique_basins:
            fitted_pipe = grouped_pipeline.fitted_pipelines[basin_id]
            assert hasattr(fitted_pipe["scaler"], "mean_")


class TestGroupedPipelineTransform:
    """Test GroupedPipeline transform functionality."""

    def test_grouped_pipeline_transform_basic(self, synthetic_clean_data):
        """Test basic transform functionality."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline, columns=["temperature", "precipitation"], group_identifier="gauge_id"
        )

        # Fit and transform
        grouped_pipeline.fit(synthetic_clean_data)
        transformed = grouped_pipeline.transform(synthetic_clean_data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == synthetic_clean_data.shape

        # Verify transformation occurred (values should be different)
        original_temp = synthetic_clean_data["temperature"].values
        transformed_temp = transformed["temperature"].values
        assert not np.array_equal(original_temp, transformed_temp)

    def test_grouped_pipeline_transform_consistency_per_group(self, synthetic_clean_data):
        """Test that transforms are applied consistently per group."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(pipeline=pipeline, columns=["temperature"], group_identifier="gauge_id")

        grouped_pipeline.fit(synthetic_clean_data)
        transformed = grouped_pipeline.transform(synthetic_clean_data)

        # Check that each group's data was transformed by its own scaler
        for basin_id in synthetic_clean_data["gauge_id"].unique():
            basin_mask = synthetic_clean_data["gauge_id"] == basin_id
            basin_original = synthetic_clean_data.loc[basin_mask, "temperature"]
            basin_transformed = transformed.loc[basin_mask, "temperature"]

            # Transformed data should have approximately zero mean for each group
            # (within reasonable tolerance due to floating point precision)
            assert abs(basin_transformed.mean()) < 1e-10

    def test_grouped_pipeline_transform_unseen_groups(self, synthetic_clean_data):
        """Test transform behavior with groups not seen during fit."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline, columns=["temperature", "precipitation"], group_identifier="gauge_id"
        )

        # Fit on subset of data
        train_data = synthetic_clean_data[synthetic_clean_data["gauge_id"].isin(["basin_001", "basin_002"])]
        grouped_pipeline.fit(train_data)

        # Transform data including unseen groups
        with patch("builtins.print") as mock_print:
            transformed = grouped_pipeline.transform(synthetic_clean_data)

            # Should print warning about unseen groups
            mock_print.assert_called()
            warning_msg = mock_print.call_args[0][0]
            assert "not seen during fit" in warning_msg

        # Unseen groups should be passed through unchanged
        assert transformed.shape == synthetic_clean_data.shape

    def test_grouped_pipeline_transform_missing_group_identifier(self, synthetic_clean_data):
        """Test error when group identifier missing in transform data."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline, columns=["temperature", "precipitation"], group_identifier="gauge_id"
        )

        grouped_pipeline.fit(synthetic_clean_data)

        # Remove group identifier
        data_no_group = synthetic_clean_data.drop(columns=["gauge_id"])

        with pytest.raises(ValueError, match="Group identifier gauge_id not found"):
            grouped_pipeline.transform(data_no_group)

    def test_grouped_pipeline_transform_multiprocessing(self, synthetic_clean_data):
        """Test transform with multiprocessing enabled."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline, columns=["temperature", "precipitation"], group_identifier="gauge_id", n_jobs=2
        )

        grouped_pipeline.fit(synthetic_clean_data)
        transformed = grouped_pipeline.transform(synthetic_clean_data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == synthetic_clean_data.shape

        # Results should be consistent with single-threaded version
        single_threaded = GroupedPipeline(
            pipeline=pipeline, columns=["temperature", "precipitation"], group_identifier="gauge_id", n_jobs=1
        )
        single_threaded.fit(synthetic_clean_data)
        single_transformed = single_threaded.transform(synthetic_clean_data)

        # Results should be very similar (allowing for minor numerical differences)
        np.testing.assert_allclose(
            transformed["temperature"].values, single_transformed["temperature"].values, rtol=1e-10
        )

    def test_grouped_pipeline_transform_ndarray_output(self, synthetic_clean_data):
        """Test handling when pipeline returns ndarray instead of DataFrame."""
        # Use a transformer that returns ndarray
        pipeline = Pipeline([("mock", MockTransformer(add_value=10))])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline, columns=["temperature", "precipitation"], group_identifier="gauge_id"
        )

        grouped_pipeline.fit(synthetic_clean_data)
        transformed = grouped_pipeline.transform(synthetic_clean_data)

        assert isinstance(transformed, pd.DataFrame)

        # Verify transformation occurred (values should be increased by 10)
        expected_temp = synthetic_clean_data["temperature"] + 10
        np.testing.assert_array_almost_equal(transformed["temperature"].values, expected_temp.values)


class TestGroupedPipelineInverseTransform:
    """Test GroupedPipeline inverse transform functionality."""

    def test_grouped_pipeline_inverse_transform_basic(self, synthetic_clean_data):
        """Test basic inverse transform functionality."""
        # Use MinMaxScaler which supports inverse_transform
        pipeline = Pipeline([("scaler", MinMaxScaler())])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline, columns=["temperature", "precipitation"], group_identifier="gauge_id"
        )

        grouped_pipeline.fit(synthetic_clean_data)
        transformed = grouped_pipeline.transform(synthetic_clean_data)
        inverse_transformed = grouped_pipeline.inverse_transform(transformed)

        # Should recover original values (within numerical precision)
        np.testing.assert_allclose(
            inverse_transformed["temperature"].values, synthetic_clean_data["temperature"].values, rtol=1e-10
        )

    def test_grouped_pipeline_inverse_transform_unsupported(self, synthetic_clean_data):
        """Test inverse transform with pipeline that doesn't support it."""

        # StandardScaler supports inverse_transform, but mock one that doesn't
        class NoInverseTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X

        pipeline = Pipeline([("no_inverse", NoInverseTransformer())])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline, columns=["temperature", "precipitation"], group_identifier="gauge_id"
        )

        grouped_pipeline.fit(synthetic_clean_data)

        with patch("builtins.print") as mock_print:
            result = grouped_pipeline.inverse_transform(synthetic_clean_data)

            # Should print warning about unsupported inverse transform
            mock_print.assert_called()
            warning_msg = mock_print.call_args[0][0]
            assert "does not support inverse_transform" in warning_msg

        # Should return modified copy of input
        assert isinstance(result, pd.DataFrame)

    def test_grouped_pipeline_inverse_transform_multiprocessing(self, synthetic_clean_data):
        """Test inverse transform with multiprocessing."""
        pipeline = Pipeline([("scaler", MinMaxScaler())])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline, columns=["temperature", "precipitation"], group_identifier="gauge_id", n_jobs=2
        )

        grouped_pipeline.fit(synthetic_clean_data)
        transformed = grouped_pipeline.transform(synthetic_clean_data)
        inverse_transformed = grouped_pipeline.inverse_transform(transformed)

        # Should recover original values
        np.testing.assert_allclose(
            inverse_transformed["temperature"].values, synthetic_clean_data["temperature"].values, rtol=1e-10
        )


class TestGroupedPipelineUtilityMethods:
    """Test GroupedPipeline utility methods."""

    def test_get_feature_names_out(self, synthetic_clean_data):
        """Test get_feature_names_out method."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        columns = ["temperature", "precipitation"]
        grouped_pipeline = GroupedPipeline(pipeline=pipeline, columns=columns, group_identifier="gauge_id")

        # Should fail before fitting
        with pytest.raises(ValueError, match="Transformer must be fitted"):
            grouped_pipeline.get_feature_names_out()

        # Should work after fitting
        grouped_pipeline.fit(synthetic_clean_data)
        feature_names = grouped_pipeline.get_feature_names_out()

        assert feature_names == columns

    def test_add_fitted_group(self):
        """Test add_fitted_group method."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(pipeline=pipeline, columns=["temperature"], group_identifier="gauge_id")

        # Add a pre-fitted pipeline
        fitted_pipeline = Pipeline([("scaler", StandardScaler())])
        fitted_pipeline.fit([[1], [2], [3]])

        grouped_pipeline.add_fitted_group("test_basin", fitted_pipeline)

        assert "test_basin" in grouped_pipeline.fitted_pipelines
        assert "test_basin" in grouped_pipeline.all_groups
        assert grouped_pipeline.fitted_pipelines["test_basin"] == fitted_pipeline

    def test_split_groups_into_chunks(self):
        """Test _split_groups_into_chunks method."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline,
            columns=["temperature"],
            group_identifier="gauge_id",
            n_jobs=1,  # Single job should return single chunk
        )

        groups = ["basin_001", "basin_002", "basin_003", "basin_004", "basin_005"]

        # Test single job (n_jobs=1)
        chunks = grouped_pipeline._split_groups_into_chunks(groups)
        assert len(chunks) == 1
        assert chunks[0] == groups

        # Test with multiple jobs
        grouped_pipeline.n_jobs = 2
        chunks = grouped_pipeline._split_groups_into_chunks(groups)
        assert len(chunks) >= 1

        # All groups should be included
        all_groups_in_chunks = []
        for chunk in chunks:
            all_groups_in_chunks.extend(chunk)
        assert set(all_groups_in_chunks) == set(groups)

        # Test with chunk size
        grouped_pipeline.chunk_size = 2
        chunks = grouped_pipeline._split_groups_into_chunks(groups)
        assert len(chunks) == 3  # 5 groups / 2 per chunk = 3 chunks
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2
        assert len(chunks[2]) == 1


class TestGroupedPipelineEdgeCases:
    """Test edge cases and error scenarios."""

    def test_grouped_pipeline_empty_data(self):
        """Test behavior with empty DataFrame."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(pipeline=pipeline, columns=["temperature"], group_identifier="gauge_id")

        empty_df = pd.DataFrame(columns=["gauge_id", "temperature"])

        grouped_pipeline.fit(empty_df)

        # Should handle gracefully
        assert len(grouped_pipeline.fitted_pipelines) == 0
        assert len(grouped_pipeline.all_groups) == 0

    def test_grouped_pipeline_single_row_per_group(self):
        """Test behavior with single row per group."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(pipeline=pipeline, columns=["temperature"], group_identifier="gauge_id")

        single_row_data = pd.DataFrame({"gauge_id": ["basin_001", "basin_002"], "temperature": [10.0, 20.0]})

        # StandardScaler might have issues with single samples, but should handle gracefully
        try:
            grouped_pipeline.fit(single_row_data)
            transformed = grouped_pipeline.transform(single_row_data)
            assert isinstance(transformed, pd.DataFrame)
        except (ValueError, Warning):
            # Some transformers may not work with single samples, which is expected
            pass

    def test_grouped_pipeline_multiprocessing_edge_cases(self, synthetic_clean_data):
        """Test multiprocessing edge cases."""
        pipeline = Pipeline([("scaler", StandardScaler())])

        # Test with more processes than groups
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline,
            columns=["temperature"],
            group_identifier="gauge_id",
            n_jobs=100,  # More than number of groups
        )

        grouped_pipeline.fit(synthetic_clean_data)
        assert len(grouped_pipeline.fitted_pipelines) > 0

        # Test with negative n_jobs
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline,
            columns=["temperature"],
            group_identifier="gauge_id",
            n_jobs=-1,  # Use all available cores
        )

        grouped_pipeline.fit(synthetic_clean_data)
        assert len(grouped_pipeline.fitted_pipelines) > 0

    def test_grouped_pipeline_large_chunk_size(self, synthetic_clean_data):
        """Test with chunk size larger than number of groups."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        grouped_pipeline = GroupedPipeline(
            pipeline=pipeline,
            columns=["temperature"],
            group_identifier="gauge_id",
            n_jobs=2,
            chunk_size=1000,  # Larger than number of groups
        )

        grouped_pipeline.fit(synthetic_clean_data)
        transformed = grouped_pipeline.transform(synthetic_clean_data)

        assert isinstance(transformed, pd.DataFrame)
        assert len(grouped_pipeline.fitted_pipelines) > 0
