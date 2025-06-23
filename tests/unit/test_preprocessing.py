"""
Comprehensive unit tests for the main preprocessing module.

Tests cover:
- create_pipeline() factory function for GroupedPipeline and UnifiedPipeline
- split_data() function with various proportions and edge cases
- batch_basins() utility function
- load_basins_timeseries_lazy() data loading function
- write_train_val_test_splits_to_disk() file operations
- batch_process_time_series_data() batch processing logic
- ProcessingConfig and ProcessingOutput dataclasses
- Error handling and configuration validation
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from hydro_forecasting.data.preprocessing import (
    ProcessingConfig,
    ProcessingOutput,
    batch_basins,
    batch_process_time_series_data,
    create_pipeline,
    load_basins_timeseries_lazy,
    split_data,
    write_train_val_test_splits_to_disk,
)
from hydro_forecasting.exceptions import (
    ConfigurationError,
    DataProcessingError,
    DataQualityError,
    FileOperationError,
)
from hydro_forecasting.preprocessing.grouped import GroupedPipeline
from hydro_forecasting.preprocessing.unified import UnifiedPipeline


class TestCreatePipeline:
    """Test the create_pipeline factory function."""

    def test_create_pipeline_grouped_strategy(self):
        """Test creating GroupedPipeline with per_group strategy."""
        grouped_pipeline = GroupedPipeline(
            pipeline=Pipeline([("scaler", StandardScaler())]),
            columns=["temperature", "precipitation"],
            group_identifier="gauge_id"
        )
        
        config = {
            "strategy": "per_group",
            "pipeline": grouped_pipeline,
            "columns": ["temperature", "precipitation"]
        }
        
        result = create_pipeline(config)
        
        assert isinstance(result, GroupedPipeline)
        assert result.columns == ["temperature", "precipitation"]
        assert result.group_identifier == "gauge_id"

    def test_create_pipeline_unified_strategy(self):
        """Test creating UnifiedPipeline with unified strategy."""
        sklearn_pipeline = Pipeline([("scaler", MinMaxScaler())])
        
        config = {
            "strategy": "unified",
            "pipeline": sklearn_pipeline,
            "columns": ["temperature", "precipitation"]
        }
        
        result = create_pipeline(config)
        
        assert isinstance(result, UnifiedPipeline)
        assert result.columns == ["temperature", "precipitation"]

    def test_create_pipeline_default_strategy(self):
        """Test default strategy (per_group) when not specified."""
        grouped_pipeline = GroupedPipeline(
            pipeline=Pipeline([("scaler", StandardScaler())]),
            columns=["temperature"],
            group_identifier="gauge_id"
        )
        
        config = {
            "pipeline": grouped_pipeline,
            "columns": ["temperature"]
        }
        
        result = create_pipeline(config)
        assert isinstance(result, GroupedPipeline)

    def test_create_pipeline_missing_pipeline_key(self):
        """Test error when pipeline key is missing."""
        config = {
            "strategy": "unified",
            "columns": ["temperature"]
        }
        
        with pytest.raises(ConfigurationError, match="Pipeline configuration must include 'pipeline' key"):
            create_pipeline(config)

    def test_create_pipeline_invalid_strategy(self):
        """Test error with unknown strategy."""
        config = {
            "strategy": "invalid_strategy",
            "pipeline": Pipeline([("scaler", StandardScaler())]),
            "columns": ["temperature"]
        }
        
        with pytest.raises(ConfigurationError, match="Unknown strategy: invalid_strategy"):
            create_pipeline(config)

    def test_create_pipeline_wrong_pipeline_type_for_grouped(self):
        """Test error when sklearn Pipeline provided for per_group strategy."""
        config = {
            "strategy": "per_group",
            "pipeline": Pipeline([("scaler", StandardScaler())]),  # Should be GroupedPipeline
            "columns": ["temperature"]
        }
        
        with pytest.raises(ConfigurationError, match="Strategy 'per_group' requires GroupedPipeline"):
            create_pipeline(config)

    def test_create_pipeline_wrong_pipeline_type_for_unified(self):
        """Test error when GroupedPipeline provided for unified strategy."""
        grouped_pipeline = GroupedPipeline(
            pipeline=Pipeline([("scaler", StandardScaler())]),
            columns=["temperature"],
            group_identifier="gauge_id"
        )
        
        config = {
            "strategy": "unified",
            "pipeline": grouped_pipeline,  # Should be sklearn Pipeline
            "columns": ["temperature"]
        }
        
        with pytest.raises(ConfigurationError, match="Strategy 'unified' requires sklearn Pipeline"):
            create_pipeline(config)


class TestProcessingConfig:
    """Test ProcessingConfig dataclass."""

    def test_processing_config_defaults(self):
        """Test ProcessingConfig with default values."""
        config = ProcessingConfig(required_columns=["streamflow"])
        
        assert config.required_columns == ["streamflow"]
        assert config.min_train_years == 5.0
        assert config.max_imputation_gap_size == 5
        assert config.group_identifier == "gauge_id"
        assert config.train_prop == 0.6
        assert config.val_prop == 0.2
        assert config.test_prop == 0.2

    def test_processing_config_custom_values(self):
        """Test ProcessingConfig with custom values."""
        config = ProcessingConfig(
            required_columns=["temp", "precip", "flow"],
            min_train_years=3.0,
            max_imputation_gap_size=10,
            group_identifier="basin_id",
            train_prop=0.7,
            val_prop=0.15,
            test_prop=0.15
        )
        
        assert config.required_columns == ["temp", "precip", "flow"]
        assert config.min_train_years == 3.0
        assert config.max_imputation_gap_size == 10
        assert config.group_identifier == "basin_id"
        assert config.train_prop == 0.7
        assert config.val_prop == 0.15
        assert config.test_prop == 0.15


class TestSplitData:
    """Test the split_data function."""

    def test_split_data_standard_proportions(self, synthetic_clean_data, processing_config):
        """Test data splitting with standard proportions."""
        df = pl.from_pandas(synthetic_clean_data)
        
        train_df, val_df, test_df = split_data(df, processing_config)
        
        assert isinstance(train_df, pl.DataFrame)
        assert isinstance(val_df, pl.DataFrame)
        assert isinstance(test_df, pl.DataFrame)
        
        # Check that all splits contain data
        assert train_df.height > 0
        assert val_df.height > 0
        assert test_df.height > 0
        
        # Check proportions are approximately correct (allowing for rounding)
        total_rows = train_df.height + val_df.height + test_df.height
        train_ratio = train_df.height / total_rows
        val_ratio = val_df.height / total_rows
        test_ratio = test_df.height / total_rows
        
        assert abs(train_ratio - processing_config.train_prop) < 0.1
        assert abs(val_ratio - processing_config.val_prop) < 0.1
        assert abs(test_ratio - processing_config.test_prop) < 0.1

    def test_split_data_custom_proportions(self, synthetic_clean_data):
        """Test data splitting with custom proportions."""
        config = ProcessingConfig(
            required_columns=["temperature", "precipitation", "streamflow"],
            train_prop=0.8,
            val_prop=0.1,
            test_prop=0.1
        )
        
        df = pl.from_pandas(synthetic_clean_data)
        train_df, val_df, test_df = split_data(df, config)
        
        # Training set should be much larger
        assert train_df.height > val_df.height
        assert train_df.height > test_df.height

    def test_split_data_missing_group_identifier(self, synthetic_clean_data, processing_config):
        """Test split_data when group identifier is missing."""
        # Remove group identifier column
        df = pl.from_pandas(synthetic_clean_data.drop(columns=['gauge_id']))
        
        train_df, val_df, test_df = split_data(df, processing_config)
        
        # Should still work by creating a single group
        assert train_df.height > 0
        assert val_df.height > 0
        assert test_df.height > 0

    def test_split_data_missing_target_column(self, synthetic_clean_data, processing_config):
        """Test split_data when target column is missing."""
        # Properly set up config with missing target column
        if not hasattr(processing_config, 'preprocessing_config') or processing_config.preprocessing_config is None:
            processing_config.preprocessing_config = {}
        processing_config.preprocessing_config["target"] = {"column": "nonexistent"}
        
        df = pl.from_pandas(synthetic_clean_data)
        
        # This should raise an error because the nonexistent column doesn't exist
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            train_df, val_df, test_df = split_data(df, config=processing_config)

    def test_split_data_no_valid_target_data(self, processing_config):
        """Test split_data when all target data is null."""
        # Create data with all null streamflow
        data = pd.DataFrame({
            'gauge_id': ['basin_001'] * 100,
            'date': pd.date_range('2020-01-01', periods=100),
            'temperature': np.random.randn(100),
            'precipitation': np.random.randn(100),
            'streamflow': [np.nan] * 100
        })
        
        df = pl.from_pandas(data)
        train_df, val_df, test_df = split_data(df, processing_config)
        
        # Should return empty DataFrames
        assert train_df.height == 0
        assert val_df.height == 0
        assert test_df.height == 0

    def test_split_data_multiple_basins(self, synthetic_clean_data, processing_config):
        """Test split_data with multiple basins."""
        df = pl.from_pandas(synthetic_clean_data)
        train_df, val_df, test_df = split_data(df, processing_config)
        
        # Check that all basins are represented in splits
        unique_basins = df.select(pl.col("gauge_id")).unique().to_series().to_list()
        
        train_basins = train_df.select(pl.col("gauge_id")).unique().to_series().to_list()
        val_basins = val_df.select(pl.col("gauge_id")).unique().to_series().to_list()
        test_basins = test_df.select(pl.col("gauge_id")).unique().to_series().to_list()
        
        # Each basin should appear in all splits (assuming sufficient data)
        assert len(train_basins) == len(unique_basins)
        assert len(val_basins) == len(unique_basins)
        assert len(test_basins) == len(unique_basins)

    @pytest.mark.parametrize("split_proportions", [
        {"train_prop": 0.6, "val_prop": 0.2, "test_prop": 0.2},
        {"train_prop": 0.7, "val_prop": 0.15, "test_prop": 0.15},
        {"train_prop": 0.8, "val_prop": 0.1, "test_prop": 0.1}
    ])
    def test_split_data_various_proportions(self, synthetic_clean_data, split_proportions):
        """Test split_data with various proportion combinations."""
        config = ProcessingConfig(
            required_columns=["temperature", "precipitation", "streamflow"],
            **split_proportions
        )
        
        df = pl.from_pandas(synthetic_clean_data)
        train_df, val_df, test_df = split_data(df, config)
        
        assert train_df.height > 0
        assert val_df.height > 0
        assert test_df.height > 0


class TestBatchBasins:
    """Test the batch_basins utility function."""

    def test_batch_basins_normal_case(self):
        """Test batch_basins with normal input."""
        basins = ["basin_001", "basin_002", "basin_003", "basin_004", "basin_005"]
        batch_size = 2
        
        batches = list(batch_basins(basins, batch_size))
        
        assert len(batches) == 3  # 5 basins / 2 = 3 batches
        assert batches[0] == ["basin_001", "basin_002"]
        assert batches[1] == ["basin_003", "basin_004"]
        assert batches[2] == ["basin_005"]

    def test_batch_basins_exact_division(self):
        """Test batch_basins when list divides exactly."""
        basins = ["basin_001", "basin_002", "basin_003", "basin_004"]
        batch_size = 2
        
        batches = list(batch_basins(basins, batch_size))
        
        assert len(batches) == 2
        assert batches[0] == ["basin_001", "basin_002"]
        assert batches[1] == ["basin_003", "basin_004"]

    def test_batch_basins_single_batch(self):
        """Test batch_basins when batch_size >= list length."""
        basins = ["basin_001", "basin_002"]
        batch_size = 5
        
        batches = list(batch_basins(basins, batch_size))
        
        assert len(batches) == 1
        assert batches[0] == ["basin_001", "basin_002"]

    def test_batch_basins_empty_list(self):
        """Test batch_basins with empty list."""
        basins = []
        batch_size = 2
        
        batches = list(batch_basins(basins, batch_size))
        
        assert len(batches) == 0

    def test_batch_basins_invalid_batch_size(self):
        """Test batch_basins with invalid batch size."""
        basins = ["basin_001", "basin_002"]
        batch_size = 0
        
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            list(batch_basins(basins, batch_size))

    def test_batch_basins_negative_batch_size(self):
        """Test batch_basins with negative batch size."""
        basins = ["basin_001", "basin_002"]
        batch_size = -1
        
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            list(batch_basins(basins, batch_size))


class TestLoadBasinsTimeseriesLazy:
    """Test the load_basins_timeseries_lazy function."""

    def test_load_basins_timeseries_lazy_success(self, create_basin_files, basin_ids):
        """Test successful loading of basin timeseries data."""
        region_dirs = create_basin_files
        required_columns = ["temperature", "precipitation", "streamflow"]
        
        lf = load_basins_timeseries_lazy(
            gauge_ids=basin_ids,
            region_time_series_base_dirs=region_dirs,
            required_columns=required_columns
        )
        
        assert isinstance(lf, pl.LazyFrame)
        
        # Collect and verify data
        df = lf.collect()
        assert df.height > 0
        assert "gauge_id" in df.columns
        assert "date" in df.columns
        for col in required_columns:
            assert col in df.columns

    def test_load_basins_timeseries_lazy_empty_gauge_ids(self, region_dirs):
        """Test error with empty gauge IDs list."""
        required_columns = ["temperature", "precipitation", "streamflow"]
        
        with pytest.raises(ConfigurationError, match="No gauge IDs provided"):
            load_basins_timeseries_lazy(
                gauge_ids=[],
                region_time_series_base_dirs=region_dirs["time_series"],
                required_columns=required_columns
            )

    def test_load_basins_timeseries_lazy_file_not_found(self, region_dirs):
        """Test error when basin file doesn't exist."""
        required_columns = ["temperature", "precipitation", "streamflow"]
        nonexistent_ids = ["nonexistent_basin"]
        
        with pytest.raises(FileOperationError, match="No base directory for region prefix"):
            load_basins_timeseries_lazy(
                gauge_ids=nonexistent_ids,
                region_time_series_base_dirs=region_dirs["time_series"],
                required_columns=required_columns
            )

    def test_load_basins_timeseries_lazy_missing_region(self, region_dirs):
        """Test error when region prefix has no base directory."""
        required_columns = ["temperature", "precipitation", "streamflow"]
        invalid_ids = ["invalid_region_basin_001"]
        
        with pytest.raises(FileOperationError, match="No base directory for region prefix"):
            load_basins_timeseries_lazy(
                gauge_ids=invalid_ids,
                region_time_series_base_dirs=region_dirs["time_series"],
                required_columns=required_columns
            )

    def test_load_basins_timeseries_lazy_missing_columns(self, create_basin_files, basin_ids):
        """Test error when required columns are missing."""
        region_dirs = create_basin_files
        required_columns = ["temperature", "precipitation", "nonexistent_column"]
        
        with pytest.raises(DataQualityError, match="Missing required columns"):
            load_basins_timeseries_lazy(
                gauge_ids=basin_ids[:1],  # Test with one basin
                region_time_series_base_dirs=region_dirs,
                required_columns=required_columns
            )

    def test_load_basins_timeseries_lazy_custom_group_identifier(self, create_basin_files, basin_ids):
        """Test loading with custom group identifier."""
        region_dirs = create_basin_files
        required_columns = ["temperature", "precipitation", "streamflow"]
        
        lf = load_basins_timeseries_lazy(
            gauge_ids=basin_ids,
            region_time_series_base_dirs=region_dirs,
            required_columns=required_columns,
            group_identifier="custom_id"
        )
        
        df = lf.collect()
        assert "custom_id" in df.columns
        assert set(df["custom_id"].unique().to_list()) == set(basin_ids)


class TestWriteTrainValTestSplitsToDisk:
    """Test the write_train_val_test_splits_to_disk function."""

    def test_write_splits_to_disk_success(self, temp_dir, synthetic_clean_data, processing_config):
        """Test successful writing of splits to disk."""
        df = pl.from_pandas(synthetic_clean_data)
        train_df, val_df, test_df = split_data(df, processing_config)
        
        write_train_val_test_splits_to_disk(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            output_dir=temp_dir,
            basin_ids=["basin_001", "basin_002"]
        )
        
        # Check directory structure
        assert (temp_dir / "train").exists()
        assert (temp_dir / "val").exists()
        assert (temp_dir / "test").exists()
        
        # Check that files were created
        assert (temp_dir / "train" / "basin_001.parquet").exists()
        assert (temp_dir / "val" / "basin_001.parquet").exists()
        assert (temp_dir / "test" / "basin_001.parquet").exists()

    def test_write_splits_to_disk_infer_basin_ids(self, temp_dir, synthetic_clean_data, processing_config):
        """Test inferring basin IDs from data when not provided."""
        df = pl.from_pandas(synthetic_clean_data)
        train_df, val_df, test_df = split_data(df, processing_config)
        
        write_train_val_test_splits_to_disk(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            output_dir=temp_dir
            # basin_ids not provided - should be inferred
        )
        
        # Should still create files for all basins
        expected_basins = train_df.select(pl.col("gauge_id")).unique().to_series().to_list()
        for basin_id in expected_basins:
            assert (temp_dir / "train" / f"{basin_id}.parquet").exists()

    def test_write_splits_to_disk_empty_dataframes(self, temp_dir):
        """Test writing empty DataFrames."""
        empty_df = pl.DataFrame(schema={"gauge_id": pl.Utf8, "date": pl.Date, "value": pl.Float64})
        
        # Should handle empty DataFrames gracefully - function will return early and not create directories
        write_train_val_test_splits_to_disk(
            train_df=empty_df,
            val_df=empty_df,
            test_df=empty_df,
            output_dir=temp_dir
        )
        
        # With empty dataframes, function should return early and not create directories
        # This is expected behavior based on the implementation

    def test_write_splits_to_disk_custom_compression(self, temp_dir, synthetic_clean_data, processing_config):
        """Test writing with custom compression."""
        df = pl.from_pandas(synthetic_clean_data)
        train_df, val_df, test_df = split_data(df, processing_config)
        
        write_train_val_test_splits_to_disk(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            output_dir=temp_dir,
            compression="gzip"
        )
        
        # Files should still be created (compression doesn't affect file names)
        train_files = list((temp_dir / "train").glob("*.parquet"))
        assert len(train_files) > 0

    def test_write_splits_to_disk_file_operation_error(self, synthetic_clean_data, processing_config):
        """Test error handling for file operation failures."""
        df = pl.from_pandas(synthetic_clean_data)
        train_df, val_df, test_df = split_data(df, processing_config)
        
        # Try to write to invalid path
        invalid_path = "/invalid/path/that/does/not/exist"
        
        with pytest.raises(FileOperationError, match="Error writing splits to disk"):
            write_train_val_test_splits_to_disk(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                output_dir=invalid_path
            )


class TestBatchProcessTimeSeriesData:
    """Test the batch_process_time_series_data function."""

    def test_batch_process_time_series_data_success(self, synthetic_clean_data, processing_config):
        """Test successful batch processing of time series data."""
        lf = pl.from_pandas(synthetic_clean_data).lazy()
        
        # Create simple pipelines
        features_pipeline = GroupedPipeline(
            pipeline=Pipeline([("scaler", StandardScaler())]),
            columns=["temperature", "precipitation"],
            group_identifier="gauge_id"
        )
        
        target_pipeline = UnifiedPipeline(
            pipeline=Pipeline([("scaler", MinMaxScaler())]),
            columns=["streamflow"]
        )
        
        # Fit the target pipeline before use
        target_pipeline.fit(synthetic_clean_data[["streamflow"]])
        
        result = batch_process_time_series_data(
            lf=lf,
            config=processing_config,
            features_pipeline=features_pipeline,
            target_pipeline=target_pipeline
        )
        
        train_df, val_df, test_df, fitted_pipelines, quality_reports = result
        
        assert isinstance(train_df, pl.DataFrame)
        assert isinstance(val_df, pl.DataFrame)
        assert isinstance(test_df, pl.DataFrame)
        assert isinstance(fitted_pipelines, dict)
        assert isinstance(quality_reports, dict)
        
        # Should have some data
        assert train_df.height > 0

    def test_batch_process_data_cleaning_failure(self, processing_config):
        """Test handling of data cleaning failures."""
        # Create data that will fail cleaning
        bad_data = pd.DataFrame({
            'gauge_id': ['bad_basin'] * 10,
            'date': pd.date_range('2020-01-01', periods=10),
            'temperature': [np.nan] * 10,  # All nulls
            'precipitation': [np.nan] * 10,
            'streamflow': [np.nan] * 10
        })
        
        lf = pl.from_pandas(bad_data).lazy()
        
        features_pipeline = GroupedPipeline(
            pipeline=Pipeline([("scaler", StandardScaler())]),
            columns=["temperature", "precipitation"],
            group_identifier="gauge_id"
        )
        
        target_pipeline = UnifiedPipeline(
            pipeline=Pipeline([("scaler", MinMaxScaler())]),
            columns=["streamflow"]
        )
        
        with pytest.raises(DataQualityError, match="No valid basins found after quality checks"):
            batch_process_time_series_data(
                lf=lf,
                config=processing_config,
                features_pipeline=features_pipeline,
                target_pipeline=target_pipeline
            )

    def test_batch_process_no_valid_basins(self, insufficient_data, processing_config):
        """Test handling when no basins pass quality checks."""
        lf = pl.from_pandas(insufficient_data).lazy()
        
        features_pipeline = GroupedPipeline(
            pipeline=Pipeline([("scaler", StandardScaler())]),
            columns=["temperature", "precipitation"],
            group_identifier="gauge_id"
        )
        
        target_pipeline = UnifiedPipeline(
            pipeline=Pipeline([("scaler", MinMaxScaler())]),
            columns=["streamflow"]
        )
        
        with pytest.raises(DataQualityError, match="Quality check failed"):
            batch_process_time_series_data(
                lf=lf,
                config=processing_config,
                features_pipeline=features_pipeline,
                target_pipeline=target_pipeline
            )

    @patch('hydro_forecasting.data.preprocessing.clean_data')
    def test_batch_process_empty_training_data(self, mock_clean_data, processing_config):
        """Test handling when training split is empty."""
        # Mock clean_data to return empty training data
        mock_df = pl.DataFrame({
            'gauge_id': ['basin_001'],
            'date': [datetime(2020, 1, 1)],
            'temperature': [15.0],
            'precipitation': [5.0],
            'streamflow': [10.0]
        })
        
        mock_reports = {
            'basin_001': Mock(passed_quality_check=True)
        }
        
        mock_clean_data.return_value = (mock_df, mock_reports)
        
        # Create mock LazyFrame
        mock_lf = Mock()
        
        features_pipeline = GroupedPipeline(
            pipeline=Pipeline([("scaler", StandardScaler())]),
            columns=["temperature", "precipitation"],
            group_identifier="gauge_id"
        )
        
        target_pipeline = UnifiedPipeline(
            pipeline=Pipeline([("scaler", MinMaxScaler())]),
            columns=["streamflow"]
        )
        
        # This should handle empty training data gracefully
        result = batch_process_time_series_data(
            lf=mock_lf,
            config=processing_config,
            features_pipeline=features_pipeline,
            target_pipeline=target_pipeline
        )
        
        train_df, val_df, test_df, fitted_pipelines, quality_reports = result
        
        # Should return empty DataFrames
        assert train_df.height == 0
        assert val_df.height == 0
        assert test_df.height == 0


class TestProcessingOutput:
    """Test ProcessingOutput dataclass."""

    def test_processing_output_creation(self, temp_dir, mock_quality_reports):
        """Test creating ProcessingOutput with all fields."""
        from hydro_forecasting.data.clean_data import SummaryQualityReport
        
        summary_report = SummaryQualityReport(
            original_basins=5,
            passed_basins=3,
            failed_basins=2,
            excluded_basins={"basin_001": "failed"},
            retained_basins=["basin_002", "basin_003"]
        )
        
        output = ProcessingOutput(
            run_output_dir=temp_dir,
            processed_timeseries_dir=temp_dir / "timeseries",
            processed_static_attributes_path=temp_dir / "static.parquet",
            fitted_time_series_pipelines_path=temp_dir / "ts_pipelines.joblib",
            fitted_static_pipeline_path=temp_dir / "static_pipeline.joblib",
            quality_reports_dir=temp_dir / "reports",
            summary_quality_report_path=temp_dir / "summary.json",
            config_path=temp_dir / "config.json",
            success_marker_path=temp_dir / "_SUCCESS",
            summary_quality_report=summary_report
        )
        
        assert output.run_output_dir == temp_dir
        assert output.processed_timeseries_dir == temp_dir / "timeseries"
        assert output.summary_quality_report.passed_basins == 3

    def test_processing_output_with_none_paths(self, temp_dir):
        """Test ProcessingOutput with optional None paths."""
        from hydro_forecasting.data.clean_data import SummaryQualityReport
        
        summary_report = SummaryQualityReport(
            original_basins=1,
            passed_basins=1,
            failed_basins=0,
            excluded_basins={},
            retained_basins=["basin_001"]
        )
        
        output = ProcessingOutput(
            run_output_dir=temp_dir,
            processed_timeseries_dir=temp_dir / "timeseries",
            processed_static_attributes_path=None,  # Static processing skipped
            fitted_time_series_pipelines_path=temp_dir / "ts_pipelines.joblib",
            fitted_static_pipeline_path=None,  # Static processing skipped
            quality_reports_dir=temp_dir / "reports",
            summary_quality_report_path=temp_dir / "summary.json",
            config_path=temp_dir / "config.json",
            success_marker_path=temp_dir / "_SUCCESS",
            summary_quality_report=summary_report
        )
        
        assert output.processed_static_attributes_path is None
        assert output.fitted_static_pipeline_path is None
        assert output.run_output_dir == temp_dir


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in preprocessing functions."""

    def test_split_data_with_invalid_proportions(self, synthetic_clean_data):
        """Test split_data with proportions that don't sum to 1."""
        config = ProcessingConfig(
            required_columns=["temperature", "precipitation", "streamflow"],
            train_prop=0.5,
            val_prop=0.3,
            test_prop=0.3  # Sum = 1.1
        )
        
        df = pl.from_pandas(synthetic_clean_data)
        
        # Should still work, just with non-standard proportions
        train_df, val_df, test_df = split_data(df, config)
        assert train_df.height > 0
        assert val_df.height > 0
        assert test_df.height > 0

    def test_create_pipeline_with_empty_columns(self):
        """Test create_pipeline with empty columns list."""
        config = {
            "strategy": "unified",
            "pipeline": Pipeline([("scaler", StandardScaler())]),
            "columns": []
        }
        
        result = create_pipeline(config)
        assert isinstance(result, UnifiedPipeline)
        assert result.columns == []

    def test_batch_basins_with_large_batch_size(self):
        """Test batch_basins with batch size larger than list."""
        basins = ["basin_001", "basin_002"]
        batch_size = 100
        
        batches = list(batch_basins(basins, batch_size))
        
        assert len(batches) == 1
        assert batches[0] == basins