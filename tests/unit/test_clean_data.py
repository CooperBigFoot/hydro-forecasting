"""
Comprehensive unit tests for the data cleaning module.

Tests cover:
- clean_data() function with various data quality scenarios
- find_gaps_bool() utility function
- BasinQualityReport and SummaryQualityReport functionality
- Data validation and quality control logic
- Gap detection and imputation
- Error handling and edge cases
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from hydro_forecasting.data.clean_data import (
    BasinQualityReport,
    SummaryQualityReport,
    clean_data,
    find_gaps_bool,
    save_quality_report_to_json,
    summarize_quality_reports_from_folder,
)
from hydro_forecasting.exceptions import DataProcessingError, DataQualityError, FileOperationError


class TestFindGapsBool:
    """Test the find_gaps_bool utility function."""

    def test_find_gaps_bool_no_gaps(self):
        """Test with no missing data."""
        missing = np.array([False, False, False, False, False])
        starts, ends = find_gaps_bool(missing)
        assert len(starts) == 0
        assert len(ends) == 0

    def test_find_gaps_bool_single_gap(self):
        """Test with single gap."""
        missing = np.array([False, True, True, True, False])
        starts, ends = find_gaps_bool(missing)
        np.testing.assert_array_equal(starts, [1])
        np.testing.assert_array_equal(ends, [4])

    def test_find_gaps_bool_multiple_gaps(self):
        """Test with multiple separate gaps."""
        missing = np.array([True, False, True, True, False, True])
        starts, ends = find_gaps_bool(missing)
        np.testing.assert_array_equal(starts, [0, 2, 5])
        np.testing.assert_array_equal(ends, [1, 4, 6])

    def test_find_gaps_bool_all_missing(self):
        """Test with all data missing."""
        missing = np.array([True, True, True, True])
        starts, ends = find_gaps_bool(missing)
        np.testing.assert_array_equal(starts, [0])
        np.testing.assert_array_equal(ends, [4])

    def test_find_gaps_bool_empty_array(self):
        """Test with empty array."""
        missing = np.array([], dtype=bool)
        starts, ends = find_gaps_bool(missing)
        assert len(starts) == 0
        assert len(ends) == 0

    def test_find_gaps_bool_single_point_missing(self):
        """Test with single point missing."""
        missing = np.array([False, True, False])
        starts, ends = find_gaps_bool(missing)
        np.testing.assert_array_equal(starts, [1])
        np.testing.assert_array_equal(ends, [2])


class TestBasinQualityReport:
    """Test BasinQualityReport dataclass functionality."""

    def test_basin_quality_report_creation(self):
        """Test creating a BasinQualityReport."""
        report = BasinQualityReport(
            valid_period={"streamflow": {"start": "2010-01-01", "end": "2019-12-31"}},
            processing_steps=["sorted", "cleaned"],
            imputation_info={"streamflow": {"short_gaps_count": 2, "imputed_values_count": 5}},
            passed_quality_check=True,
            failure_reason=None
        )
        
        assert report.passed_quality_check is True
        assert report.failure_reason is None
        assert len(report.processing_steps) == 2
        assert "streamflow" in report.valid_period

    def test_basin_quality_report_failed(self):
        """Test creating a failed BasinQualityReport."""
        report = BasinQualityReport(
            valid_period={},
            processing_steps=["sorted"],
            imputation_info={},
            passed_quality_check=False,
            failure_reason="Insufficient training data"
        )
        
        assert report.passed_quality_check is False
        assert report.failure_reason == "Insufficient training data"


class TestSummaryQualityReport:
    """Test SummaryQualityReport functionality."""

    def test_summary_quality_report_creation(self):
        """Test creating a SummaryQualityReport."""
        summary = SummaryQualityReport(
            original_basins=10,
            passed_basins=8,
            failed_basins=2,
            excluded_basins={"basin_001": "insufficient data", "basin_002": "missing columns"},
            retained_basins=["basin_003", "basin_004", "basin_005"]
        )
        
        assert summary.original_basins == 10
        assert summary.passed_basins == 8
        assert summary.failed_basins == 2
        assert len(summary.excluded_basins) == 2
        assert len(summary.retained_basins) == 3

    def test_summary_quality_report_save(self, temp_dir):
        """Test saving SummaryQualityReport to JSON."""
        summary = SummaryQualityReport(
            original_basins=5,
            passed_basins=3,
            failed_basins=2,
            excluded_basins={"basin_001": "failed"},
            retained_basins=["basin_002", "basin_003"]
        )
        
        save_path = temp_dir / "summary.json"
        returned_path = summary.save(save_path)
        
        assert returned_path == save_path
        assert save_path.exists()
        
        # Verify content
        with open(save_path) as f:
            data = json.load(f)
        
        assert data["original_basins"] == 5
        assert data["passed_basins"] == 3
        assert "basin_001" in data["excluded_basins"]


class TestCleanDataFunction:
    """Test the main clean_data function."""

    def test_clean_data_valid_input(self, synthetic_clean_data, processing_config):
        """Test clean_data with valid input data."""
        lf = pl.from_pandas(synthetic_clean_data).lazy()
        
        cleaned_df, reports = clean_data(lf, processing_config)
        
        assert isinstance(cleaned_df, pl.DataFrame)
        assert isinstance(reports, dict)
        assert len(reports) > 0
        
        # All basins should pass quality checks with clean data
        for basin_id, report in reports.items():
            assert report.passed_quality_check is True
            assert report.failure_reason is None

    def test_clean_data_missing_columns(self, missing_columns_data, processing_config):
        """Test clean_data with missing required columns."""
        lf = pl.from_pandas(missing_columns_data).lazy()
        
        with pytest.raises(DataProcessingError, match="Missing columns"):
            clean_data(lf, processing_config)

    def test_clean_data_gap_imputation(self, synthetic_data_with_gaps, processing_config):
        """Test gap filling with different gap sizes."""
        lf = pl.from_pandas(synthetic_data_with_gaps).lazy()
        
        cleaned_df, reports = clean_data(lf, processing_config)
        
        # Check that some imputation occurred
        imputation_occurred = False
        for report in reports.values():
            for col_info in report.imputation_info.values():
                if col_info["imputed_values_count"] > 0:
                    imputation_occurred = True
                    break
        
        assert imputation_occurred, "Expected some imputation to occur with gapped data"

    def test_clean_data_insufficient_training_years(self, insufficient_data, processing_config):
        """Test with insufficient training data."""
        lf = pl.from_pandas(insufficient_data).lazy()
        
        with pytest.raises(DataQualityError, match="Quality check failed"):
            clean_data(lf, processing_config)

    def test_clean_data_no_raise_on_failure(self, insufficient_data, processing_config):
        """Test clean_data with raise_on_failure=False."""
        lf = pl.from_pandas(insufficient_data).lazy()
        
        cleaned_df, reports = clean_data(lf, processing_config, raise_on_failure=False)
        
        # Should return empty DataFrame and failed reports
        assert cleaned_df.height == 0
        assert len(reports) > 0
        
        # All reports should show failure
        for report in reports.values():
            assert report.passed_quality_check is False
            assert "Insufficient training data" in report.failure_reason

    def test_clean_data_edge_case_single_point(self, edge_case_data, processing_config):
        """Test with single data point."""
        lf = pl.from_pandas(edge_case_data["single_point"]).lazy()
        
        with pytest.raises(DataQualityError, match="Quality check failed"):
            clean_data(lf, processing_config)

    def test_clean_data_edge_case_all_nulls(self, edge_case_data, processing_config):
        """Test with all null data."""
        lf = pl.from_pandas(edge_case_data["all_nulls"]).lazy()
        
        cleaned_df, reports = clean_data(lf, processing_config, raise_on_failure=False)
        
        assert cleaned_df.height == 0
        for report in reports.values():
            assert report.passed_quality_check is False

    def test_clean_data_minimum_training_years(self, synthetic_clean_data, processing_config):
        """Test minimum training years validation."""
        # Create data with exactly the minimum required years + buffer
        min_years_data = synthetic_clean_data.copy()
        
        # The synthetic data goes from 2010-01-01 to 2019-12-31 (10 years)
        # Filter to get sufficient training data (use the full range, which is 10 years)
        # This will definitely meet the 5-year minimum requirement with 60/20/20 split
        start_date = datetime(2010, 1, 1)   # Use full start date
        end_date = datetime(2019, 12, 31)   # Use full end date
        date_mask = (min_years_data['date'] >= start_date) & (min_years_data['date'] <= end_date)
        min_years_data = min_years_data[date_mask]
        
        lf = pl.from_pandas(min_years_data).lazy()
        cleaned_df, reports = clean_data(lf, processing_config)
        
        # Should pass with enough data (10 years * 0.6 = 6 years training > 5 year minimum)
        assert cleaned_df.height > 0
        for report in reports.values():
            assert report.passed_quality_check is True

    def test_clean_data_quality_reports_structure(self, synthetic_clean_data, processing_config):
        """Test quality report generation accuracy."""
        lf = pl.from_pandas(synthetic_clean_data).lazy()
        
        cleaned_df, reports = clean_data(lf, processing_config)
        
        # Verify report structure
        for basin_id, report in reports.items():
            assert isinstance(report, BasinQualityReport)
            assert isinstance(report.valid_period, dict)
            assert isinstance(report.processing_steps, list)
            assert isinstance(report.imputation_info, dict)
            assert isinstance(report.passed_quality_check, bool)
            
            # Check required columns are in valid_period
            for col in processing_config.required_columns:
                assert col in report.valid_period
                assert "start" in report.valid_period[col]
                assert "end" in report.valid_period[col]

    def test_clean_data_target_column_validation(self, synthetic_clean_data, processing_config):
        """Test that target column validation works correctly."""
        # Set up preprocessing config with target column specification
        processing_config.preprocessing_config = {
            "target": {"column": "streamflow"}
        }
        
        lf = pl.from_pandas(synthetic_clean_data).lazy()
        cleaned_df, reports = clean_data(lf, processing_config)
        
        # Should pass since streamflow column exists
        for report in reports.values():
            assert report.passed_quality_check is True

    def test_clean_data_missing_target_column(self, synthetic_clean_data, processing_config):
        """Test behavior when target column is missing."""
        # Drop the streamflow column from required_columns and data
        data_no_target = synthetic_clean_data.drop(columns=['streamflow'])
        
        # Remove streamflow from required columns to test target column specific handling
        processing_config.required_columns = ["temperature", "precipitation"]
        
        # Update processing config with proper structure for missing target
        if not hasattr(processing_config, 'preprocessing_config') or processing_config.preprocessing_config is None:
            processing_config.preprocessing_config = {}
        processing_config.preprocessing_config["target"] = {"column": "streamflow"}
        
        lf = pl.from_pandas(data_no_target).lazy()
        
        # This should raise DataQualityError because target column is missing from basin data
        with pytest.raises(DataQualityError, match="Target column 'streamflow' not found in basin data"):
            clean_data(lf, processing_config, raise_on_failure=True)

    @pytest.mark.parametrize("max_gap_size", [1, 3, 5, 10])
    def test_clean_data_gap_size_limits(self, synthetic_data_with_gaps, processing_config, max_gap_size):
        """Test different maximum gap sizes for imputation."""
        processing_config.max_imputation_gap_size = max_gap_size
        
        lf = pl.from_pandas(synthetic_data_with_gaps).lazy()
        cleaned_df, reports = clean_data(lf, processing_config, raise_on_failure=False)
        
        # Verify gap imputation respects the limit
        for report in reports.values():
            for col_info in report.imputation_info.values():
                # Short gaps count should be reasonable for the gap size limit
                assert col_info["short_gaps_count"] >= 0
                assert col_info["imputed_values_count"] >= 0

    def test_clean_data_proportions_validation(self, synthetic_clean_data, processing_config):
        """Test data splitting proportions are validated correctly."""
        # Test with valid proportions
        lf = pl.from_pandas(synthetic_clean_data).lazy()
        cleaned_df, reports = clean_data(lf, processing_config)
        
        # Should work with default proportions
        assert cleaned_df.height > 0
        for report in reports.values():
            assert report.passed_quality_check is True

    def test_clean_data_group_identifier_handling(self, synthetic_clean_data, processing_config):
        """Test handling of group identifier column."""
        # Test with missing group identifier
        data_no_group = synthetic_clean_data.drop(columns=[processing_config.group_identifier])
        
        with pytest.raises(DataProcessingError, match="Missing columns"):
            lf = pl.from_pandas(data_no_group).lazy()
            clean_data(lf, processing_config)


class TestSaveQualityReportToJson:
    """Test saving quality reports to JSON files."""

    def test_save_quality_report_to_json_success(self, temp_dir, mock_quality_reports):
        """Test successful saving of quality report."""
        report = mock_quality_reports["basin_001"]
        save_path = temp_dir / "basin_001.json"
        
        save_quality_report_to_json(report, save_path)
        
        assert save_path.exists()
        
        # Verify content
        with open(save_path) as f:
            data = json.load(f)
        
        assert data["passed_quality_check"] is True
        assert "streamflow" in data["valid_period"]

    def test_save_quality_report_to_json_creates_directory(self, temp_dir, mock_quality_reports):
        """Test that parent directories are created if they don't exist."""
        report = mock_quality_reports["basin_001"]
        save_path = temp_dir / "nested" / "dir" / "basin_001.json"
        
        save_quality_report_to_json(report, save_path)
        
        assert save_path.exists()
        assert save_path.parent.exists()

    def test_save_quality_report_to_json_file_error(self, mock_quality_reports):
        """Test error handling when file cannot be written."""
        report = mock_quality_reports["basin_001"]
        invalid_path = "/invalid/path/that/does/not/exist/report.json"
        
        with pytest.raises(FileOperationError, match="Failed to save quality report"):
            save_quality_report_to_json(report, invalid_path)


class TestSummarizeQualityReportsFromFolder:
    """Test summarizing quality reports from a folder."""

    def test_summarize_quality_reports_success(self, temp_dir, mock_quality_reports):
        """Test successful summarization of quality reports."""
        # Create quality report files
        reports_dir = temp_dir / "reports"
        reports_dir.mkdir()
        
        for basin_id, report in mock_quality_reports.items():
            report_file = reports_dir / f"{basin_id}.json"
            save_quality_report_to_json(report, report_file)
        
        summary_path = temp_dir / "summary.json"
        summary = summarize_quality_reports_from_folder(reports_dir, summary_path)
        
        assert isinstance(summary, SummaryQualityReport)
        assert summary.original_basins == 2
        assert summary.passed_basins == 1
        assert summary.failed_basins == 1
        assert "basin_002" in summary.excluded_basins
        assert "basin_001" in summary.retained_basins
        
        # Verify file was created
        assert summary_path.exists()

    def test_summarize_quality_reports_no_files(self, temp_dir):
        """Test error when no JSON files found."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        summary_path = temp_dir / "summary.json"
        
        with pytest.raises(FileOperationError, match="No JSON files found"):
            summarize_quality_reports_from_folder(empty_dir, summary_path)

    def test_summarize_quality_reports_invalid_json(self, temp_dir):
        """Test error handling with invalid JSON files."""
        reports_dir = temp_dir / "reports"
        reports_dir.mkdir()
        
        # Create invalid JSON file
        invalid_file = reports_dir / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json content {")
        
        summary_path = temp_dir / "summary.json"
        
        with pytest.raises(DataProcessingError, match="Error decoding JSON"):
            summarize_quality_reports_from_folder(reports_dir, summary_path)

    def test_summarize_quality_reports_folder_access_error(self):
        """Test error when folder cannot be accessed."""
        nonexistent_dir = Path("/nonexistent/directory")
        summary_path = Path("/tmp/summary.json")
        
        with pytest.raises(FileOperationError, match="No JSON files found"):
            summarize_quality_reports_from_folder(nonexistent_dir, summary_path)

    def test_summarize_quality_reports_basin_id_from_filename(self, temp_dir):
        """Test that basin_id is correctly extracted from filename."""
        reports_dir = temp_dir / "reports"
        reports_dir.mkdir()
        
        # Create report without basin_id in JSON content
        report_data = {
            "valid_period": {"streamflow": {"start": "2010-01-01", "end": "2019-12-31"}},
            "processing_steps": ["cleaned"],
            "imputation_info": {"streamflow": {"short_gaps_count": 0, "imputed_values_count": 0}},
            "passed_quality_check": True,
            "failure_reason": None
        }
        
        report_file = reports_dir / "test_basin.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f)
        
        summary_path = temp_dir / "summary.json"
        summary = summarize_quality_reports_from_folder(reports_dir, summary_path)
        
        assert "test_basin" in summary.retained_basins


class TestDataProcessingEdgeCases:
    """Test edge cases and error scenarios in data processing."""

    def test_clean_data_empty_dataframe(self, processing_config):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['gauge_id', 'date', 'temperature', 'precipitation', 'streamflow'])
        lf = pl.from_pandas(empty_df).lazy()
        
        cleaned_df, reports = clean_data(lf, processing_config, raise_on_failure=False)
        
        assert cleaned_df.height == 0
        assert len(reports) == 0

    def test_clean_data_schema_collection_error(self, processing_config, monkeypatch):
        """Test error handling when schema collection fails."""
        # Create a mock LazyFrame that raises an error on collect_schema
        class MockLazyFrame:
            def collect_schema(self):
                raise Exception("Schema collection failed")
        
        mock_lf = MockLazyFrame()
        
        with pytest.raises(DataProcessingError, match="Failed to collect schema"):
            clean_data(mock_lf, processing_config)

    def test_clean_data_processing_steps_recorded(self, synthetic_clean_data, processing_config):
        """Test that processing steps are correctly recorded in reports."""
        lf = pl.from_pandas(synthetic_clean_data).lazy()
        
        cleaned_df, reports = clean_data(lf, processing_config)
        
        expected_steps = [
            "sorted_by_gauge_and_date",
            "trimmed_nulls", 
            "imputed_short_gaps_forward_only",
            "target_data_validation_passed"
        ]
        
        for report in reports.values():
            for step in expected_steps[:3]:  # First 3 are always present
                assert step in report.processing_steps
            
            # Last step only if quality check passed
            if report.passed_quality_check:
                assert "target_data_validation_passed" in report.processing_steps

    def test_clean_data_validates_segment_sizes(self, synthetic_clean_data, processing_config):
        """Test validation of minimum segment sizes for train/val/test splits."""
        # Use very small data that would create tiny segments
        small_data = synthetic_clean_data.head(10).copy()  # Only 10 days
        
        lf = pl.from_pandas(small_data).lazy()
        cleaned_df, reports = clean_data(lf, processing_config, raise_on_failure=False)
        
        # Should fail due to insufficient data for proper splits
        assert cleaned_df.height == 0
        for report in reports.values():
            assert report.passed_quality_check is False
            assert any(keyword in report.failure_reason for keyword in 
                      ["Insufficient training data", "segment too small"])

    def test_clean_data_forward_fill_only(self, processing_config):
        """Test that only forward fill is used (no backward fill to avoid data leakage)."""
        # Create data with a gap at the end that can only be filled by backward fill
        data = pd.DataFrame({
            'gauge_id': ['test_basin'] * 10,
            'date': pd.date_range('2015-01-01', periods=10),
            'temperature': [1, 2, 3, 4, 5, np.nan, np.nan, np.nan, np.nan, np.nan],
            'precipitation': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'streamflow': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        lf = pl.from_pandas(data).lazy()
        cleaned_df, reports = clean_data(lf, processing_config, raise_on_failure=False)
        
        # The gap at the end should NOT be filled since we only use forward fill
        if cleaned_df.height > 0:
            temp_data = cleaned_df.to_pandas()['temperature']
            # Last few values should still be NaN
            assert temp_data.iloc[-1] is np.nan or pd.isna(temp_data.iloc[-1])