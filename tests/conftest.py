"""
Comprehensive pytest fixtures for hydrological forecasting preprocessing pipeline tests.

This module provides fixtures for:
- Synthetic hydrological data generation
- Mock pipeline configurations
- Temporary directories and file operations
- Common test data scenarios (clean, missing, gaps, outliers)
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from hydro_forecasting.data.preprocessing import ProcessingConfig
from hydro_forecasting.exceptions import (
    ConfigurationError,
    DataProcessingError,
    DataQualityError,
    FileOperationError,
)
from hydro_forecasting.preprocessing.grouped import GroupedPipeline
from hydro_forecasting.preprocessing.unified import UnifiedPipeline


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def processing_config():
    """Standard processing configuration for tests."""
    return ProcessingConfig(
        required_columns=["temperature", "precipitation", "streamflow"],
        min_train_years=5.0,
        max_imputation_gap_size=5,
        group_identifier="gauge_id",
        train_prop=0.6,
        val_prop=0.2,
        test_prop=0.2,
    )


@pytest.fixture
def basin_ids():
    """List of basin IDs for testing."""
    return ["basin_001", "basin_002", "basin_003", "basin_004", "basin_005"]


@pytest.fixture
def date_range():
    """Standard date range for synthetic data (10 years)."""
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2019, 12, 31)
    return pd.date_range(start=start_date, end=end_date, freq="D")


@pytest.fixture
def synthetic_clean_data(basin_ids, date_range):
    """Generate clean synthetic hydrological data for multiple basins."""
    np.random.seed(42)
    
    all_data = []
    
    for basin_id in basin_ids:
        n_days = len(date_range)
        
        # Generate seasonal patterns
        day_of_year = np.array([d.timetuple().tm_yday for d in date_range])
        seasonal_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365.25)
        seasonal_precip = 50 + 30 * np.sin(2 * np.pi * (day_of_year + 90) / 365.25)
        
        # Add basin-specific characteristics
        basin_offset = hash(basin_id) % 100 / 10.0
        
        # Generate correlated variables with noise
        temperature = seasonal_temp + basin_offset + np.random.normal(0, 3, n_days)
        precipitation = np.maximum(0, seasonal_precip + np.random.exponential(2, n_days))
        
        # Streamflow correlated with precipitation (with lag) and temperature
        streamflow_base = np.convolve(precipitation, np.array([0.1, 0.3, 0.4, 0.2]), mode='same')
        streamflow_temp_effect = -0.5 * (temperature - 15)  # Higher temp = lower flow
        streamflow = np.maximum(0.1, streamflow_base + streamflow_temp_effect + 
                               np.random.normal(0, 5, n_days) + basin_offset * 2)
        
        basin_data = pd.DataFrame({
            'gauge_id': basin_id,
            'date': date_range,
            'temperature': temperature,
            'precipitation': precipitation,
            'streamflow': streamflow
        })
        
        all_data.append(basin_data)
    
    return pd.concat(all_data, ignore_index=True)


@pytest.fixture
def synthetic_data_with_gaps(synthetic_clean_data):
    """Create synthetic data with realistic gap patterns."""
    np.random.seed(42)
    data = synthetic_clean_data.copy()
    
    # Add different types of gaps for different basins
    for basin_id in data['gauge_id'].unique():
        basin_mask = data['gauge_id'] == basin_id
        basin_indices = data.index[basin_mask].tolist()
        
        # Random short gaps (1-3 days)
        n_short_gaps = np.random.randint(5, 15)
        for _ in range(n_short_gaps):
            gap_start = np.random.choice(basin_indices)
            gap_length = np.random.randint(1, 4)
            gap_end = min(gap_start + gap_length, len(basin_indices))
            gap_indices = basin_indices[gap_start:gap_end]
            
            # Randomly choose which column to create gap in
            column = np.random.choice(['temperature', 'precipitation', 'streamflow'])
            data.loc[gap_indices, column] = np.nan
        
        # One longer gap (6-10 days) for some basins
        if np.random.random() < 0.3:  # 30% chance
            gap_start = np.random.choice(basin_indices[100:-100])  # Avoid edges
            gap_length = np.random.randint(6, 11)
            gap_end = min(gap_start + gap_length, len(basin_indices))
            gap_indices = basin_indices[gap_start:gap_end]
            
            column = np.random.choice(['temperature', 'precipitation'])
            data.loc[gap_indices, column] = np.nan
    
    return data


@pytest.fixture
def synthetic_data_with_outliers(synthetic_clean_data):
    """Create synthetic data with outliers."""
    np.random.seed(42)
    data = synthetic_clean_data.copy()
    
    # Add outliers
    n_outliers = int(len(data) * 0.001)  # 0.1% outliers
    outlier_indices = np.random.choice(data.index, size=n_outliers, replace=False)
    
    for idx in outlier_indices:
        column = np.random.choice(['temperature', 'precipitation', 'streamflow'])
        current_value = data.loc[idx, column]
        
        # Create extreme outliers (5-10x normal value)
        multiplier = np.random.uniform(5, 10) if np.random.random() < 0.5 else np.random.uniform(0.1, 0.2)
        data.loc[idx, column] = current_value * multiplier
    
    return data


@pytest.fixture
def insufficient_data(basin_ids):
    """Generate data with insufficient training years."""
    np.random.seed(42)
    
    # Only 2 years of data - below the 5-year minimum
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2019, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    
    all_data = []
    for basin_id in basin_ids[:2]:  # Only 2 basins
        n_days = len(date_range)
        
        basin_data = pd.DataFrame({
            'gauge_id': basin_id,
            'date': date_range,
            'temperature': np.random.normal(15, 5, n_days),
            'precipitation': np.random.exponential(2, n_days),
            'streamflow': np.random.exponential(10, n_days)
        })
        
        all_data.append(basin_data)
    
    return pd.concat(all_data, ignore_index=True)


@pytest.fixture
def missing_columns_data(synthetic_clean_data):
    """Data missing required columns."""
    return synthetic_clean_data.drop(columns=['temperature'])


@pytest.fixture
def grouped_pipeline_config():
    """Configuration for GroupedPipeline testing."""
    return {
        "strategy": "per_group",
        "pipeline": GroupedPipeline(
            pipeline=Pipeline([("scaler", StandardScaler())]),
            columns=["temperature", "precipitation"],
            group_identifier="gauge_id"
        )
    }


@pytest.fixture
def unified_pipeline_config():
    """Configuration for UnifiedPipeline testing."""
    return {
        "strategy": "unified",
        "pipeline": Pipeline([("scaler", MinMaxScaler())]),
        "columns": ["temperature", "precipitation"]
    }


@pytest.fixture
def preprocessing_config_valid():
    """Valid preprocessing configuration."""
    return {
        "features": {
            "strategy": "per_group",
            "pipeline": GroupedPipeline(
                pipeline=Pipeline([("scaler", StandardScaler())]),
                columns=["temperature", "precipitation"],
                group_identifier="gauge_id"
            )
        },
        "target": {
            "strategy": "unified",
            "pipeline": Pipeline([("scaler", MinMaxScaler())]),
            "columns": ["streamflow"],
            "column": "streamflow"
        }
    }


@pytest.fixture
def preprocessing_config_invalid_strategy():
    """Invalid preprocessing configuration with unknown strategy."""
    return {
        "features": {
            "strategy": "invalid_strategy",
            "pipeline": Pipeline([("scaler", StandardScaler())]),
            "columns": ["temperature", "precipitation"]
        }
    }


@pytest.fixture
def preprocessing_config_missing_pipeline():
    """Invalid preprocessing configuration missing pipeline."""
    return {
        "features": {
            "strategy": "unified",
            "columns": ["temperature", "precipitation"]
        }
    }


@pytest.fixture
def region_dirs(temp_dir):
    """Create temporary directory structure for region data."""
    regions = {
        "basin": temp_dir / "time_series" / "basin",
        "river": temp_dir / "time_series" / "river"
    }
    
    static_regions = {
        "basin": temp_dir / "static" / "basin",
        "river": temp_dir / "static" / "river"
    }
    
    for region_dir in regions.values():
        region_dir.mkdir(parents=True, exist_ok=True)
    
    for static_dir in static_regions.values():
        static_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "time_series": regions,
        "static": static_regions
    }


@pytest.fixture
def create_basin_files(region_dirs, synthetic_clean_data, basin_ids):
    """Create actual parquet files for basin data."""
    for basin_id in basin_ids:
        basin_data = synthetic_clean_data[synthetic_clean_data['gauge_id'] == basin_id].copy()
        
        # Convert to polars and save
        pl_data = pl.from_pandas(basin_data)
        
        # Determine region prefix
        prefix = basin_id.split("_")[0]
        if prefix in region_dirs["time_series"]:
            file_path = region_dirs["time_series"][prefix] / f"{basin_id}.parquet"
        else:
            file_path = region_dirs["time_series"]["basin"] / f"{basin_id}.parquet"
        
        pl_data.write_parquet(file_path)
    
    return region_dirs["time_series"]


@pytest.fixture
def invalid_config_proportions():
    """Configuration with invalid train/val/test proportions."""
    return ProcessingConfig(
        required_columns=["temperature", "precipitation", "streamflow"],
        min_train_years=5.0,
        max_imputation_gap_size=5,
        group_identifier="gauge_id",
        train_prop=0.7,
        val_prop=0.3,
        test_prop=0.2  # Sum > 1.0
    )


@pytest.fixture
def mock_quality_reports():
    """Mock quality reports for testing."""
    from hydro_forecasting.data.clean_data import BasinQualityReport
    
    reports = {}
    
    # Passing report
    reports["basin_001"] = BasinQualityReport(
        valid_period={
            "temperature": {"start": "2010-01-01", "end": "2019-12-31"},
            "precipitation": {"start": "2010-01-01", "end": "2019-12-31"},
            "streamflow": {"start": "2010-01-01", "end": "2019-12-31"}
        },
        processing_steps=["sorted_by_gauge_and_date", "trimmed_nulls", "imputed_short_gaps_forward_only"],
        imputation_info={
            "temperature": {"short_gaps_count": 2, "imputed_values_count": 3},
            "precipitation": {"short_gaps_count": 1, "imputed_values_count": 2},
            "streamflow": {"short_gaps_count": 0, "imputed_values_count": 0}
        },
        passed_quality_check=True,
        failure_reason=None
    )
    
    # Failing report
    reports["basin_002"] = BasinQualityReport(
        valid_period={
            "temperature": {"start": "2018-01-01", "end": "2019-12-31"},
            "precipitation": {"start": "2018-01-01", "end": "2019-12-31"},
            "streamflow": {"start": "2018-01-01", "end": "2019-12-31"}
        },
        processing_steps=["sorted_by_gauge_and_date", "trimmed_nulls"],
        imputation_info={
            "temperature": {"short_gaps_count": 0, "imputed_values_count": 0},
            "precipitation": {"short_gaps_count": 0, "imputed_values_count": 0},
            "streamflow": {"short_gaps_count": 0, "imputed_values_count": 0}
        },
        passed_quality_check=False,
        failure_reason="Insufficient training data: 2.00 years available (730 data points). Minimum required: 5.0 years (1826 data points)"
    )
    
    return reports


@pytest.fixture
def polars_lazyframe(synthetic_clean_data):
    """Convert synthetic data to polars LazyFrame."""
    return pl.from_pandas(synthetic_clean_data).lazy()


@pytest.fixture
def large_dataset(basin_ids):
    """Generate a large dataset for performance testing."""
    np.random.seed(42)
    
    # 50 basins, 20 years each for performance testing
    large_basin_ids = [f"perf_basin_{i:03d}" for i in range(50)]
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2019, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    
    all_data = []
    
    for basin_id in large_basin_ids:
        n_days = len(date_range)
        
        basin_data = pd.DataFrame({
            'gauge_id': basin_id,
            'date': date_range,
            'temperature': np.random.normal(15, 5, n_days),
            'precipitation': np.random.exponential(2, n_days),
            'streamflow': np.random.exponential(10, n_days),
            'humidity': np.random.uniform(30, 90, n_days),
            'wind_speed': np.random.exponential(5, n_days)
        })
        
        all_data.append(basin_data)
    
    return pd.concat(all_data, ignore_index=True)


@pytest.fixture
def edge_case_data():
    """Generate edge case scenarios for testing."""
    scenarios = {}
    
    # Single data point
    scenarios["single_point"] = pd.DataFrame({
        'gauge_id': ['single_basin'],
        'date': [datetime(2020, 1, 1)],
        'temperature': [15.0],
        'precipitation': [5.0],
        'streamflow': [10.0]
    })
    
    # All nulls
    scenarios["all_nulls"] = pd.DataFrame({
        'gauge_id': ['null_basin'] * 100,
        'date': pd.date_range('2020-01-01', periods=100),
        'temperature': [np.nan] * 100,
        'precipitation': [np.nan] * 100,
        'streamflow': [np.nan] * 100
    })
    
    # Duplicate timestamps
    scenarios["duplicate_dates"] = pd.DataFrame({
        'gauge_id': ['dup_basin'] * 4,
        'date': [datetime(2020, 1, 1)] * 2 + [datetime(2020, 1, 2)] * 2,
        'temperature': [15.0, 16.0, 17.0, 18.0],
        'precipitation': [5.0, 6.0, 7.0, 8.0],
        'streamflow': [10.0, 11.0, 12.0, 13.0]
    })
    
    return scenarios


@pytest.fixture
def mock_file_paths(temp_dir):
    """Create mock file path structures for testing."""
    paths = {
        "config": temp_dir / "config.json",
        "quality_report": temp_dir / "basin_001.json",
        "summary_report": temp_dir / "summary.json",
        "processed_data": temp_dir / "processed",
        "pipelines": temp_dir / "pipelines.joblib"
    }
    
    # Create directories
    paths["processed_data"].mkdir(parents=True, exist_ok=True)
    
    return paths


# Parametrize fixtures for common test scenarios
@pytest.fixture(params=[
    "clean_data",
    "data_with_gaps", 
    "data_with_outliers"
])
def data_scenario(request, synthetic_clean_data, synthetic_data_with_gaps, synthetic_data_with_outliers):
    """Parametrized fixture for different data quality scenarios."""
    scenarios = {
        "clean_data": synthetic_clean_data,
        "data_with_gaps": synthetic_data_with_gaps,
        "data_with_outliers": synthetic_data_with_outliers
    }
    return scenarios[request.param]


@pytest.fixture(params=[
    {"train_prop": 0.6, "val_prop": 0.2, "test_prop": 0.2},
    {"train_prop": 0.7, "val_prop": 0.15, "test_prop": 0.15},
    {"train_prop": 0.8, "val_prop": 0.1, "test_prop": 0.1}
])
def split_proportions(request):
    """Parametrized fixture for different data split proportions."""
    return request.param


# Helper functions that can be used in tests
def create_quality_report_files(reports_dir: Path, reports: Dict[str, Any]) -> None:
    """Helper function to create quality report JSON files."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    for basin_id, report in reports.items():
        report_file = reports_dir / f"{basin_id}.json"
        with open(report_file, 'w') as f:
            if hasattr(report, '__dict__'):
                # If it's a dataclass, convert to dict
                from dataclasses import asdict
                json.dump(asdict(report), f, indent=2)
            else:
                json.dump(report, f, indent=2)


def assert_dataframe_quality(df: pd.DataFrame, expected_columns: List[str], min_rows: int = 1) -> None:
    """Helper function to assert basic DataFrame quality."""
    assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
    assert len(df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"
    
    missing_cols = [col for col in expected_columns if col not in df.columns]
    assert not missing_cols, f"Missing expected columns: {missing_cols}"
    
    # Check for reasonable data types
    for col in expected_columns:
        if col == 'date':
            assert pd.api.types.is_datetime64_any_dtype(df[col]), f"Column {col} should be datetime"
        elif col in ['gauge_id']:
            assert df[col].dtype == 'object', f"Column {col} should be string/object"
        else:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"