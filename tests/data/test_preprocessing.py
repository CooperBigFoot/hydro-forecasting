import pytest
import pandas as pd
import numpy as np
from returns.pipeline import is_successful

from hydro_forecasting.data.preprocessing import (
    validate_input,
    find_valid_data_period,
    check_data_period,
    find_gaps,
    ensure_complete_date_range,
    check_missing_percentage,
    check_missing_gaps,
    trim_leading_trailing_nans,
    impute_short_gaps,
    process_basin,
    check_data_quality,
)

# Fixtures for common test data
@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    gauge_ids = ["G1", "G2"]
    
    data = []
    for gauge_id in gauge_ids:
        for date in dates:
            data.append({
                "gauge_id": gauge_id,
                "date": date,
                "flow": 10.0 if gauge_id == "G1" else 20.0,
                "precipitation": 5.0 if gauge_id == "G1" else 7.0,
            })
    
    return pd.DataFrame(data)

@pytest.fixture
def empty_quality_report():
    """Create an empty quality report structure."""
    return {
        "original_basins": 0,
        "retained_basins": 0,
        "excluded_basins": {},
        "basins": {},
        "split_method": "proportional"
    }

@pytest.fixture
def df_with_gaps():
    """Create a DataFrame with missing values."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    gauge_ids = ["G1"]
    
    data = []
    for gauge_id in gauge_ids:
        for i, date in enumerate(dates):
            # Create gaps in the data
            flow_val = 10.0
            precip_val = 5.0
            
            # Add NaN values for specific dates to create gaps
            if 3 <= i <= 5:  # 3-day gap
                flow_val = np.nan
            if 10 <= i <= 15:  # 6-day gap
                precip_val = np.nan
                
            data.append({
                "gauge_id": gauge_id,
                "date": date,
                "flow": flow_val,
                "precipitation": precip_val,
            })
    
    return pd.DataFrame(data)

@pytest.fixture
def long_df():
    """Create a DataFrame with a long time series for testing period splits."""
    start_date = pd.Timestamp("2010-01-01")
    end_date = pd.Timestamp("2020-12-31")
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    data = []
    for date in dates:
        data.append({
            "gauge_id": "G1",
            "date": date,
            "flow": 10.0 + np.sin(date.dayofyear / 365 * 2 * np.pi) * 5.0,  # Seasonal pattern
            "precipitation": 5.0 + np.random.normal(0, 1),
        })
    
    return pd.DataFrame(data)

# Tests for validate_input
def test_validate_input_valid(sample_df):
    """Test validate_input with valid input."""
    result = validate_input(sample_df, ["flow", "precipitation"])
    assert is_successful(result)
    assert result.unwrap().equals(sample_df)

def test_validate_input_missing_required_column(sample_df):
    """Test validate_input with missing required column."""
    result = validate_input(sample_df, ["flow", "temperature"])
    assert not is_successful(result)
    assert "Required columns not found: ['temperature']" in result.failure()

def test_validate_input_missing_date_column():
    """Test validate_input with missing date column."""
    df = pd.DataFrame({"gauge_id": ["G1"], "flow": [10.0]})
    result = validate_input(df, ["flow"])
    assert not is_successful(result)
    assert "DataFrame must contain 'gauge_id' and 'date' columns" in result.failure()

def test_validate_input_missing_gauge_id_column():
    """Test validate_input with missing gauge_id column."""
    df = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "flow": [10.0]})
    result = validate_input(df, ["flow"])
    assert not is_successful(result)
    assert "DataFrame must contain 'gauge_id' and 'date' columns" in result.failure()

def test_validate_input_custom_group_identifier():
    """Test validate_input with custom group identifier."""
    df = pd.DataFrame({
        "basin_id": ["B1"],
        "date": [pd.Timestamp("2020-01-01")],
        "flow": [10.0]
    })
    result = validate_input(df, ["flow"], group_identifier="basin_id")
    assert is_successful(result)
    assert result.unwrap().equals(df)

# Tests for find_valid_data_period
def test_find_valid_data_period_all_valid():
    """Test find_valid_data_period with all valid data."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    series = pd.Series(np.random.rand(len(dates)))
    
    result = find_valid_data_period(series, dates)
    assert is_successful(result)
    
    start_date, end_date = result.unwrap()
    assert start_date == dates[0]
    assert end_date == dates[-1]

def test_find_valid_data_period_leading_trailing_nans():
    """Test find_valid_data_period with leading and trailing NaNs."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    values = np.random.rand(len(dates))
    values[0:2] = np.nan  # Leading NaNs
    values[-2:] = np.nan  # Trailing NaNs
    series = pd.Series(values)
    
    result = find_valid_data_period(series, dates)
    assert is_successful(result)
    
    start_date, end_date = result.unwrap()
    assert start_date == dates[2]
    assert end_date == dates[-3]

def test_find_valid_data_period_all_nan():
    """Test find_valid_data_period with all NaN values."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    series = pd.Series([np.nan] * len(dates))
    
    result = find_valid_data_period(series, dates)
    assert is_successful(result)
    
    start_date, end_date = result.unwrap()
    assert start_date is None
    assert end_date is None

def test_find_valid_data_period_length_mismatch():
    """Test find_valid_data_period with series and dates of different lengths."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    series = pd.Series(np.random.rand(len(dates) - 1))
    
    result = find_valid_data_period(series, dates)
    assert not is_successful(result)
    assert "Series and dates must have the same length" in result.failure()

def test_find_valid_data_period_unsorted_dates():
    """Test find_valid_data_period with unsorted dates."""
    dates = pd.DatetimeIndex([
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
    ])
    series = pd.Series([1.0, 2.0, 3.0])
    
    result = find_valid_data_period(series, dates)
    assert not is_successful(result)
    assert "Dates must be sorted in ascending order" in result.failure()

# Tests for check_data_period
def test_check_data_period_valid():
    """Test check_data_period with a valid period."""
    start_date = pd.Timestamp("2010-01-01")
    end_date = pd.Timestamp("2020-01-01")
    
    result = check_data_period((start_date, end_date))
    assert is_successful(result)
    assert result.unwrap() == (start_date, end_date)

def test_check_data_period_null_dates():
    """Test check_data_period with null dates."""
    result = check_data_period((None, None))
    assert not is_successful(result)
    assert "Missing start or end date" in result.failure()

def test_check_data_period_invalid_date_type():
    """Test check_data_period with invalid date types."""
    result = check_data_period(("2010-01-01", "2020-01-01"))
    assert not is_successful(result)
    assert "Invalid date format" in result.failure()

def test_check_data_period_start_after_end():
    """Test check_data_period with start date after end date."""
    start_date = pd.Timestamp("2020-01-01")
    end_date = pd.Timestamp("2010-01-01")
    
    result = check_data_period((start_date, end_date))
    assert not is_successful(result)
    assert "Start date is after end date" in result.failure()

def test_check_data_period_insufficient_days():
    """Test check_data_period with insufficient days."""
    start_date = pd.Timestamp("2020-01-01")
    end_date = pd.Timestamp("2020-01-10")  # Only 10 days
    
    result = check_data_period((start_date, end_date), min_valid_days=30)
    assert not is_successful(result)
    assert "Insufficient training period" in result.failure()


def test_check_data_period_insufficient_training():
    """Test check_data_period with insufficient training period."""
    start_date = pd.Timestamp("2020-01-01")
    end_date = pd.Timestamp("2021-01-01")  # 1 year
    
    result = check_data_period(
        (start_date, end_date),
        min_train_years=2.0,
        train_prop=0.6,
    )
    assert not is_successful(result)
    assert "Insufficient training period" in result.failure()


# Tests for find_gaps
def test_find_gaps_no_missing():
    """Test find_gaps with no missing values."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    
    result = find_gaps(series, max_gap_length=2)
    assert is_successful(result)
    
    gap_starts, gap_ends = result.unwrap()
    assert len(gap_starts) == 0
    assert len(gap_ends) == 0

def test_find_gaps_single_gap():
    """Test find_gaps with a single gap."""
    series = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0])
    
    result = find_gaps(series, max_gap_length=2)
    assert is_successful(result)
    
    gap_starts, gap_ends = result.unwrap()
    assert len(gap_starts) == 1
    assert len(gap_ends) == 1
    assert gap_starts[0] == 1
    assert gap_ends[0] == 3

def test_find_gaps_multiple_gaps():
    """Test find_gaps with multiple gaps."""
    series = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
    
    result = find_gaps(series, max_gap_length=2)
    assert is_successful(result)
    
    gap_starts, gap_ends = result.unwrap()
    assert len(gap_starts) == 2
    assert len(gap_ends) == 2
    assert gap_starts[0] == 1
    assert gap_ends[0] == 2
    assert gap_starts[1] == 3
    assert gap_ends[1] == 5

def test_find_gaps_end_gap():
    """Test find_gaps with a gap at the end of the series."""
    series = pd.Series([1.0, 2.0, 3.0, np.nan, np.nan])
    
    result = find_gaps(series, max_gap_length=2)
    assert is_successful(result)
    
    gap_starts, gap_ends = result.unwrap()
    assert len(gap_starts) == 1
    assert len(gap_ends) == 1
    assert gap_starts[0] == 3
    assert gap_ends[0] == 5  # end of series

# Tests for trim_leading_trailing_nans
def test_trim_leading_trailing_nans(df_with_gaps, empty_quality_report):
    """Test trim_leading_trailing_nans function."""
    required_columns = ["flow", "precipitation"]
    
    result = trim_leading_trailing_nans(
        (df_with_gaps, empty_quality_report),
        required_columns,
    )
    assert is_successful(result)
    
    df, quality_report = result.unwrap()
    
    # Check that DataFrame is unchanged
    assert df.equals(df_with_gaps)
    
    # Check that quality report is updated with valid periods
    basin_report = quality_report["basins"]["G1"]
    assert "valid_period" in basin_report
    assert "flow" in basin_report["valid_period"]
    assert "precipitation" in basin_report["valid_period"]
    
    # Flow has NaNs from index 3-5 (dates 2020-01-04 to 2020-01-06)
    assert basin_report["valid_period"]["flow"]["start"] == pd.Timestamp("2020-01-01")
    assert basin_report["valid_period"]["flow"]["end"] == pd.Timestamp("2020-01-20")
    
    # Precipitation has NaNs from index 10-15 (dates 2020-01-11 to 2020-01-16)
    assert basin_report["valid_period"]["precipitation"]["start"] == pd.Timestamp("2020-01-01")
    assert basin_report["valid_period"]["precipitation"]["end"] == pd.Timestamp("2020-01-20")

def test_trim_leading_trailing_nans_missing_date_column(empty_quality_report):
    """Test trim_leading_trailing_nans with missing date column."""
    df = pd.DataFrame({"gauge_id": ["G1"], "flow": [10.0]})
    required_columns = ["flow"]
    
    result = trim_leading_trailing_nans(
        (df, empty_quality_report),
        required_columns,
    )
    assert not is_successful(result)
    assert "DataFrame must contain a 'date' column" in result.failure()

# Tests for ensure_complete_date_range
def test_ensure_complete_date_range(df_with_gaps, empty_quality_report):
    """Test ensure_complete_date_range function."""
    # First we need to set up valid periods in the quality report
    required_columns = ["flow", "precipitation"]
    _, quality_report = trim_leading_trailing_nans(
        (df_with_gaps, empty_quality_report),
        required_columns,
    ).unwrap()
    
    result = ensure_complete_date_range(
        (df_with_gaps, quality_report),
    )
    assert is_successful(result)
    
    filled_df, updated_report = result.unwrap()
    
    # Check that result includes all dates
    assert len(filled_df) == 20  # All 20 days
    assert filled_df["gauge_id"].nunique() == 1
    
    # Check that quality report is updated
    assert "date_gaps" in updated_report["basins"]["G1"]
    assert updated_report["basins"]["G1"]["date_gaps"]["valid_start"] is not None
    assert updated_report["basins"]["G1"]["date_gaps"]["valid_end"] is not None

def test_ensure_complete_date_range_no_valid_period(df_with_gaps, empty_quality_report):
    """Test ensure_complete_date_range with no valid period."""
    # Set up an empty basins entry without valid_period
    quality_report = empty_quality_report
    quality_report["basins"] = {
        "G1": {
            "processing_steps": [],
        }
    }
    quality_report["excluded_basins"] = {}
    
    result = ensure_complete_date_range(
        (df_with_gaps, quality_report),
    )
    assert not is_successful(result)
    assert "Valid period not found for basin G1" in result.failure()

# Tests for check_missing_percentage
def test_check_missing_percentage_below_threshold(df_with_gaps, empty_quality_report):
    """Test check_missing_percentage with missing percentage below threshold."""
    required_columns = ["flow", "precipitation"]
    
    # In df_with_gaps, flow has 3/20 = 15% missing and precipitation has 6/20 = 30% missing
    result = check_missing_percentage(
        (df_with_gaps, empty_quality_report),
        required_columns,
        max_missing_pct=35.0,  # High threshold to pass both columns
    )
    assert is_successful(result)
    
    filtered_df, quality_report = result.unwrap()
    
    # DataFrame should be unchanged since all basins pass
    assert len(filtered_df) == len(df_with_gaps)
    
    # Check quality report
    basin_report = quality_report["basins"]["G1"]
    assert "missing_data" in basin_report
    assert basin_report["missing_data"]["flow"]["missing_percentage"] == 15.0
    assert basin_report["missing_data"]["precipitation"]["missing_percentage"] == 30.0
    assert "Passed missing percentage check" in basin_report["processing_steps"]

def test_check_missing_percentage_above_threshold(df_with_gaps, empty_quality_report):
    """Test check_missing_percentage with missing percentage above threshold."""
    required_columns = ["flow", "precipitation"]
    
    # In df_with_gaps, flow has 3/20 = 15% missing and precipitation has 6/20 = 30% missing
    result = check_missing_percentage(
        (df_with_gaps, empty_quality_report),
        required_columns,
        max_missing_pct=25.0,  # Threshold that precipitation will exceed
    )
    assert is_successful(result)
    
    filtered_df, quality_report = result.unwrap()
    
    # DataFrame should be empty since the basin fails
    assert len(filtered_df) == 0
    
    # Check quality report
    basin_report = quality_report["basins"]["G1"]
    assert "missing_data" in basin_report
    assert "Exceeded maximum missing percentage" in quality_report["excluded_basins"]["G1"]
    assert any("Failed: Exceeded maximum missing percentage" in step for step in basin_report["processing_steps"])

# Tests for check_missing_gaps
def test_check_missing_gaps_below_threshold(df_with_gaps, empty_quality_report):
    """Test check_missing_gaps with gap length below threshold."""
    required_columns = ["flow", "precipitation"]
    
    # In df_with_gaps, flow has a 3-day gap and precipitation has a 6-day gap
    result = check_missing_gaps(
        (df_with_gaps, empty_quality_report),
        required_columns,
        max_gap_length=10,  # High threshold to pass both columns
    )
    assert is_successful(result)
    
    filtered_df, quality_report = result.unwrap()
    
    # DataFrame should be unchanged since all basins pass
    assert len(filtered_df) == len(df_with_gaps)
    
    # Check quality report
    basin_report = quality_report["basins"]["G1"]
    assert "gaps" in basin_report
    assert basin_report["gaps"]["flow"]["max_gap_length"] == 3
    assert basin_report["gaps"]["precipitation"]["max_gap_length"] == 6
    assert "Passed gap check" in basin_report["processing_steps"]

def test_check_missing_gaps_above_threshold(df_with_gaps, empty_quality_report):
    """Test check_missing_gaps with gap length above threshold."""
    required_columns = ["flow", "precipitation"]
    
    # In df_with_gaps, flow has a 3-day gap and precipitation has a 6-day gap
    result = check_missing_gaps(
        (df_with_gaps, empty_quality_report),
        required_columns,
        max_gap_length=5,  # Threshold that precipitation gap will exceed
    )
    assert is_successful(result)
    
    filtered_df, quality_report = result.unwrap()
    
    # DataFrame should be empty since the basin fails
    assert len(filtered_df) == 0
    
    # Check quality report
    basin_report = quality_report["basins"]["G1"]
    assert "gaps" in basin_report
    assert "Found gaps exceeding maximum length" in quality_report["excluded_basins"]["G1"]
    assert any("Failed: Found gaps exceeding maximum length" in step for step in basin_report["processing_steps"])

def test_check_missing_gaps_missing_date_column(empty_quality_report):
    """Test check_missing_gaps with missing date column."""
    df = pd.DataFrame({"gauge_id": ["G1"], "flow": [10.0]})
    required_columns = ["flow"]
    
    result = check_missing_gaps(
        (df, empty_quality_report),
        required_columns,
        max_gap_length=5,
    )
    assert not is_successful(result)
    assert "DataFrame must contain a 'date' column" in result.failure()

def test_check_missing_gaps_non_datetime_date(empty_quality_report):
    """Test check_missing_gaps with non-datetime date column."""
    df = pd.DataFrame({
        "gauge_id": ["G1"],
        "date": ["2020-01-01"],  # String, not datetime
        "flow": [10.0]
    })
    required_columns = ["flow"]
    
    result = check_missing_gaps(
        (df, empty_quality_report),
        required_columns,
        max_gap_length=5,
    )
    assert not is_successful(result)
    assert "'date' column must be datetime type" in result.failure()

# Tests for impute_short_gaps
def test_impute_short_gaps(df_with_gaps, empty_quality_report):
    """Test impute_short_gaps function."""
    required_columns = ["flow", "precipitation"]
    
    # In df_with_gaps, flow has a 3-day gap and precipitation has a 6-day gap
    result = impute_short_gaps(
        (df_with_gaps, empty_quality_report),
        required_columns,
        max_imputation_gap_size=4,  # Only flow gap will be imputed
    )
    assert is_successful(result)
    
    imputed_df, quality_report = result.unwrap()
    
    # Flow should be imputed, precipitation should still have NaNs
    flow_na_count = imputed_df["flow"].isna().sum()
    precip_na_count = imputed_df["precipitation"].isna().sum()
    
    assert flow_na_count == 0  # Flow gap (3 days) should be imputed
    assert precip_na_count == 6  # Precipitation gap (6 days) should remain
    
    # Check quality report
    basin_report = quality_report["basins"]["G1"]
    assert "imputation_info" in basin_report
    assert basin_report["imputation_info"]["flow"]["short_gaps_count"] == 1
    assert basin_report["imputation_info"]["flow"]["imputed_values_count"] == 3
    assert basin_report["imputation_info"]["precipitation"]["short_gaps_count"] == 0
    assert basin_report["imputation_info"]["precipitation"]["long_gaps_count"] == 1

def test_impute_short_gaps_missing_date_column(empty_quality_report):
    """Test impute_short_gaps with missing date column."""
    df = pd.DataFrame({"gauge_id": ["G1"], "flow": [10.0]})
    required_columns = ["flow"]
    
    result = impute_short_gaps(
        (df, empty_quality_report),
        required_columns,
        max_imputation_gap_size=5,
    )
    assert not is_successful(result)
    assert "DataFrame must contain a 'date' column" in result.failure()

# Tests for process_basin
def test_process_basin(df_with_gaps, empty_quality_report):
    """Test process_basin with valid data."""
    basin_data = df_with_gaps.copy()
    basin_id = "G1"
    required_columns = ["flow", "precipitation"]
    
    # Exclude the gaps for this test to make a valid dataset
    basin_data.loc[3:5, "flow"] = 10.0
    basin_data.loc[10:15, "precipitation"] = 5.0
    
    result = process_basin(
        basin_data,
        basin_id,
        required_columns,
        min_train_years=0.01,  # Low threshold for this short test dataset
        train_prop=0.6,
        val_prop=0.2,
        test_prop=0.2,
        quality_report=empty_quality_report.copy(),
    )
    assert is_successful(result)
    
    processed_data = result.unwrap()
    assert len(processed_data) == len(basin_data)

def test_process_basin_insufficient_data(df_with_gaps, empty_quality_report):
    """Test process_basin with insufficient data."""
    basin_data = df_with_gaps.copy()
    basin_id = "G1"
    required_columns = ["flow", "precipitation"]
    
    result = process_basin(
        basin_data,
        basin_id,
        required_columns,
        min_train_years=2.0,  # Higher than the available data
        train_prop=0.6,
        val_prop=0.2,
        test_prop=0.2,
        quality_report=empty_quality_report.copy(),
    )
    assert not is_successful(result)
    assert "Insufficient training period" in result.failure()

# Tests for check_data_quality
def test_check_data_quality_valid(sample_df):
    """Test check_data_quality with valid data."""
    # Create a longer dataset to pass the min_train_years check
    dates = pd.date_range(start="2010-01-01", end="2020-12-31", freq="D")
    gauge_ids = ["G1"]
    
    data = []
    for gauge_id in gauge_ids:
        for date in dates:
            data.append({
                "gauge_id": gauge_id,
                "date": date,
                "flow": 10.0,
                "precipitation": 5.0,
            })
    
    long_df = pd.DataFrame(data)
    
    result = check_data_quality(
        long_df,
        required_columns=["flow", "precipitation"],
        min_train_years=5.0,
    )
    assert is_successful(result)
    
    processed_df, quality_report = result.unwrap()
    assert len(processed_df) > 0
    assert quality_report["retained_basins"] == 1

def test_check_data_quality_invalid_input():
    """Test check_data_quality with invalid input."""
    df = pd.DataFrame({"gauge_id": ["G1"], "flow": [10.0]})  # Missing date column
    
    result = check_data_quality(
        df,
        required_columns=["flow"],
    )
    assert not is_successful(result)
    assert "Input validation failed" in result.failure()

def test_check_data_quality_all_basins_filtered(df_with_gaps):
    """Test check_data_quality when all basins are filtered out."""
    result = check_data_quality(
        df_with_gaps,
        required_columns=["flow", "precipitation"],
        min_train_years=2.0,  # Too high for the test dataset
    )
    assert is_successful(result)
    
    processed_df, quality_report = result.unwrap()
    assert len(processed_df) == 0
    assert quality_report["retained_basins"] == 0
    assert "G1" in quality_report["excluded_basins"]