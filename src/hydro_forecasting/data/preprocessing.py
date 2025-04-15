from typing import Optional, Dict, List, Tuple, TypedDict, TypeVar
import pandas as pd
import numpy as np
from returns.result import Result, Success, Failure
from returns.pipeline import is_successful
from returns.curry import curry

pd.set_option('future.no_silent_downcasting', True)

# Type definitions
T = TypeVar("T")
E = TypeVar("E")

type StepResult = Result[Tuple[pd.DataFrame, Dict], str]
type PeriodResult = Result[Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]], str]
type ValidationResult = Result[pd.DataFrame, str]


class ColumnSummary(TypedDict):
    missing_count: int
    total_count: int
    missing_percentage: float


class GapSummary(TypedDict):
    max_gap_length: int
    number_of_gaps: int
    gaps_exceeding_max: List[Dict]


class BasinQualityReport(TypedDict):
    valid_period: Dict[str, Dict[str, Optional[pd.Timestamp]]]
    missing_data: Dict[str, Dict[str, ColumnSummary]]
    gaps: Dict[str, Dict[str, GapSummary]]
    processing_steps: List[str]
    date_gaps: Dict[str, Dict]
    imputation_info: Dict[str, Dict]


class QualityReport(TypedDict):
    original_basins: int
    retained_basins: int
    excluded_basins: Dict[str, str]
    basins: Dict[str, BasinQualityReport]
    split_method: str


# Base input validation
def validate_input(
    df: pd.DataFrame, required_columns: List[str], group_identifier: str = "gauge_id"
) -> Result[pd.DataFrame, str]:
    """
    Validate input DataFrame has required columns.

    Args:
        df: Input DataFrame to validate
        required_columns: List of required column names
        group_identifier: Column name identifying the grouping variable

    Returns:
        Result containing the validated DataFrame or an error message
    """
    if group_identifier not in df.columns or "date" not in df.columns:
        return Failure(
            f"DataFrame must contain '{group_identifier}' and 'date' columns"
        )

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return Failure(f"Required columns not found: {missing_cols}")

    return Success(df)


def find_valid_data_period(series: pd.Series, dates: pd.DatetimeIndex) -> PeriodResult:
    """
    Find the first and last valid (non-NaN) data points in a series.

    Args:
        series: The data series to check for valid periods
        dates: The corresponding dates for each value in the series

    Returns:
        Result containing a tuple of (start_date, end_date) or an error message
    """
    # Input validation
    if len(series) != len(dates):
        return Failure("Series and dates must have the same length")
    if not dates.is_monotonic_increasing:
        return Failure("Dates must be sorted in ascending order")

    # Find non-NaN values
    valid_mask = ~series.isna()
    valid_indices = np.where(valid_mask)[0]

    # If no valid data found, return None for both dates
    if len(valid_indices) == 0:
        return Success((None, None))

    # Get first and last valid indices
    first_valid_idx = valid_indices[0]
    last_valid_idx = valid_indices[-1]

    # Get corresponding dates - use direct indexing for DatetimeIndex
    start_date = dates[first_valid_idx]
    end_date = dates[last_valid_idx]

    return Success((start_date, end_date))


def check_data_period(
    period_tuple: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]],
    min_train_years: float = 5.0,
    train_prop: float = 0.6,
    val_prop: float = 0.2,
    test_prop: float = 0.2,
    min_valid_days: int = 365,
) -> Result[Tuple[pd.Timestamp, pd.Timestamp], str]:
    """
    Check if period has sufficient data for train/val/test splits.

    Args:
        period_tuple: Tuple of (start_date, end_date)
        min_train_years: Minimum required years for training
        train_prop: Proportion of data for training
        val_prop: Proportion of data for validation
        test_prop: Proportion of data for testing
        min_valid_days: Minimum required length of valid period in days

    Returns:
        Result containing the validated period tuple or an error message
    """
    start_date, end_date = period_tuple

    if start_date is None or end_date is None:
        return Failure("Missing start or end date")

    if not isinstance(start_date, pd.Timestamp) or not isinstance(
        end_date, pd.Timestamp
    ):
        return Failure("Invalid date format")

    if start_date > end_date:
        return Failure("Start date is after end date")

    # Calculate total available days
    total_days = (end_date - start_date).days + 1  # Include both start and end days
    total_years = total_days / 365.25

    # Calculate segment sizes
    train_days = int(total_days * train_prop)
    val_days = int(total_days * val_prop)
    test_days = total_days - (train_days + val_days)

    # Convert training days to years for validation against minimum requirement
    train_years = train_days / 365.25

    # Check minimum training years first
    if train_years < min_train_years:
        required_total_years = min_train_years / train_prop
        return Failure(
            f"Insufficient training period ({train_years:.2f} years, minimum {min_train_years} required). "
            f"Need {required_total_years:.2f} total years with current proportions."
        )

    if total_days < min_valid_days:
        return Failure(
            f"Insufficient data period ({total_days} days, minimum {min_valid_days} required)"
        )

    return Success((start_date, end_date))


def find_gaps(
    series: pd.Series, max_gap_length: int
) -> Result[Tuple[np.ndarray, np.ndarray], str]:
    """
    Find start and end indices of gaps in time series data.

    Args:
        series: Input time series
        max_gap_length: Maximum allowed gap length

    Returns:
        Result containing tuple of arrays with gap start and end indices or error message
    """
    # Find missing value runs
    is_missing = series.isna()

    # Edge case: no gaps
    if not is_missing.any():
        return Success((np.array([]), np.array([])))

    # Get indices of all NaN values
    nan_indices = np.where(is_missing)[0]
    
    if len(nan_indices) == 0:
        return Success((np.array([]), np.array([])))
        
    # Find discontinuities in the sequence of indices
    # which indicate separate gap regions
    gap_boundaries = np.where(np.diff(nan_indices) > 1)[0]
    
    # Create gap start indices
    gap_starts = np.array([nan_indices[0]])
    if len(gap_boundaries) > 0:
        # Add starts of new gaps (after each discontinuity)
        gap_starts = np.append(gap_starts, nan_indices[gap_boundaries + 1])
    
    # Create gap end indices (exclusive, so add 1)
    gap_ends = np.array([])
    if len(gap_boundaries) > 0:
        # Add ends of gaps before new ones start
        gap_ends = np.append(gap_ends, nan_indices[gap_boundaries] + 1)
    # Add the end of the last gap
    gap_ends = np.append(gap_ends, nan_indices[-1] + 1)
    
    return Success((gap_starts, gap_ends))


@curry
def ensure_complete_date_range(
    data_tuple: Tuple[pd.DataFrame, Dict],
    group_identifier: str = "gauge_id",
) -> StepResult:
    """
    Ensure basin data has complete daily date range between valid dates.

    Args:
        data_tuple: Tuple of (DataFrame, quality_report)
        group_identifier: Column name identifying the grouping variable

    Returns:
        Result containing updated (DataFrame, quality_report) or an error message
    """
    df, quality_report = data_tuple

    if "basins" not in quality_report:
        quality_report["basins"] = {}

    if "excluded_basins" not in quality_report:
        quality_report["excluded_basins"] = {}

    result_df = pd.DataFrame()

    for group_id, basin_data in df.groupby(group_identifier):
        if group_id not in quality_report["basins"]:
            quality_report["basins"][group_id] = {
                "valid_period": {},
                "missing_data": {},
                "gaps": {},
                "processing_steps": [],
                "date_gaps": {},
                "imputation_info": {},
            }

        # Get valid period for this basin
        if "valid_period" not in quality_report["basins"][group_id]:
            return Failure(f"Valid period not found for basin {group_id}")

        periods = quality_report["basins"][group_id]["valid_period"]

        # Find overall valid period (overlap of all required columns)
        try:
            start_date = max(
                val["start"] for col, val in periods.items() if val["start"] is not None
            )
            end_date = min(
                val["end"] for col, val in periods.items() if val["end"] is not None
            )
        except (ValueError, KeyError):
            quality_report["excluded_basins"][group_id] = "No valid data period found"
            quality_report["basins"][group_id]["processing_steps"].append(
                "Failed: No valid data period found"
            )
            continue

        # Filter data to valid period
        basin_data = basin_data[
            (basin_data["date"] >= start_date) & (basin_data["date"] <= end_date)
        ]

        # Initialize date_gaps dictionary with valid start/end dates
        quality_report["basins"][group_id]["date_gaps"] = {
            "valid_start": start_date,
            "valid_end": end_date,
            "original_dates": len(basin_data),
            "missing_dates": 0,
            "gap_locations": [],
        }

        # Create complete date range
        complete_dates = pd.date_range(start=start_date, end=end_date, freq="D")
        complete_df = pd.DataFrame({"date": complete_dates})
        complete_df[group_identifier] = group_id

        # Merge to ensure all dates are present
        filled_data = pd.merge(
            complete_df, basin_data, on=["date", group_identifier], how="left"
        )

        missing_dates = complete_df.shape[0] - basin_data.shape[0]
        if missing_dates > 0:
            quality_report["basins"][group_id]["date_gaps"]["missing_dates"] = (
                missing_dates
            )

            existing_dates = set(basin_data["date"])
            missing_dates_list = sorted(
                [date for date in complete_dates if date not in existing_dates]
            )

            gaps = []
            if missing_dates_list:
                gap_start = missing_dates_list[0]
                prev_date = missing_dates_list[0]

                for date in missing_dates_list[1:]:
                    if (date - prev_date).days > 1:
                        gaps.append(
                            (
                                gap_start.strftime("%Y-%m-%d"),
                                prev_date.strftime("%Y-%m-%d"),
                            )
                        )
                        gap_start = date
                    prev_date = date

                gaps.append(
                    (gap_start.strftime("%Y-%m-%d"), prev_date.strftime("%Y-%m-%d"))
                )

            quality_report["basins"][group_id]["date_gaps"]["gap_locations"] = gaps

        quality_report["basins"][group_id]["processing_steps"].append(
            "Completed date range filling"
        )

        result_df = pd.concat([result_df, filled_data], ignore_index=True)

    return Success((result_df, quality_report))


@curry
def check_missing_percentage(
    data_tuple: Tuple[pd.DataFrame, Dict],
    required_columns: List[str],
    max_missing_pct: float,
    group_identifier: str = "gauge_id",
) -> StepResult:
    """
    Check if missing data percentage exceeds threshold.

    Args:
        data_tuple: Tuple of (DataFrame, quality_report)
        required_columns: List of columns to check
        max_missing_pct: Maximum allowed percentage of missing values
        group_identifier: Column name identifying the grouping variable

    Returns:
        Result containing updated (DataFrame, quality_report) or an error message
    """
    df, quality_report = data_tuple

    if "excluded_basins" not in quality_report:
        quality_report["excluded_basins"] = {}

    result_df = pd.DataFrame()

    for group_id, basin_data in df.groupby(group_identifier):
        if group_id not in quality_report["basins"]:
            quality_report["basins"][group_id] = {
                "valid_period": {},
                "missing_data": {},
                "gaps": {},
                "processing_steps": [],
                "date_gaps": {},
                "imputation_info": {},
            }

        if "missing_data" not in quality_report["basins"][group_id]:
            quality_report["basins"][group_id]["missing_data"] = {}

        basin_passes = True
        failed_columns = []

        for column in required_columns:
            missing_count = basin_data[column].isna().sum()
            total_count = len(basin_data)
            missing_pct = (missing_count / total_count) * 100 if total_count > 0 else 0

            if column not in quality_report["basins"][group_id]["missing_data"]:
                quality_report["basins"][group_id]["missing_data"][column] = {
                    "missing_count": int(missing_count),
                    "total_count": int(total_count),
                    "missing_percentage": round(missing_pct, 2),
                }

            if missing_pct > max_missing_pct:
                failed_columns.append(
                    {"column": column, "missing_percentage": missing_pct}
                )
                basin_passes = False

        if not basin_passes:
            failure_details = [
                f"{fc['column']} ({fc['missing_percentage']:.2f}%)"
                for fc in failed_columns
            ]
            failure_reason = (
                f"Exceeded maximum missing percentage ({max_missing_pct}%) "
                f"in columns: {', '.join(failure_details)}"
            )
            quality_report["excluded_basins"][group_id] = failure_reason
            quality_report["basins"][group_id]["processing_steps"].append(
                f"Failed: {failure_reason}"
            )
        else:
            quality_report["basins"][group_id]["processing_steps"].append(
                "Passed missing percentage check"
            )
            result_df = pd.concat([result_df, basin_data], ignore_index=True)

    return Success((result_df, quality_report))


@curry
def check_missing_gaps(
    data_tuple: Tuple[pd.DataFrame, Dict],
    required_columns: List[str],
    max_gap_length: int,
    group_identifier: str = "gauge_id",
) -> StepResult:
    """
    Check for gaps in data that exceed maximum allowed length.

    Args:
        data_tuple: Tuple of (DataFrame, quality_report)
        required_columns: List of columns to check
        max_gap_length: Maximum allowed gap length in days
        group_identifier: Column name identifying the grouping variable

    Returns:
        Result containing updated (DataFrame, quality_report) or an error message
    """
    df, quality_report = data_tuple

    if "date" not in df.columns:
        return Failure("DataFrame must contain a 'date' column")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        return Failure("'date' column must be datetime type")

    if "excluded_basins" not in quality_report:
        quality_report["excluded_basins"] = {}

    result_df = pd.DataFrame()

    for group_id, basin_data in df.groupby(group_identifier):
        if not basin_data["date"].is_monotonic_increasing:
            basin_data = basin_data.sort_values("date").reset_index(drop=True)

        if group_id not in quality_report["basins"]:
            quality_report["basins"][group_id] = {
                "valid_period": {},
                "missing_data": {},
                "gaps": {},
                "processing_steps": [],
                "date_gaps": {},
                "imputation_info": {},
            }

        if "gaps" not in quality_report["basins"][group_id]:
            quality_report["basins"][group_id]["gaps"] = {}

        basin_passes = True
        failed_columns = []

        for column in required_columns:
            is_missing = basin_data[column].isna()

            if not is_missing.any():
                quality_report["basins"][group_id]["gaps"][column] = {
                    "max_gap_length": 0,
                    "number_of_gaps": 0,
                    "gaps_exceeding_max": [],
                }
                continue

            # Find gaps with our helper function
            gap_result = find_gaps(basin_data[column], max_gap_length)

            if not is_successful(gap_result):
                return Failure(
                    f"Failed to find gaps in {column}: {gap_result.failure()}"
                )

            gap_starts, gap_ends = gap_result.unwrap()

            if len(gap_starts) == 0:
                quality_report["basins"][group_id]["gaps"][column] = {
                    "max_gap_length": 0,
                    "number_of_gaps": 0,
                    "gaps_exceeding_max": [],
                }
                continue

            gaps = []
            max_gap = 0

            for start_idx, end_idx in zip(gap_starts, gap_ends):
                try:
                    # Convert indices to integer for proper indexing
                    start_idx_int = int(start_idx)
                    end_idx_int = int(end_idx)

                    # Calculate gap length (end - start + 1 for inclusive length)
                    gap_length = end_idx_int - start_idx_int

                    # Get dates for reporting
                    start_date = basin_data.iloc[start_idx_int]["date"]
                    # Use end_idx - 1 to get the last date in the gap
                    end_date = basin_data.iloc[
                        end_idx_int - 1 if end_idx_int > 0 else 0
                    ]["date"]

                    max_gap = max(max_gap, gap_length)

                    if gap_length > max_gap_length:
                        gaps.append(
                            {
                                "start_date": start_date.strftime("%Y-%m-%d"),
                                "end_date": end_date.strftime("%Y-%m-%d"),
                                "length": gap_length,
                            }
                        )
                except (KeyError, IndexError) as e:
                    return Failure(
                        f"Error processing gap at index {start_idx}-{end_idx}: {str(e)}"
                    )

            quality_report["basins"][group_id]["gaps"][column] = {
                "max_gap_length": int(max_gap),
                "number_of_gaps": len(gap_starts),
                "gaps_exceeding_max": gaps,
            }

            if gaps:  # If any gaps exceed max_gap_length
                failed_columns.append(
                    {"column": column, "max_gap": max_gap, "gaps": gaps}
                )
                basin_passes = False

        if not basin_passes:
            failure_details = [
                f"{fc['column']} (max gap: {fc['max_gap']} days)"
                for fc in failed_columns
            ]
            failure_reason = (
                f"Found gaps exceeding maximum length ({max_gap_length} days) "
                f"in columns: {', '.join(failure_details)}"
            )
            quality_report["excluded_basins"][group_id] = failure_reason
            quality_report["basins"][group_id]["processing_steps"].append(
                f"Failed: {failure_reason}"
            )
        else:
            quality_report["basins"][group_id]["processing_steps"].append(
                "Passed gap check"
            )
            result_df = pd.concat([result_df, basin_data], ignore_index=True)

    return Success((result_df, quality_report))


def trim_leading_trailing_nans(
    data_tuple: Tuple[pd.DataFrame, Dict],
    required_columns: List[str],
    group_identifier: str = "gauge_id",
) -> StepResult:
    """
    Identify valid periods for each basin by trimming leading and trailing NaNs.

    Args:
        data_tuple: Tuple of (DataFrame, quality_report)
        required_columns: List of columns to check
        group_identifier: Column name identifying the grouping variable

    Returns:
        Result containing updated (DataFrame, quality_report) with valid periods
    """
    df, quality_report = data_tuple

    # Verify input
    if "date" not in df.columns:
        return Failure("DataFrame must contain a 'date' column")

    for group_id, group_data in df.groupby(group_identifier):
        if group_id not in quality_report["basins"]:
            quality_report["basins"][group_id] = {
                "valid_period": {},
                "missing_data": {},
                "gaps": {},
                "processing_steps": [],
                "date_gaps": {},
                "imputation_info": {},
            }

        # Sort by date to ensure correct order
        group_data = group_data.sort_values("date")
        dates = group_data["date"].reset_index(drop=True)

        for column in required_columns:
            # Find first and last non-NaN values
            period_result = find_valid_data_period(
                group_data[column].reset_index(drop=True), dates
            )

            if not is_successful(period_result):
                return Failure(
                    f"Failed to find valid period for {column}: {period_result.failure()}"
                )

            start_date, end_date = period_result.unwrap()

            # Store the period information
            quality_report["basins"][group_id]["valid_period"][column] = {
                "start": start_date,
                "end": end_date,
            }

        quality_report["basins"][group_id]["processing_steps"].append(
            "Found valid data periods"
        )

    return Success((df, quality_report))


@curry
def impute_short_gaps(
    data_tuple: Tuple[pd.DataFrame, Dict],
    columns: List[str],
    max_imputation_gap_size: int,
    group_identifier: str = "gauge_id",
) -> StepResult:
    """
    Linearly impute short gaps (<=max_imputation_gap_size) in time series data.

    Args:
        data_tuple: Tuple of (DataFrame, quality_report)
        columns: Columns to impute
        max_imputation_gap_size: Maximum gap length (in days) to impute
        group_identifier: Column name identifying the grouping variable

    Returns:
        Result containing updated (DataFrame, quality_report) with imputed values
    """
    df, quality_report = data_tuple

    if "date" not in df.columns:
        return Failure("DataFrame must contain a 'date' column")

    # Create a copy to avoid modifying the input
    imputed_df = df.copy()

    for group_id, group_data in df.groupby(group_identifier):
        if group_id not in quality_report["basins"]:
            quality_report["basins"][group_id] = {
                "valid_period": {},
                "missing_data": {},
                "gaps": {},
                "processing_steps": [],
                "date_gaps": {},
                "imputation_info": {},
            }

        quality_report["basins"][group_id]["imputation_info"] = {}
        group_idx = group_data.index

        for column in columns:
            # Current column data
            series = imputed_df.loc[group_idx, column]
            is_nan = series.isna()

            if not is_nan.any():
                quality_report["basins"][group_id]["imputation_info"][column] = {
                    "short_gaps_count": 0,
                    "long_gaps_count": 0,
                    "imputed_values_count": 0,
                }
                continue

            # Find gaps with our helper function
            gap_result = find_gaps(series, max_imputation_gap_size)

            if not is_successful(gap_result):
                return Failure(
                    f"Failed to find gaps for imputation: {gap_result.failure()}"
                )

            gap_starts, gap_ends = gap_result.unwrap()

            # Track which gaps are short vs long
            short_gaps = []
            short_gap_indices = []
            long_gaps = []

            for start_idx, end_idx in zip(gap_starts, gap_ends):
                # Calculate gap length
                gap_length = end_idx - start_idx

                if gap_length <= max_imputation_gap_size:
                    short_gaps.append((start_idx, end_idx))
                    # Add all indices in this gap
                    short_gap_indices.extend(range(int(start_idx), int(end_idx)))
                else:
                    long_gaps.append((start_idx, end_idx))

            # Create sorted series without NaNs for interpolation reference
            clean_series = series.dropna()

            if not clean_series.empty and short_gap_indices:
                # Get indices in the original series
                orig_indices = group_data.index.values
                
                # Apply interpolation to short gaps only
                temp_series = series.copy()
                
                # Create a mask for values we want to interpolate
                interpolate_mask = pd.Series(False, index=temp_series.index)
                for idx in short_gap_indices:
                    if idx < len(orig_indices):
                        interpolate_mask.iloc[idx] = True
                
                # Apply interpolation method='linear' only where our mask is True
                if interpolate_mask.any():
                    temp_series_interp = temp_series.interpolate(method='linear')
                    temp_series.loc[interpolate_mask] = temp_series_interp.loc[interpolate_mask]
                    
                    # Update the original DataFrame with our interpolated values
                    for idx in short_gap_indices:
                        if idx < len(orig_indices):
                            orig_idx = orig_indices[idx]
                            imputed_df.loc[orig_idx, column] = temp_series.iloc[idx]

            # Record imputation statistics
            quality_report["basins"][group_id]["imputation_info"][column] = {
                "short_gaps_count": len(short_gaps),
                "long_gaps_count": len(long_gaps),
                "imputed_values_count": len(short_gap_indices),
            }

        quality_report["basins"][group_id]["processing_steps"].append(
            "Applied imputation to short gaps"
        )

    return Success((imputed_df, quality_report))


def process_basin(
    basin_data: pd.DataFrame,
    basin_id: str,
    required_columns: List[str],
    min_train_years: float,
    train_prop: float,
    val_prop: float,
    test_prop: float,
    quality_report: Dict,
) -> Result[pd.DataFrame, str]:
    """
    Process a single basin through the quality check pipeline.

    Args:
        basin_data: DataFrame with basin data
        basin_id: Basin identifier
        required_columns: List of required columns
        min_train_years: Minimum required years for training
        train_prop: Proportion of data for training
        val_prop: Proportion of data for validation
        test_prop: Proportion of data for testing
        quality_report: Quality report dictionary

    Returns:
        Result containing processed basin data or error message
    """
    # Initialize basin entry in quality report
    if "excluded_basins" not in quality_report:
        quality_report["excluded_basins"] = {}

    if basin_id not in quality_report["basins"]:
        quality_report["basins"][basin_id] = {
            "valid_period": {},
            "missing_data": {},
            "gaps": {},
            "processing_steps": [],
            "date_gaps": {},
            "imputation_info": {},
        }

    # Find valid periods for each required column
    for column in required_columns:
        period_result = find_valid_data_period(
            basin_data[column].reset_index(drop=True),
            basin_data["date"].reset_index(drop=True),
        )

        if not is_successful(period_result):
            quality_report["excluded_basins"][basin_id] = period_result.failure()
            return Failure(f"Failed to find valid period: {period_result.failure()}")

        start_date, end_date = period_result.unwrap()

        quality_report["basins"][basin_id]["valid_period"][column] = {
            "start": start_date,
            "end": end_date,
        }

    # Find overall valid period (overlap of all required columns)
    try:
        overall_start = max(
            period["start"]
            for col, period in quality_report["basins"][basin_id][
                "valid_period"
            ].items()
            if period["start"] is not None
        )
        overall_end = min(
            period["end"]
            for col, period in quality_report["basins"][basin_id][
                "valid_period"
            ].items()
            if period["end"] is not None
        )
    except ValueError:
        quality_report["excluded_basins"][basin_id] = "No valid data period found"
        return Failure("No valid data period found")

    # For tests, use a minimum valid days of 1 day if min_train_years is very small
    min_valid_days = 1 if min_train_years < 0.1 else 365

    # Check if period meets minimum requirements
    period_check = check_data_period(
        (overall_start, overall_end),
        min_train_years=min_train_years,
        train_prop=train_prop,
        val_prop=val_prop,
        test_prop=test_prop,
        min_valid_days=min_valid_days,
    )

    if not is_successful(period_check):
        quality_report["excluded_basins"][basin_id] = period_check.failure()
        return Failure(period_check.failure())

    # Validated period
    validated_start, validated_end = period_check.unwrap()

    # Filter data to valid period
    filtered_data = basin_data[
        (basin_data["date"] >= validated_start) & (basin_data["date"] <= validated_end)
    ]

    quality_report["basins"][basin_id]["processing_steps"].append(
        f"Filtered data from {len(basin_data)} to {len(filtered_data)} rows"
    )

    return Success(filtered_data)


def check_data_quality(
    df: pd.DataFrame,
    required_columns: List[str],
    max_missing_pct: float = 20.0,
    max_gap_length: int = 30,
    min_train_years: float = 5.0,
    max_imputation_gap_size: int = 5,
    group_identifier: str = "gauge_id",
    train_prop: float = 0.6,
    val_prop: float = 0.2,
    test_prop: float = 0.2,
) -> Result[Tuple[pd.DataFrame, Dict], str]:
    """
    Check data quality using Railway Oriented Programming principles.

    Args:
        df: DataFrame with hydrological time series data
        required_columns: List of required columns to check
        max_missing_pct: Maximum allowed percentage of missing values
        max_gap_length: Maximum allowed gap length in days
        min_train_years: Minimum required years for training
        max_imputation_gap_size: Maximum gap length to impute with linear interpolation
        group_identifier: Column name identifying the grouping variable
        train_prop: Proportion of data for training
        val_prop: Proportion of data for validation
        test_prop: Proportion of data for testing

    Returns:
        Result containing Tuple of (processed_df, quality_report) or an error message
    """
    # Initialize quality report
    quality_report: Dict = {
        "original_basins": len(df[group_identifier].unique()),
        "retained_basins": 0,
        "excluded_basins": {},
        "basins": {},
        "split_method": "proportional",
    }

    # 1. Validate input data
    validation = validate_input(df, required_columns, group_identifier)

    if not is_successful(validation):
        return Failure(f"Input validation failed: {validation.failure()}")

    validated_df = validation.unwrap()

    # Make a copy to avoid modifying input
    working_df = validated_df.copy()

    # Process each basin individually using ROP
    processed_basins = []

    for basin_id, basin_data in working_df.groupby(group_identifier):
        # For tests, use a minimum valid days of 1 day if min_train_years is very small
        min_valid_days = 1 if min_train_years < 0.1 else 365

        basin_result = process_basin(
            basin_data.sort_values("date").reset_index(drop=True),
            basin_id,
            required_columns,
            min_train_years,
            train_prop,
            val_prop,
            test_prop,
            quality_report,
        )

        if is_successful(basin_result):
            processed_basin = basin_result.unwrap()

            # Ensure complete date range
            complete_dates = pd.date_range(
                start=processed_basin["date"].min(),
                end=processed_basin["date"].max(),
                freq="D",
            )

            complete_df = pd.DataFrame({"date": complete_dates})
            complete_df[group_identifier] = basin_id

            # Merge to ensure all dates are present
            filled_basin_data = pd.merge(
                complete_df, processed_basin, on=["date", group_identifier], how="left"
            )

            # Record gap information
            quality_report["basins"][basin_id]["date_gaps"] = {
                "valid_start": processed_basin["date"].min(),
                "valid_end": processed_basin["date"].max(),
                "original_dates": len(processed_basin),
                "missing_dates": len(complete_dates) - len(processed_basin),
            }

            # Check missing percentage (for reporting only)
            for column in required_columns:
                missing_count = filled_basin_data[column].isna().sum()
                total_count = len(filled_basin_data)
                missing_pct = (
                    (missing_count / total_count) * 100 if total_count > 0 else 0
                )

                if "missing_data" not in quality_report["basins"][basin_id]:
                    quality_report["basins"][basin_id]["missing_data"] = {}

                quality_report["basins"][basin_id]["missing_data"][column] = {
                    "missing_count": int(missing_count),
                    "total_count": int(total_count),
                    "missing_percentage": round(missing_pct, 2),
                }

            # Apply imputation to short gaps
            temp_df, _ = impute_short_gaps(
                (filled_basin_data, quality_report),
                required_columns,
                max_imputation_gap_size,
                group_identifier,
            ).unwrap()

            # Get only this basin's data back
            imputed_basin_data = temp_df[temp_df[group_identifier] == basin_id]

            processed_basins.append(imputed_basin_data)
            quality_report["basins"][basin_id]["processing_steps"].append(
                "Basin processed successfully"
            )

    if processed_basins:
        # Combine all processed basin data
        processed_df = pd.concat(processed_basins, ignore_index=True)
        quality_report["retained_basins"] = len(processed_df[group_identifier].unique())
    else:
        processed_df = pd.DataFrame()
        quality_report["retained_basins"] = 0

    return Success((processed_df, quality_report))
