import json
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from returns.result import Result, Success, Failure

from sklearn.pipeline import Pipeline
from ..preprocessing.grouped import GroupedPipeline
from ..preprocessing import Config


def find_gaps(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Find start and end indices of gaps in a time series.

    Args:
        series: Input time series

    Returns:
        Tuple of arrays with gap start and end indices
    """
    missing = series.isna()
    if not missing.any():
        return np.array([]), np.array([])

    nan_idxs = np.where(missing)[0]
    boundaries = np.where(np.diff(nan_idxs) > 1)[0]

    starts = np.insert(nan_idxs[boundaries + 1], 0, nan_idxs[0])
    ends = np.append(nan_idxs[boundaries] + 1, nan_idxs[-1] + 1)
    return starts, ends


def impute_short_gaps(
    df: pd.DataFrame, columns: list[str], max_gap: int, report: dict[str, any]
) -> Result[tuple[pd.DataFrame, dict[str, any]], str]:
    """
    Linearly impute short NaN gaps up to max_gap length.

    Args:
        df: DataFrame containing time series data
        columns: List of columns to impute
        max_gap: Maximum gap size to impute
        report: Dictionary for reporting statistics and steps

    Returns:
        Result with tuple of imputed DataFrame and updated report, or error message
    """
    try:
        df_out = df.copy()
        for col in columns:
            series = df_out[col]
            starts, ends = find_gaps(series)
            short_idxs = []
            for s, e in zip(starts, ends):
                if (e - s) <= max_gap:
                    short_idxs.extend(range(s, e))
            if short_idxs:
                interp = series.interpolate(method="linear")
                series.iloc[short_idxs] = interp.iloc[short_idxs]
                df_out[col] = series
            report["imputation_info"][col] = {
                "short_gaps_count": len(starts),
                "imputed_values_count": len(short_idxs),
            }
        report["processing_steps"].append("Imputed short gaps")
        return Success((df_out, report))
    except Exception as e:
        return Failure(f"Error imputing short gaps: {str(e)}")


def check_required_columns(
    df: pd.DataFrame, required: list[str]
) -> Result[pd.DataFrame, str]:
    """
    Ensure DataFrame contains all required cols plus 'date'.

    Args:
        df: DataFrame to check
        required: List of required column names

    Returns:
        Result with DataFrame if all columns exist, or error message
    """
    missing = [c for c in required + ["date"] if c not in df.columns]
    if missing:
        return Failure(f"Missing columns: {missing}")
    return Success(df)


def compute_valid_periods(
    df: pd.DataFrame, columns: list[str], report: dict[str, any]
) -> Result[dict[str, any], str]:
    """
    Populate report['valid_period'] for each column.

    Args:
        df: DataFrame containing time series data
        columns: Columns to compute valid periods for
        report: Report dictionary to update

    Returns:
        Result with updated report or error message
    """
    try:
        for col in columns:
            non_null_dates = df.loc[df[col].notna(), "date"]
            if non_null_dates.empty:
                report["valid_period"][col] = {"start": None, "end": None}
            else:
                report["valid_period"][col] = {
                    "start": non_null_dates.min().strftime("%Y-%m-%d"),
                    "end": non_null_dates.max().strftime("%Y-%m-%d"),
                }
        return Success(report)
    except Exception as e:
        return Failure(f"Error computing valid periods: {str(e)}")


def filter_valid_period(
    df: pd.DataFrame, report: dict[str, any]
) -> Result[pd.DataFrame, str]:
    """
    Trim df to the intersection of all valid periods in report.

    Args:
        df: DataFrame to filter
        report: Report containing valid_period information

    Returns:
        Result with filtered DataFrame or error message
    """
    try:
        starts = [
            pd.to_datetime(v["start"])
            for v in report["valid_period"].values()
            if v["start"]
        ]
        ends = [
            pd.to_datetime(v["end"])
            for v in report["valid_period"].values()
            if v["end"]
        ]
        if not (starts and ends):
            return Failure("No valid periods found for filtering")

        overall_start = max(starts)
        overall_end = min(ends)
        filtered_df = (
            df[(df["date"] >= overall_start) & (df["date"] <= overall_end)]
            .sort_values("date")
            .reset_index(drop=True)
        )

        if filtered_df.empty:
            return Failure("No overlapping valid period")

        return Success(filtered_df)
    except Exception as e:
        return Failure(f"Error filtering valid period: {str(e)}")


def validate_training_data(
    df: pd.DataFrame, config: Config, report: dict[str, any]
) -> Result[dict[str, any], str]:
    """
    Check if df has enough history per config requirements.

    Args:
        df: DataFrame to validate
        config: Configuration with training requirements
        report: Report to update

    Returns:
        Result with updated report or error message
    """
    try:
        days = (df["date"].iloc[-1] - df["date"].iloc[0]).days + 1
        train_days = int(days * config.train_prop)
        min_days = int(config.min_train_years * 365.25)
        if train_days < min_days:
            return Failure(
                f"Insufficient training data: requires {config.min_train_years} years"
            )
        report["processing_steps"].append("Validated training period")
        return Success(report)
    except Exception as e:
        return Failure(f"Error validating training data: {str(e)}")



def prepare_report() -> dict[str, any]:
    """
    Initialize processing report structure.

    Returns:
        Empty report dictionary
    """
    return {
        "valid_period": {},
        "processing_steps": [],
        "imputation_info": {},
    }


def initialize_imputation_info(
    report: dict[str, any], columns: list[str]
) -> dict[str, any]:
    """
    Initialize imputation tracking in report.

    Args:
        report: Report to update
        columns: Columns to track imputation for

    Returns:
        Report with initialized imputation info
    """
    for col in columns:
        report["imputation_info"][col] = {
            "short_gaps_count": 0,
            "imputed_values_count": 0,
        }
    return report



def process_basin(
    basin: pd.DataFrame ,
    config: Config,
    fitted_pipelines: dict[str, Pipeline | GroupedPipeline] | None = None,
) -> Result[tuple[dict[str, any], Path, Path], str]:

