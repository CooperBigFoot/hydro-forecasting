from __future__ import annotations


import json
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from functools import wraps
from typing import Optional, TYPE_CHECKING, Union

from returns.result import Result, Success, Failure

if TYPE_CHECKING:
    from .preprocessing import ProcessingConfig


def step(name: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(
            context_result: Result[CleanContext, str],
        ) -> Result[CleanContext, str]:
            # short-circuit if already failed
            if isinstance(context_result, Failure):
                return context_result

            # safe to unwrap
            context = context_result.unwrap()
            try:
                # run the actual step
                ctx = fn(context)
                # record that we ran it
                ctx.report.processing_steps.append(name)
                return Success(ctx)
            except Exception as e:
                return Failure(f"{fn.__name__} failed: {e}")

        return wrapper

    return decorator


@dataclass
class BasinQualityReport:
    valid_period: dict[str, dict[str, Optional[str]]]
    processing_steps: list[str]
    imputation_info: dict[str, dict[str, int]]


@dataclass
class CleanContext:
    df: pd.DataFrame
    config: "ProcessingConfig" 
    report: BasinQualityReport


def find_gaps(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    missing = series.isna()
    if not missing.any():
        return np.array([]), np.array([])
    nan_idxs = np.where(missing)[0]
    boundaries = np.where(np.diff(nan_idxs) > 1)[0]
    starts = np.insert(nan_idxs[boundaries + 1], 0, nan_idxs[0])
    ends = np.append(nan_idxs[boundaries] + 1, nan_idxs[-1] + 1)
    return starts, ends


@step("Initialized report")
def init_report(context: CleanContext) -> CleanContext:
    cols = context.config.required_columns
    # reset our dataclass fields
    context.report.valid_period = {}
    context.report.processing_steps = []
    context.report.imputation_info = {
        col: {"short_gaps_count": 0, "imputed_values_count": 0} for col in cols
    }
    return context


@step("Checked required columns")
def check_required_columns(context: CleanContext) -> CleanContext:
    required = context.config.required_columns
    missing = [c for c in required + ["date"] if c not in context.df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return context


@step("Trimmed leading/trailing NaN rows")
def trim_leading_trailing_nans(context: CleanContext) -> CleanContext:
    cols = context.config.required_columns
    mask = context.df[cols].notna().any(axis=1)
    if not mask.any():
        raise ValueError("All rows are NaN in required columns")
    first, last = mask.idxmax(), mask[::-1].idxmax()
    context.df = context.df.loc[first:last].reset_index(drop=True)
    return context


@step("Imputed short gaps")
def impute_short_gaps_step(context: CleanContext) -> CleanContext:
    cols = context.config.required_columns
    max_imputation_gap_size = context.config.max_imputation_gap_size
    df = context.df.copy()

    # count segments ≤ max_imputation_gap_size
    segment_counts: dict[str, int] = {}
    for col in cols:
        starts, ends = find_gaps(df[col])
        segment_counts[col] = sum(
            1 for s, e in zip(starts, ends) if (e - s) <= max_imputation_gap_size
        )

    # record before/after nulls and interpolate
    before_na = df[cols].isna().sum()
    df[cols] = df[cols].interpolate(
        method="linear",
        limit=max_imputation_gap_size,
        limit_direction="both",
    )
    after_na = df[cols].isna().sum()

    # write into the report dataclass
    for col in cols:
        imputed = int(before_na[col] - after_na[col])
        context.report.imputation_info[col] = {
            "short_gaps_count": segment_counts[col],
            "imputed_values_count": imputed,
        }

    context.df = df
    return context


@step("Computed valid periods")
def compute_valid_periods_step(context: CleanContext) -> CleanContext:
    cols = context.config.required_columns
    for col in cols:
        non_null = context.df.loc[context.df[col].notna(), "date"]
        if non_null.empty:
            context.report.valid_period[col] = {"start": None, "end": None}
        else:
            context.report.valid_period[col] = {
                "start": non_null.min().strftime("%Y-%m-%d"),
                "end": non_null.max().strftime("%Y-%m-%d"),
            }
    return context


@step("Filtered valid period")
def filter_valid_period_step(context: CleanContext) -> CleanContext:
    periods = context.report.valid_period.values()
    starts = [pd.to_datetime(v["start"]) for v in periods if v["start"]]
    ends = [pd.to_datetime(v["end"]) for v in periods if v["end"]]
    if not (starts and ends):
        raise ValueError("No valid periods found for filtering")
    overall_start = max(starts)
    overall_end = min(ends)

    filtered = (
        context.df.loc[
            (context.df["date"] >= overall_start) & (context.df["date"] <= overall_end)
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if filtered.empty:
        raise ValueError("No overlapping valid period")
    context.df = filtered
    return context


@step("Validated training period")
def validate_training_data_step(context: CleanContext) -> CleanContext:
    df = context.df
    cfg = context.config
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days + 1
    train_days = int(days * cfg.train_prop)
    min_days = int(cfg.min_train_years * 365.25)
    if train_days < min_days:
        raise ValueError(
            f"Insufficient training data: requires {cfg.min_train_years} years"
        )
    return context


def clean_data(
    df: pd.DataFrame,
    config: ProcessingConfig,
) -> Result[tuple[pd.DataFrame, BasinQualityReport], str]:
    """
    Runs a series of quality‐control steps on your time series, short‐circuits on any failure,
    and returns either Failure(error_message) or Success((cleaned_df, report)).
    """
    # start with an empty report dataclass
    initial_report = BasinQualityReport(
        valid_period={},
        processing_steps=[],
        imputation_info={},
    )
    ctx0 = CleanContext(df=df, config=config, report=initial_report)

    # chain our steps by unwrapping only on Success
    result: Result[CleanContext, str] = Success(ctx0)
    for step_fn in [
        init_report,
        check_required_columns,
        trim_leading_trailing_nans,
        impute_short_gaps_step,
        compute_valid_periods_step,
        filter_valid_period_step,
        validate_training_data_step,
    ]:
        result = step_fn(result)
        if isinstance(result, Failure):
            break

    # if all succeeded, project to (df, report)
    return result.map(lambda ctx: (ctx.df, ctx.report))


def save_quality_report_to_json(
    report: BasinQualityReport,
    path: Union[str, Path],
) -> tuple[bool, Optional[Path], Optional[str]]:
    """
    Save a BasinQualityReport to a JSON file.

    Args:
        report: The BasinQualityReport dataclass instance.
        path:  File path (or string) where the JSON will be written.

    Returns:
        (success flag, path if successful, error message if any)
    """
    try:
        output_path = Path(path)
        # ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # dump report as dict
        with open(output_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        return True, output_path, None
    except Exception as e:
        return False, None, f"Failed to save quality report: {e}"
