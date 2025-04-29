from __future__ import annotations

import polars as pl
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Optional
from returns.result import Result, Success, Failure

type BasinId = str


def find_gaps_bool(missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find start and end indices of gaps (True values) in a boolean mask.

    Args:
        missing: Boolean numpy array where True indicates missing data.

    Returns:
        Tuple of numpy arrays (starts, ends) with gap boundaries.
    """
    nan_idxs = np.where(missing)[0]
    if nan_idxs.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    boundaries = np.where(np.diff(nan_idxs) > 1)[0]
    starts = np.insert(nan_idxs[boundaries + 1], 0, nan_idxs[0])
    ends = np.append(nan_idxs[boundaries] + 1, nan_idxs[-1] + 1)
    return starts, ends


@dataclass
class BasinQualityReport:
    valid_period: dict[str, dict[str, str | None]]
    processing_steps: list[str]
    imputation_info: dict[str, dict[str, int]]
    passed_quality_check: bool = True
    failure_reason: Optional[str] = None


def clean_data(
    lf: pl.LazyFrame,
    config,
) -> Result[tuple[pl.DataFrame, dict[str, BasinQualityReport]], str]:
    """
    Clean multiple basins in one LazyFrame, using window functions over group_identifier,
    and validate that each basin has sufficient data (min_train_years) for training.

    Args:
        lf: Input LazyFrame containing hydrological data.
        config: Configuration object with required_columns, max_imputation_gap_size,
               group_identifier, min_train_years, and train_prop.

    Returns:
        Success((cleaned_df, reports)) on success, or Failure(message) on error.
    """
    try:
        cols = config.required_columns
        max_gap = config.max_imputation_gap_size
        gid = config.group_identifier
        min_train_years = config.min_train_years
        train_prop = config.train_prop
        min_required_train_days = int(min_train_years * 365.25)

        # 1. Check required columns
        schema_names = lf.collect_schema().names()
        required = set(cols + [gid, "date"])
        missing = required - set(schema_names)
        if missing:
            return Failure(f"Missing columns: {sorted(missing)}")

        # 2. Sort by gauge and date
        lf = lf.sort([gid, "date"])

        # 3. Trim leading/trailing nulls per basin
        fwd = [
            pl.col(c).is_not_null().cum_sum().over(gid).alias(f"_fwd_{c}") for c in cols
        ]
        bwd = [
            pl.col(c).is_not_null().reverse().cum_sum().over(gid).alias(f"_bwd_{c}")
            for c in cols
        ]
        lf = lf.with_columns(fwd + bwd)
        fwd_mask = pl.all_horizontal([pl.col(f"_fwd_{c}") > 0 for c in cols])
        bwd_mask = pl.all_horizontal([pl.col(f"_bwd_{c}") > 0 for c in cols])
        lf = lf.filter(fwd_mask & bwd_mask).drop(
            [f"_fwd_{c}" for c in cols] + [f"_bwd_{c}" for c in cols]
        )

        # 4. Mark original nulls per column
        for c in cols:
            lf = lf.with_columns(pl.col(c).is_null().alias(f"_before_null_{c}"))

        # 5. Impute short gaps via ffill/bfill
        for c in cols:
            lf = lf.with_columns(
                pl.col(c).forward_fill(limit=max_gap).over(gid).alias(c)
            ).with_columns(pl.col(c).backward_fill(limit=max_gap).over(gid).alias(c))

        # 6. Collect to eager DataFrame
        df = lf.collect()

        # 7. Build reports per basin and check data sufficiency
        reports: dict[str, BasinQualityReport] = {}
        valid_basin_ids: list[str] = []

        # Use unique() to get basin IDs as plain strings
        for basin_id in df[gid].unique().to_list():
            # Filter for this specific basin
            group = df.filter(pl.col(gid) == basin_id)

            # Imputation info
            info: dict[str, dict[str, int]] = {}
            for c in cols:
                before_na = int(group[f"_before_null_{c}"].sum())
                after_na = int(group[c].is_null().sum())
                starts, ends = find_gaps_bool(group[f"_before_null_{c}"].to_numpy())
                short_gaps = int(sum((e - s) <= max_gap for s, e in zip(starts, ends)))
                info[c] = {
                    "short_gaps_count": short_gaps,
                    "imputed_values_count": before_na - after_na,
                }

            # Valid period info
            valid_period: dict[str, dict[str, str | None]] = {}
            valid_starts: list = []
            valid_ends: list = []

            for c in cols:
                nonnull = group.filter(pl.col(c).is_not_null())["date"]
                if nonnull.is_empty():
                    valid_period[c] = {"start": None, "end": None}
                else:
                    start_date = nonnull.min()
                    end_date = nonnull.max()
                    valid_period[c] = {
                        "start": start_date.strftime("%Y-%m-%d"),
                        "end": end_date.strftime("%Y-%m-%d"),
                    }
                    valid_starts.append(start_date)
                    valid_ends.append(end_date)

            # Assemble report
            report = BasinQualityReport(
                valid_period=valid_period,
                processing_steps=[
                    "sorted_by_gauge_and_date",
                    "trimmed_nulls",
                    "imputed_short_gaps",
                ],
                imputation_info=info,
            )

            # Data sufficiency check
            if valid_starts and valid_ends:
                overall_start = max(valid_starts)
                overall_end = min(valid_ends)
                total_days = (overall_end - overall_start).days + 1
                train_days = int(total_days * train_prop)
                if train_days < min_required_train_days:
                    report.passed_quality_check = False
                    available_years = train_days / 365.25
                    report.failure_reason = (
                        f"Insufficient training data ({available_years:.2f} years available). \
                         Minimum required training years: {min_train_years}"
                    )
                else:
                    report.processing_steps.append("data_sufficiency_check_passed")
                    valid_basin_ids.append(basin_id)
            else:
                report.passed_quality_check = False
                report.failure_reason = "No valid data period found"

            reports[basin_id] = report

        # 8. Filter DataFrame to only valid basins
        if valid_basin_ids:
            filtered_df = df.filter(pl.col(gid).is_in(valid_basin_ids))
        else:
            filtered_df = pl.DataFrame()

        # 9. Drop helper columns
        helper_cols = [f"_before_null_{c}" for c in cols]
        existing_helpers = [h for h in helper_cols if h in filtered_df.columns]
        if existing_helpers:
            filtered_df = filtered_df.drop(existing_helpers)

        print(
            f"INFO: Processed {len(reports)} basins, {len(valid_basin_ids)} passed quality checks"
        )

        return Success((filtered_df, reports))

    except Exception as e:
        return Failure(f"clean_data failed: {e}")


def save_quality_report_to_json(
    report: BasinQualityReport,
    path: str | Path,
) -> tuple[bool, Path | None, str | None]:
    """
    Save a BasinQualityReport to a JSON file.

    Args:
        report: BasinQualityReport instance to save.
        path: Output file path (str or Path).

    Returns:
        Tuple of (success flag, output Path if successful, error message if failed).
    """
    try:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        return True, output_path, None
    except Exception as e:
        return False, None, f"Failed to save quality report: {e}"
