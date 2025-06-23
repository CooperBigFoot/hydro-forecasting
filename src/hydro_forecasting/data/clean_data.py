import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..exceptions import DataProcessingError, DataQualityError, FileOperationError


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
    failure_reason: str | None = None


@dataclass
class SummaryQualityReport:
    original_basins: int
    passed_basins: int
    failed_basins: int
    excluded_basins: dict[str, str]  # basin_id -> failure_reason
    retained_basins: list[str]

    def save(self, path: Path | str) -> Path:
        """
        Save the summary report as JSON to the given path.

        Args:
            path: File path or string where to save the report

        Returns:
            The Path to which the report was saved
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            # Use asdict for dataclass serialization, handle potential non-serializable types if needed
            json.dump(asdict(self), f, indent=2, default=str)
        return output_path


def clean_data(
    lf: pl.LazyFrame,
    config,
    raise_on_failure: bool = True,
) -> tuple[pl.DataFrame, dict[str, BasinQualityReport]]:
    """
    Clean multiple basins in one LazyFrame, using window functions over group_identifier,
    and validate that each basin has sufficient data (min_train_years) for training.

    Only forward fill is used for imputation to avoid potential data leakage.

    Uses target-data-based validation that exactly mirrors split_data() logic to ensure
    consistency between quality checks and actual data splitting.

    Args:
        lf: Input LazyFrame containing hydrological data.
        config: Configuration object with required_columns, max_imputation_gap_size,
               group_identifier, min_train_years, and train_prop.
        raise_on_failure: Whether to raise DataQualityError when basins fail quality checks.
                         If False, returns all results without raising.

    Returns:
        Tuple of (cleaned_df, reports) containing the cleaned DataFrame and quality reports.

    Raises:
        DataQualityError: When basins fail quality checks or data validation.
        DataProcessingError: When data processing operations fail.
    """
    cols = config.required_columns
    max_gap = config.max_imputation_gap_size
    gid = config.group_identifier
    min_train_years = config.min_train_years
    train_prop = config.train_prop
    val_prop = config.val_prop
    test_prop = config.test_prop

    # Convert minimum training years to minimum data points needed
    # Assuming daily data: 1 year ≈ 365.25 days
    min_required_train_points = int(min_train_years * 365.25)

    # Determine target column name - same logic as split_data
    target_col_name = "streamflow"  # Default fallback
    if (
        hasattr(config, "preprocessing_config")
        and config.preprocessing_config
        and "target" in config.preprocessing_config
    ):
        target_cfg = config.preprocessing_config.get("target", {})
        target_col_name = target_cfg.get("column", "streamflow")

    # Step 1: Check required columns
    try:
        schema_names = lf.collect_schema().names()
    except Exception as e:
        raise DataProcessingError(f"Failed to collect schema: {e}") from e

    required = set(cols + [gid, "date"])
    missing = required - set(schema_names)
    if missing:
        raise DataProcessingError(f"Missing columns: {sorted(missing)}")

    # Step 2: Sort by gauge and date
    try:
        lf = lf.sort([gid, "date"])

        # Step 3: Trim leading/trailing nulls per basin
        fwd = [pl.col(c).is_not_null().cum_sum().over(gid).alias(f"_fwd_{c}") for c in cols]
        bwd = [pl.col(c).is_not_null().reverse().cum_sum().over(gid).alias(f"_bwd_{c}") for c in cols]
        lf = lf.with_columns(fwd + bwd)
        fwd_mask = pl.all_horizontal([pl.col(f"_fwd_{c}") > 0 for c in cols])
        bwd_mask = pl.all_horizontal([pl.col(f"_bwd_{c}") > 0 for c in cols])
        lf = lf.filter(fwd_mask & bwd_mask).drop([f"_fwd_{c}" for c in cols] + [f"_bwd_{c}" for c in cols])

        # Step 4: Mark original nulls per column
        for c in cols:
            lf = lf.with_columns(pl.col(c).is_null().alias(f"_before_null_{c}"))

        # Step 5: Impute short gaps via ffill only (removed bfill to avoid data leakage)
        for c in cols:
            lf = lf.with_columns(pl.col(c).forward_fill(limit=max_gap).over(gid).alias(c))

        # Step 6: Collect to eager DataFrame
        df = lf.collect()
    except Exception as e:
        raise DataProcessingError(f"Data processing failed: {e}") from e

    # Step 7: Build reports per basin and check data sufficiency
    reports: dict[str, BasinQualityReport] = {}
    valid_basin_ids: list[str] = []
    failed_basins: list[str] = []

    # Use unique() to get basin IDs as plain strings
    try:
        basin_ids = df[gid].unique().to_list()
    except Exception as e:
        raise DataProcessingError(f"Failed to extract basin IDs: {e}") from e

    for basin_id in basin_ids:
        # Filter for this specific basin
        group = df.filter(pl.col(gid) == basin_id)

        # Imputation info
        info: dict[str, dict[str, int]] = {}
        for c in cols:
            before_na = int(group[f"_before_null_{c}"].sum())
            after_na = int(group[c].is_null().sum())
            starts, ends = find_gaps_bool(group[f"_before_null_{c}"].to_numpy())
            short_gaps = int(sum((e - s) <= max_gap for s, e in zip(starts, ends, strict=False)))
            info[c] = {
                "short_gaps_count": short_gaps,
                "imputed_values_count": before_na - after_na,
            }

        # Valid period info
        valid_period: dict[str, dict[str, str | None]] = {}
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

        # Assemble report
        report = BasinQualityReport(
            valid_period=valid_period,
            processing_steps=[
                "sorted_by_gauge_and_date",
                "trimmed_nulls",
                "imputed_short_gaps_forward_only",
            ],
            imputation_info=info,
        )

        # TARGET-DATA-BASED VALIDATION (mirrors split_data() exactly)
        if target_col_name not in group.columns:
            report.passed_quality_check = False
            report.failure_reason = f"Target column '{target_col_name}' not found in basin data"
            reports[basin_id] = report
            failed_basins.append(basin_id)
            continue

        # Filter to only non-null target data (exactly as split_data does)
        target_valid_df = group.filter(pl.col(target_col_name).is_not_null())
        n_valid_target = target_valid_df.height

        if n_valid_target == 0:
            report.passed_quality_check = False
            report.failure_reason = "No non-null target data available"
            reports[basin_id] = report
            failed_basins.append(basin_id)
            continue

        # Calculate segment lengths using identical logic to split_data
        # Using integer truncation (as in split_data)
        calc_train_end = int(n_valid_target * train_prop)
        calc_val_end = calc_train_end + int(n_valid_target * val_prop)

        # Calculate actual segment lengths
        calc_train_len = calc_train_end
        calc_val_len = calc_val_end - calc_train_end
        calc_test_len = n_valid_target - calc_val_end

        # Minimum points per segment requirement
        min_points_per_segment = 1

        # Check training data sufficiency (convert points back to years for comparison)
        actual_train_years = calc_train_len / 365.25

        if calc_train_len < min_required_train_points:
            report.passed_quality_check = False
            report.failure_reason = (
                f"Insufficient training data: {actual_train_years:.2f} years available "
                f"({calc_train_len} data points). Minimum required: {min_train_years} years "
                f"({min_required_train_points} data points)"
            )
            reports[basin_id] = report
            failed_basins.append(basin_id)
            continue

        # Check all segments meet minimum size requirements
        if calc_val_len < min_points_per_segment:
            report.passed_quality_check = False
            report.failure_reason = f"Validation segment too small: {calc_val_len} data points"
            reports[basin_id] = report
            failed_basins.append(basin_id)
            continue

        if calc_test_len < min_points_per_segment:
            report.passed_quality_check = False
            report.failure_reason = f"Test segment too small: {calc_test_len} data points"
            reports[basin_id] = report
            failed_basins.append(basin_id)
            continue

        # All checks passed
        report.processing_steps.extend(
            [
                "target_data_validation_passed",
                f"training_segment_validated_{actual_train_years:.2f}_years",
            ]
        )
        valid_basin_ids.append(basin_id)
        reports[basin_id] = report

    # Check if any basins failed quality checks and raise exception if needed
    if failed_basins and raise_on_failure:
        failed_reasons = [f"{basin_id}: {reports[basin_id].failure_reason}" for basin_id in failed_basins]
        raise DataQualityError(f"Quality check failed for {len(failed_basins)} basin(s): {'; '.join(failed_reasons)}")

    # Step 8: Filter DataFrame to only valid basins
    try:
        filtered_df = df.filter(pl.col(gid).is_in(valid_basin_ids)) if valid_basin_ids else pl.DataFrame()

        # Step 9: Drop helper columns
        helper_cols = [f"_before_null_{c}" for c in cols]
        existing_helpers = [h for h in helper_cols if h in filtered_df.columns]
        if existing_helpers:
            filtered_df = filtered_df.drop(existing_helpers)
    except Exception as e:
        raise DataProcessingError(f"Failed to filter and clean DataFrame: {e}") from e

    print(f"INFO: Processed {len(reports)} basins, {len(valid_basin_ids)} passed quality checks")

    return (filtered_df, reports)


def save_quality_report_to_json(
    report: BasinQualityReport,
    path: str | Path,
) -> None:
    """
    Save a BasinQualityReport to a JSON file.

    Args:
        report: BasinQualityReport instance to save.
        path: Output file path (str or Path).

    Raises:
        FileOperationError: If saving the report fails.
    """
    try:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
    except Exception as e:
        raise FileOperationError(f"Failed to save quality report to {path}: {e}")


def summarize_quality_reports_from_folder(
    folder_path: str | Path,
    save_path: str | Path,
) -> SummaryQualityReport:
    """
    Summarize BasinQualityReports stored as JSON files in a folder, focusing on basin counts.

    Args:
        folder_path: Directory containing {basin_id}.json files.
        save_path: Path to save the summary JSON.

    Returns:
        SummaryQualityReport containing the summary of quality reports.

    Raises:
        FileOperationError: When file operations fail (reading, writing, or accessing files).
        DataProcessingError: When data processing or JSON parsing fails.
    """
    folder = Path(folder_path)

    try:
        json_files = sorted([f for f in folder.glob("*.json") if f.is_file()])
    except Exception as e:
        raise FileOperationError(f"Failed to access folder {folder}: {e}") from e

    if not json_files:
        raise FileOperationError(f"No JSON files found in {folder}")

    # Read individual JSON files and combine them
    all_reports_data: list[dict[str, Any]] = []
    for file_path in json_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                report_data = json.load(f)
                # Add basin_id from filename if not present in the JSON structure itself
                if "basin_id" not in report_data:
                    report_data["basin_id"] = file_path.stem
                all_reports_data.append(report_data)
        except json.JSONDecodeError as e:
            raise DataProcessingError(f"Error decoding JSON file {file_path}: {e}") from e
        except Exception as e:
            raise FileOperationError(f"Error reading file {file_path}: {e}") from e

    if not all_reports_data:
        raise DataProcessingError("No valid report data could be loaded.")

    # Convert list of dicts to list for easier processing
    reports = all_reports_data  # Already a list of dicts

    try:
        total = len(reports)
        passed_reports = [r for r in reports if r.get("passed_quality_check", True)]
        passed = len(passed_reports)
        failed = total - passed

        excluded = {
            r["basin_id"]: r.get("failure_reason", "Unknown reason")  # Provide default reason
            for r in reports
            if not r.get("passed_quality_check", True)
        }
        passed_ids = [r["basin_id"] for r in reports if r.get("passed_quality_check")]

        # Create the simplified summary report
        summary = SummaryQualityReport(
            original_basins=total,
            passed_basins=passed,
            failed_basins=failed,
            excluded_basins=excluded,
            retained_basins=passed_ids,
        )

        # Save the summary report
        try:
            summary.save(save_path)
        except Exception as e:
            raise FileOperationError(f"Failed to save summary report to {save_path}: {e}") from e

        return summary

    except Exception as e:
        raise DataProcessingError(f"Failed to process quality reports: {e}") from e
