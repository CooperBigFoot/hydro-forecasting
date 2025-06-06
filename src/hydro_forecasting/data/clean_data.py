import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from returns.result import Failure, Result, Success


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
) -> Result[tuple[pl.DataFrame, dict[str, BasinQualityReport]], str]:
    """
    Clean multiple basins in one LazyFrame, using window functions over group_identifier,
    and validate that each basin has sufficient data (min_train_years) for training.

    Only forward fill is used for imputation to avoid potential data leakage.

    Enhanced to ensure that basins marked as passed_quality_check=True will have
    non-empty train, validation, AND test segments when processed by split_data.

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
        val_prop = config.val_prop
        test_prop = config.test_prop
        min_required_train_days = int(min_train_years * 365.25)

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
        schema_names = lf.collect_schema().names()
        required = set(cols + [gid, "date"])
        missing = required - set(schema_names)
        if missing:
            return Failure(f"Missing columns: {sorted(missing)}")

        # Step 2: Sort by gauge and date
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

        # Step 7: Build reports per basin and check data sufficiency
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
                short_gaps = int(sum((e - s) <= max_gap for s, e in zip(starts, ends, strict=False)))
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
                    "imputed_short_gaps_forward_only",  # Updated processing step description
                ],
                imputation_info=info,
            )

            # ORIGINAL DATA SUFFICIENCY CHECK (BASELINE)
            if valid_starts and valid_ends:
                overall_start = max(valid_starts)
                overall_end = min(valid_ends)
                total_days = (overall_end - overall_start).days + 1
                train_days = int(total_days * train_prop)
                if train_days < min_required_train_days:
                    report.passed_quality_check = False
                    available_years = train_days / 365.25
                    report.failure_reason = f"Insufficient training data ({available_years:.2f} years available). \
                         Minimum required training years: {min_train_years}"
                    reports[basin_id] = report
                    continue  # Skip additional checks if baseline fails
                else:
                    # Original check passes, proceed to the new target-specific checks
                    report.processing_steps.append("data_sufficiency_check_passed")
            else:
                report.passed_quality_check = False
                report.failure_reason = "No valid data period found"
                reports[basin_id] = report
                continue  # Skip additional checks if no valid period

            # NEW CHECK 1: TARGET-SPECIFIC DATA AVAILABILITY
            # This simulates the filtering that happens in split_data
            if target_col_name in group.columns:
                target_valid_df = group.filter(pl.col(target_col_name).is_not_null())
                n_valid_target = target_valid_df.height

                if n_valid_target == 0:
                    report.passed_quality_check = False
                    report.failure_reason = (
                        "Initial checks passed, but no non-null target data available for splitting."
                    )
                    reports[basin_id] = report
                    continue  # Skip to next basin if no valid target data

                # NEW CHECK 2: SIMULATE SPLIT CALCULATION & VERIFY NON-EMPTY SEGMENTS
                # This directly mirrors how split_data calculates segment lengths
                min_points_per_segment = 1  # Minimum required points per segment (train/val/test)

                # Calculate expected segment lengths using integer truncation (as in split_data)
                calc_train_end = int(n_valid_target * train_prop)
                calc_val_end = calc_train_end + int(n_valid_target * val_prop)

                # Calculate segment lengths
                calc_train_len = calc_train_end
                calc_val_len = calc_val_end - calc_train_end
                calc_test_len = n_valid_target - calc_val_end

                # Check if all segments meet minimum size requirement
                if calc_train_len < min_points_per_segment:
                    report.passed_quality_check = False
                    report.failure_reason = f"Train segment would be empty or too small ({calc_train_len} points)"
                    reports[basin_id] = report
                    continue

                if calc_val_len < min_points_per_segment:
                    report.passed_quality_check = False
                    report.failure_reason = f"Validation segment would be empty or too small ({calc_val_len} points)"
                    reports[basin_id] = report
                    continue

                if calc_test_len < min_points_per_segment:
                    report.passed_quality_check = False
                    report.failure_reason = f"Test segment would be empty or too small ({calc_test_len} points)"
                    reports[basin_id] = report
                    continue

                # If we reach here, all checks have passed including the stricter target-based checks
                report.processing_steps.append("data_sufficiency_check_passed_strict_target_splits")
                valid_basin_ids.append(basin_id)
            else:
                report.passed_quality_check = False
                report.failure_reason = f"Target column '{target_col_name}' not found in basin data"

            reports[basin_id] = report

        # Step 8: Filter DataFrame to only valid basins
        filtered_df = df.filter(pl.col(gid).is_in(valid_basin_ids)) if valid_basin_ids else pl.DataFrame()

        # Step 9: Drop helper columns
        helper_cols = [f"_before_null_{c}" for c in cols]
        existing_helpers = [h for h in helper_cols if h in filtered_df.columns]
        if existing_helpers:
            filtered_df = filtered_df.drop(existing_helpers)

        print(f"INFO: Processed {len(reports)} basins, {len(valid_basin_ids)} passed quality checks")

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


def summarize_quality_reports_from_folder(
    folder_path: str | Path,
    save_path: str | Path,
) -> Result[SummaryQualityReport, str]:
    """
    Summarize BasinQualityReports stored as JSON files in a folder, focusing on basin counts.

    Args:
        folder_path: Directory containing {basin_id}.json files.
        save_path: Path to save the summary JSON.

    Returns:
        Result containing SummaryQualityReport on success, or error message on failure.
    """
    try:
        folder = Path(folder_path)
        json_files = sorted([f for f in folder.glob("*.json") if f.is_file()])
        if not json_files:
            return Failure(f"No JSON files found in {folder}")

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
                return Failure(f"Error decoding JSON file {file_path}: {e}")
            except Exception as e:
                return Failure(f"Error reading file {file_path}: {e}")

        if not all_reports_data:
            return Failure("No valid report data could be loaded.")

        # Convert list of dicts to list for easier processing
        reports = all_reports_data  # Already a list of dicts

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
        summary.save(save_path)
        return Success(summary)
    except Exception as e:
        import traceback

        # Log the full traceback for better debugging
        print(f"ERROR in summarize_quality_reports_from_folder: {e}\n{traceback.format_exc()}")
        return Failure(f"summarize_quality_reports_from_folder failed: {e}")
