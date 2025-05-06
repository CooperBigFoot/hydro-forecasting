# filename: src/hydro_forecasting/data/preprocessing.py

# (Keep existing imports)
from pathlib import Path
from dataclasses import dataclass
import polars as pl
import pandas as pd
from typing import Optional, Any, Union, Iterator
from returns.result import Failure, Success, Result
from sklearn.pipeline import Pipeline, clone
from .clean_data import (
    BasinQualityReport,
    SummaryQualityReport,
    clean_data,
    save_quality_report_to_json,
    summarize_quality_reports_from_folder,
)
from ..preprocessing.grouped import GroupedPipeline
from ..preprocessing.time_series_preprocessing import (
    fit_time_series_pipelines,
    transform_time_series_data,
    save_time_series_pipelines,
)
from ..preprocessing.static_preprocessing import (
    process_static_data,
    save_static_pipeline,
)
from .config_utils import save_config

import logging  # Added logging

logger = logging.getLogger(__name__)  # Added logger


# --- Keep Existing Dataclasses (ProcessingConfig, ProcessingOutput) ---
@dataclass
class ProcessingConfig:
    required_columns: list[str]
    preprocessing_config: Optional[dict[str, dict[str, Any]]] = None
    min_train_years: float = 5.0
    max_imputation_gap_size: int = 5
    group_identifier: str = "gauge_id"
    train_prop: float = 0.6
    val_prop: float = 0.2
    test_prop: float = 0.2


@dataclass
class ProcessingOutput:
    """Output paths and artifacts from a hydro data processing run."""

    run_output_dir: Path
    processed_timeseries_dir: Path
    processed_static_attributes_path: Optional[Path]
    fitted_time_series_pipelines_path: Path
    fitted_static_pipeline_path: Optional[Path]
    quality_reports_dir: Path
    summary_quality_report_path: Path
    config_path: Path
    success_marker_path: Path
    summary_quality_report: SummaryQualityReport


# --- Keep Existing Helper Functions ---
# (split_data, batch_basins, _load_single_basin_lazy, load_basins_timeseries_lazy,
# write_train_val_test_splits_to_disk, batch_process_time_series_data)
# Make sure they use Polars and handle Results appropriately as in the provided source.


def split_data(
    df: pl.DataFrame, config: ProcessingConfig
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split data into training, validation, and test sets based on proportions.
    Filters out nulls first to ensure splits contain only valid data points.

    Args:
        df: Polars DataFrame with data
        config: Configuration object

    Returns:
        Tuple of (train_df, val_df, test_df), each a Polars DataFrame
    """
    train_list: list[pl.DataFrame] = []
    val_list: list[pl.DataFrame] = []
    test_list: list[pl.DataFrame] = []

    # Ensure preprocessing_config and target keys exist, provide default if needed
    target_cfg = (
        config.preprocessing_config.get("target", {})
        if config.preprocessing_config
        else {}
    )
    target_col = target_cfg.get("column", "streamflow")  # Default target
    group_id = config.group_identifier

    # iterate over each basin
    unique_basins = []
    if group_id in df.columns:
        unique_basins = df.select(pl.col(group_id)).unique().to_series().to_list()
    else:
        logger.warning(
            f"Group identifier '{group_id}' not found in DataFrame columns during split."
        )
        # Decide how to handle this: error, or assume single group?
        # Assuming single group for now if group_id is missing
        unique_basins = ["single_group"]  # Placeholder
        df = df.with_columns(pl.lit("single_group").alias(group_id))

    for basin_id in unique_basins:
        basin_df = df.filter(pl.col(group_id) == basin_id).sort("date")
        valid_df = basin_df.filter(pl.col(target_col).is_not_null())
        n_valid = valid_df.height

        if n_valid == 0:
            logger.warning(
                f"Basin {basin_id} has no valid points for target '{target_col}', skipping split."
            )
            continue

        # compute split sizes
        train_end = int(n_valid * config.train_prop)
        val_end = train_end + int(n_valid * config.val_prop)

        # Ensure split indices are valid
        train_end = min(train_end, n_valid)
        val_end = min(val_end, n_valid)

        train_seg = valid_df.slice(0, train_end)
        val_seg = valid_df.slice(train_end, val_end - train_end)
        test_seg = valid_df.slice(val_end, n_valid - val_end)  # Use remaining

        train_list.append(train_seg)
        val_list.append(val_seg)
        test_list.append(test_seg)

    # concatenate or return empty DataFrames
    train_df = pl.concat(train_list, how="vertical") if train_list else pl.DataFrame()
    val_df = pl.concat(val_list, how="vertical") if val_list else pl.DataFrame()
    test_df = pl.concat(test_list, how="vertical") if test_list else pl.DataFrame()

    return train_df, val_df, test_df


def batch_basins(
    basins: list[str],
    batch_size: int,
) -> Iterator[list[str]]:
    """
    Yield successive batches of basin identifiers from the input sequence.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    total = len(basins)
    for start in range(0, total, batch_size):
        yield list(basins[start : start + batch_size])


def _load_single_basin_lazy(
    gauge_id: str,
    region_time_series_base_dirs: dict[str, Path],
    required_columns: list[str],
    group_identifier: str,
) -> Result[pl.LazyFrame, str]:
    """Helper: lazily load one basin, wrapped in a Result."""
    try:
        prefix = gauge_id.split("_", 1)[0]
        base_dir = region_time_series_base_dirs.get(prefix)
        if base_dir is None:
            return Failure(f"No base directory for region prefix '{prefix}'")

        file_path = Path(base_dir) / f"{gauge_id}.parquet"
        if not file_path.exists():
            return Failure(f"Timeseries file not found: {file_path}")

        lf = pl.scan_parquet(str(file_path))

        schema = lf.collect_schema().names()
        missing = [c for c in required_columns if c not in schema]
        if "date" not in schema:
            missing.append("date")
        if missing:
            return Failure(f"Missing required columns in basin {gauge_id}: {missing}")

        # Add group identifier and select/sort
        lf = lf.with_columns(pl.lit(gauge_id).alias(group_identifier))
        # Select only necessary columns including the new group identifier
        select_cols = [group_identifier, "date"] + required_columns
        select_cols = list(dict.fromkeys(select_cols))  # Remove duplicates
        lf = lf.select(select_cols).sort("date")

        return Success(lf)

    except Exception as exc:
        return Failure(f"Error loading basin {gauge_id}: {exc}")


def load_basins_timeseries_lazy(
    gauge_ids: list[str],
    region_time_series_base_dirs: dict[str, Path],
    required_columns: list[str],
    group_identifier: str = "gauge_id",
) -> Result[pl.LazyFrame, str]:
    """Lazily load & stack time-series for multiple basins."""
    if not gauge_ids:
        return Failure("No gauge IDs provided for loading")

    lazy_frames: list[pl.LazyFrame] = []
    for gid in gauge_ids:
        result = _load_single_basin_lazy(
            gid, region_time_series_base_dirs, required_columns, group_identifier
        )
        if isinstance(result, Failure):
            return result  # Short-circuit on first error
        lazy_frames.append(result.unwrap())

    if not lazy_frames:
        return Failure("No basins were successfully loaded.")

    # concatenate and final sort
    combined = pl.concat(lazy_frames, how="vertical").sort([group_identifier, "date"])
    return Success(combined)


def write_train_val_test_splits_to_disk(
    train_df: pl.DataFrame,  # Changed to eager DataFrame
    val_df: pl.DataFrame,  # Changed to eager DataFrame
    test_df: pl.DataFrame,  # Changed to eager DataFrame
    output_dir: Union[str, Path],
    group_identifier: str = "gauge_id",
    compression: str = "zstd",
    basin_ids: Optional[list[str]] = None,
) -> Result[None, str]:
    """
    Write train, validation, and test splits to disk as individual Parquet files per group.
    """
    base_path = Path(output_dir)

    # Infer basin IDs from the training data if not provided
    if not basin_ids:
        if train_df.height > 0 and group_identifier in train_df.columns:
            basin_ids = train_df[group_identifier].unique().to_list()
        else:
            # Handle case where train_df might be empty or missing the identifier
            # Maybe try inferring from val or test, or return Failure
            if val_df.height > 0 and group_identifier in val_df.columns:
                basin_ids = val_df[group_identifier].unique().to_list()
            elif test_df.height > 0 and group_identifier in test_df.columns:
                basin_ids = test_df[group_identifier].unique().to_list()
            else:
                logger.warning("Cannot infer basin IDs, no data found in splits.")
                basin_ids = []  # Or return Failure("Could not infer basin IDs")?

    if not basin_ids:
        logger.warning("No basin IDs to process for writing splits.")
        return Success(None)  # Nothing to write

    try:
        for split_name, df in (
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ):
            split_path = base_path / split_name
            split_path.mkdir(parents=True, exist_ok=True)

            # Iterate through the unique basin IDs determined earlier
            for gid in basin_ids:
                # Filter the current split's DataFrame for the specific basin
                basin_split_df = df.filter(pl.col(group_identifier) == gid)

                # Only write if the filtered DataFrame is not empty
                if basin_split_df.height > 0:
                    out_file = split_path / f"{gid}.parquet"
                    basin_split_df.write_parquet(out_file, compression=compression)

        return Success(None)
    except Exception as exc:
        return Failure(f"Error writing splits to disk: {exc}")


def batch_process_time_series_data(
    lf: pl.LazyFrame,
    config: ProcessingConfig,
    features_pipeline: GroupedPipeline,
    target_pipeline: GroupedPipeline,
) -> Result[
    tuple[
        pl.DataFrame,  # Changed to eager DataFrame
        pl.DataFrame,  # Changed to eager DataFrame
        pl.DataFrame,  # Changed to eager DataFrame
        dict[str, GroupedPipeline],
        dict[str, BasinQualityReport],
    ],
    str,  # Added error type to Result
]:
    """
    Clean, split, fit on train, and transform time-series data.

    Args:
        lf: Polars LazyFrame containing the time-series data for a batch of basins.
        config: Configuration object with processing parameters.
        features_pipeline: GroupedPipeline for feature transformation (passed for fitting).
        target_pipeline: GroupedPipeline for target transformation (passed for fitting).

    Returns:
        Success containing a tuple:
            - train_df: Eager Polars DataFrame for the training split.
            - val_df: Eager Polars DataFrame for the validation split.
            - test_df: Eager Polars DataFrame for the test split.
            - fitted_pipelines: Dictionary of fitted GroupedPipelines for this batch.
            - quality_reports: Dictionary of BasinQualityReport objects for basins in this batch.
        Failure containing an error message string.
    """
    # 1. Clean Data
    clean_result = clean_data(lf, config)
    if isinstance(clean_result, Failure):
        return Failure(f"Data cleaning failed: {clean_result.failure()}")
    cleaned_df, quality_reports = clean_result.unwrap()

    # 2. Identify valid basins after cleaning
    valid_basins = [
        basin_id
        for basin_id, report in quality_reports.items()
        if report.passed_quality_check
    ]
    if not valid_basins:
        return Failure("No valid basins found after quality checks in this batch.")

    # Filter cleaned_df to include only valid basins before splitting
    cleaned_df_valid_basins = cleaned_df.filter(
        pl.col(config.group_identifier).is_in(valid_basins)
    )

    if cleaned_df_valid_basins.height == 0:
        return Failure("Cleaned DataFrame is empty after filtering for valid basins.")

    # 3. Split Data (using eager DataFrame)
    train_df, val_df, test_df = split_data(cleaned_df_valid_basins, config)

    # 4. Fit Pipelines on Training Data (convert to pandas for scikit-learn compatibility)
    if train_df.height == 0:
        logger.warning("Training split is empty for this batch. Cannot fit pipelines.")
        # Return empty dataframes and indicate no pipelines were fitted for this batch
        return Success(
            (pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), {}, quality_reports)
        )

    train_pd_df = train_df.to_pandas()

    # Clone pipelines before fitting to avoid modifying the originals shared across batches
    batch_features_pipeline = clone(features_pipeline)
    batch_target_pipeline = clone(target_pipeline)

    fit_result = fit_time_series_pipelines(
        train_pd_df,
        batch_features_pipeline,  # Use cloned version
        batch_target_pipeline,  # Use cloned version
    )
    if isinstance(fit_result, Failure):
        return Failure(f"Pipeline fitting failed: {fit_result.failure()}")
    fitted_batch_pipelines = (
        fit_result.unwrap()
    )  # Pipelines fitted *only* on this batch's train data

    # 5. Transform All Splits (using pandas for compatibility)
    # Convert other splits to pandas
    val_pd_df = val_df.to_pandas() if val_df.height > 0 else pd.DataFrame()
    test_pd_df = test_df.to_pandas() if test_df.height > 0 else pd.DataFrame()

    # Transform train set
    train_transformed_result = transform_time_series_data(
        train_pd_df, fitted_batch_pipelines
    )
    if isinstance(train_transformed_result, Failure):
        return Failure(f"Train transform failed: {train_transformed_result.failure()}")
    train_transformed_pd = train_transformed_result.unwrap()

    # Transform validation set
    val_transformed_pd = pd.DataFrame()
    if not val_pd_df.empty:
        val_transformed_result = transform_time_series_data(
            val_pd_df, fitted_batch_pipelines
        )
        if isinstance(val_transformed_result, Failure):
            return Failure(
                f"Validation transform failed: {val_transformed_result.failure()}"
            )
        val_transformed_pd = val_transformed_result.unwrap()

    # Transform test set
    test_transformed_pd = pd.DataFrame()
    if not test_pd_df.empty:
        test_transformed_result = transform_time_series_data(
            test_pd_df, fitted_batch_pipelines
        )
        if isinstance(test_transformed_result, Failure):
            return Failure(
                f"Test transform failed: {test_transformed_result.failure()}"
            )
        test_transformed_pd = test_transformed_result.unwrap()

    # 6. Convert back to Polars DataFrames (eager)
    final_train_df = (
        pl.from_pandas(train_transformed_pd)
        if not train_transformed_pd.empty
        else pl.DataFrame()
    )
    final_val_df = (
        pl.from_pandas(val_transformed_pd)
        if not val_transformed_pd.empty
        else pl.DataFrame()
    )
    final_test_df = (
        pl.from_pandas(test_transformed_pd)
        if not test_transformed_pd.empty
        else pl.DataFrame()
    )

    return Success(
        (
            final_train_df,
            final_val_df,
            final_test_df,
            fitted_batch_pipelines,
            quality_reports,
        )
    )


# --- Main Processor ---
def run_hydro_processor(
    region_time_series_base_dirs: dict[str, Path],
    region_static_attributes_base_dirs: dict[str, Path],
    path_to_preprocessing_output_directory: Union[str, Path],
    required_columns: list[str],
    run_uuid: str,
    datamodule_config: dict[str, Any],
    preprocessing_config: dict[str, dict[str, GroupedPipeline | Pipeline]],
    min_train_years: float = 5.0,
    max_imputation_gap_size: int = 5,
    group_identifier: str = "gauge_id",
    train_prop: float = 0.5,
    val_prop: float = 0.25,
    test_prop: float = 0.25,
    list_of_gauge_ids_to_process: Optional[list[str]] = None,
    basin_batch_size: int = 50,
) -> Result[ProcessingOutput, str]:
    """
    Main function to run the hydrological data processor with preprocessing pipelines.

    Processes both static attributes and time series data for multiple basins:
    1. Creates necessary directories for the processing run.
    2. Processes static attributes if available and configured.
    3. Batch-processes time series data: loads, cleans, splits, fits pipelines on train, transforms all splits.
    4. Saves processed data splits (train/val/test) per basin.
    5. Saves quality reports per basin and a summary report.
    6. Saves fitted pipelines (static and time series).
    7. Creates a success marker file if everything succeeds.

    **Note:** This version removes the index entry creation step.
    """
    try:
        # Setup processing configuration
        config = ProcessingConfig(
            required_columns=required_columns,
            preprocessing_config=preprocessing_config,
            min_train_years=min_train_years,
            max_imputation_gap_size=max_imputation_gap_size,
            group_identifier=group_identifier,
            train_prop=train_prop,
            val_prop=val_prop,
            test_prop=test_prop,
        )

        # Prepare output directories
        run_output_dir = Path(path_to_preprocessing_output_directory) / run_uuid
        run_output_dir.mkdir(parents=True, exist_ok=True)

        ts_output_dir = run_output_dir / "processed_time_series"
        ts_output_dir.mkdir(
            parents=True, exist_ok=True
        )  # Will contain train/val/test subdirs

        quality_reports_dir = run_output_dir / "quality_reports"
        quality_reports_dir.mkdir(parents=True, exist_ok=True)

        config_path = run_output_dir / "config.json"
        summary_quality_report_path = run_output_dir / "quality_summary.json"
        success_marker_path = run_output_dir / "_SUCCESS"

        # Save DataModule config early
        save_result = save_config(datamodule_config, config_path)
        if isinstance(save_result, Failure):
            return Failure(f"Failed to save config: {save_result.failure()}")
        logger.info(f"Config saved to {config_path}")

        # --- 1. Process Static Data ---
        static_features_path = run_output_dir / "processed_static_features.parquet"
        fitted_static_path = run_output_dir / "fitted_static_pipeline.joblib"
        static_pipeline_fitted: Optional[Pipeline] = (
            None  # Store the fitted static pipeline
        )

        if "static_features" in preprocessing_config and list_of_gauge_ids_to_process:
            logger.info("Processing static features...")
            static_processing_results = process_static_data(
                region_static_attributes_base_dirs,
                list_of_gauge_ids_to_process,
                preprocessing_config,
                static_features_path,
                group_identifier,
            )

            if isinstance(static_processing_results, Success):
                save_path_static, static_pipeline_fitted = (
                    static_processing_results.unwrap()
                )
                save_results = save_static_pipeline(
                    static_pipeline_fitted, fitted_static_path
                )
                if isinstance(save_results, Failure):
                    return Failure(
                        f"Failed to save static pipeline: {save_results.failure()}"
                    )
                logger.info(f"Static features saved to {save_path_static}")
                logger.info(f"Static features pipeline saved to {fitted_static_path}")
            else:
                logger.warning(
                    f"Static data processing failed: {static_processing_results.failure()}"
                )
                static_features_path = None  # Mark as unavailable
                fitted_static_path = None
        else:
            logger.info(
                "Skipping static feature processing (not configured or no gauge IDs)."
            )
            static_features_path = None
            fitted_static_path = None

        # --- 2. Batch Process Time Series Data ---
        fitted_ts_pipelines_path = (
            run_output_dir / "fitted_time_series_pipelines.joblib"
        )

        # Initialize main pipelines (cloned from config) to accumulate fitted groups
        main_pipelines: dict[str, GroupedPipeline] = {}
        if (
            "features" in preprocessing_config
            and "pipeline" in preprocessing_config["features"]
        ):
            main_pipelines["features"] = clone(
                preprocessing_config["features"]["pipeline"]
            )
            main_pipelines[
                "features"
            ].fitted_pipelines.clear()  # Ensure it's empty initially
        if (
            "target" in preprocessing_config
            and "pipeline" in preprocessing_config["target"]
        ):
            main_pipelines["target"] = clone(preprocessing_config["target"]["pipeline"])
            main_pipelines[
                "target"
            ].fitted_pipelines.clear()  # Ensure it's empty initially

        if not main_pipelines:
            return Failure(
                "No time series pipelines ('features' or 'target') found in preprocessing_config."
            )

        all_quality_reports: dict[str, BasinQualityReport] = {}
        processed_basin_count = 0

        if not list_of_gauge_ids_to_process:
            return Failure("No gauge IDs provided for time series processing.")

        logger.info(
            f"Starting time series processing for {len(list_of_gauge_ids_to_process)} basins in batches of {basin_batch_size}..."
        )
        for batch_num, batch_ids in enumerate(
            batch_basins(list_of_gauge_ids_to_process, basin_batch_size)
        ):
            logger.info(
                f"--- Processing Batch {batch_num + 1} ({len(batch_ids)} basins) ---"
            )

            # Load data for the batch
            loading_results = load_basins_timeseries_lazy(
                batch_ids,
                region_time_series_base_dirs,
                required_columns,
                group_identifier,
            )
            if isinstance(loading_results, Failure):
                logger.error(
                    f"Failed to load batch {batch_num + 1}: {loading_results.failure()}"
                )
                continue  # Skip to next batch

            lf = loading_results.unwrap()

            # Process the batch: clean, split, fit, transform
            batch_result = batch_process_time_series_data(
                lf,
                config=config,
                features_pipeline=main_pipelines[
                    "features"
                ],  # Pass the main pipeline instance
                target_pipeline=main_pipelines[
                    "target"
                ],  # Pass the main pipeline instance
            )

            if isinstance(batch_result, Failure):
                logger.error(
                    f"Failed processing batch {batch_num + 1}: {batch_result.failure()}"
                )
                continue  # Skip to next batch

            # Unpack successful batch results
            train_df, val_df, test_df, batch_fitted_pipelines, quality_reports = (
                batch_result.unwrap()
            )

            # Save quality reports for this batch
            for gauge_id, report in quality_reports.items():
                report_name = f"{gauge_id}.json"
                save_path = quality_reports_dir / report_name
                succ_flag, _, error = save_quality_report_to_json(
                    report=report, path=save_path
                )
                if not succ_flag:
                    logger.warning(
                        f"Failed to save quality report for {gauge_id}: {error}"
                    )
                all_quality_reports[gauge_id] = report  # Collect for summary

            # Merge fitted pipelines from this batch into the main pipelines
            # Only add if the pipeline was successfully fitted for this batch
            if batch_fitted_pipelines:
                for pipeline_type in ["features", "target"]:
                    if pipeline_type in batch_fitted_pipelines:
                        batch_pipe = batch_fitted_pipelines[pipeline_type]
                        if isinstance(batch_pipe, GroupedPipeline):
                            for (
                                group_id,
                                group_pipeline,
                            ) in batch_pipe.fitted_pipelines.items():
                                main_pipelines[pipeline_type].add_fitted_group(
                                    group_id, group_pipeline
                                )
                        else:
                            logger.warning(
                                f"Pipeline type '{pipeline_type}' in batch results was not a GroupedPipeline."
                            )

            # Write processed train/val/test splits to disk for this batch
            write_results = write_train_val_test_splits_to_disk(
                train_df,
                val_df,
                test_df,
                output_dir=ts_output_dir,  # Base directory for splits
                group_identifier=group_identifier,
                basin_ids=batch_ids,  # Write only for basins in this batch
            )
            if isinstance(write_results, Failure):
                # Log error but continue to next batch if possible
                logger.error(
                    f"Failed to write processed data for batch {batch_num + 1}: {write_results.failure()}"
                )
                continue

            processed_basin_count += len(
                batch_ids
            )  # Count basins attempted in the batch
            logger.info(f"--- Finished Batch {batch_num + 1} ---")

        logger.info(
            f"Finished processing all time series batches. Attempted {processed_basin_count} basins."
        )

        # 3. Save the *accumulated* fitted time series pipelines
        if main_pipelines:
            save_results = save_time_series_pipelines(
                main_pipelines, fitted_ts_pipelines_path
            )
            if isinstance(save_results, Failure):
                return Failure(
                    f"Failed to save final time series pipelines: {save_results.failure()}"
                )
            logger.info(
                f"Fitted time series pipelines saved to {fitted_ts_pipelines_path}"
            )
        else:
            # Handle case where no time series pipelines were defined/fitted
            logger.warning("No main time series pipelines were fitted or saved.")
            fitted_ts_pipelines_path = None  # Mark as unavailable

        # 4. Create summary quality report from all collected reports
        summary_result = summarize_quality_reports_from_folder(
            quality_reports_dir, summary_quality_report_path
        )
        if isinstance(summary_result, Failure):
            return Failure(
                f"Failed to create summary report: {summary_result.failure()}"
            )
        summary_report = summary_result.unwrap()
        logger.info(f"Summary quality report saved to {summary_quality_report_path}")

        # 5. Create success marker
        success_marker_path.touch(exist_ok=True)
        logger.info(
            f"SUCCESS: Preprocessing completed successfully. Output at {run_output_dir}"
        )

        # 6. Return processing output
        output = ProcessingOutput(
            run_output_dir=run_output_dir,
            processed_timeseries_dir=ts_output_dir,
            processed_static_attributes_path=static_features_path,
            fitted_time_series_pipelines_path=fitted_ts_pipelines_path,
            fitted_static_pipeline_path=fitted_static_path,
            quality_reports_dir=quality_reports_dir,
            summary_quality_report_path=summary_quality_report_path,
            config_path=config_path,
            success_marker_path=success_marker_path,
            summary_quality_report=summary_report,
        )

        return Success(output)

    except Exception as e:
        import traceback

        logger.error(f"ERROR in run_hydro_processor: {e}\n{traceback.format_exc()}")
        return Failure(f"Unexpected error during hydro processing: {e}")
