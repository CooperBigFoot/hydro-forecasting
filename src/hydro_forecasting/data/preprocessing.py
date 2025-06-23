import logging  # Added logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from sklearn.pipeline import Pipeline, clone

from ..exceptions import (
    ConfigurationError,
    DataProcessingError,
    DataQualityError,
    FileOperationError,
)
from ..experiment_utils.seed_manager import SeedManager
from ..preprocessing.grouped import GroupedPipeline
from ..preprocessing.static_preprocessing import (
    process_static_data,
    save_static_pipeline,
)
from ..preprocessing.time_series_preprocessing import (
    fit_time_series_pipelines,
    save_time_series_pipelines,
    transform_time_series_data,
)
from ..preprocessing.unified import UnifiedPipeline
from .clean_data import (
    BasinQualityReport,
    SummaryQualityReport,
    clean_data,
    save_quality_report_to_json,
    summarize_quality_reports_from_folder,
)
from .config_utils import save_config

logger = logging.getLogger(__name__)


def create_pipeline(pipeline_config: dict[str, Any]) -> GroupedPipeline | UnifiedPipeline:
    """
    Factory function to create the appropriate pipeline based on configuration.

    Args:
        pipeline_config: Dictionary containing pipeline configuration with:
            - 'strategy': Either 'per_group' or 'unified'
            - 'pipeline': The sklearn Pipeline or GroupedPipeline object
            - 'columns': Optional list of columns to process
            - Other strategy-specific parameters

    Returns:
        Either a GroupedPipeline or UnifiedPipeline instance

    Raises:
        ConfigurationError: If strategy is invalid or required parameters are missing
    """
    strategy = pipeline_config.get("strategy", "per_group")  # Default to per_group for backwards compatibility
    pipeline = pipeline_config.get("pipeline")
    columns = pipeline_config.get("columns", [])

    if pipeline is None:
        raise ConfigurationError("Pipeline configuration must include 'pipeline' key")

    if strategy == "per_group":
        # For per_group strategy, expect a GroupedPipeline
        if isinstance(pipeline, GroupedPipeline):
            return clone(pipeline)
        else:
            raise ConfigurationError(f"Strategy 'per_group' requires GroupedPipeline, got {type(pipeline)}")

    elif strategy == "unified":
        # For unified strategy, expect a sklearn Pipeline and wrap it in UnifiedPipeline
        if isinstance(pipeline, Pipeline):
            return UnifiedPipeline(pipeline=clone(pipeline), columns=columns)
        else:
            raise ConfigurationError(f"Strategy 'unified' requires sklearn Pipeline, got {type(pipeline)}")

    else:
        raise ConfigurationError(f"Unknown strategy: {strategy}. Must be 'per_group' or 'unified'")


@dataclass
class ProcessingConfig:
    required_columns: list[str]
    preprocessing_config: dict[str, dict[str, Any]] | None = None
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
    processed_static_attributes_path: Path | None
    fitted_time_series_pipelines_path: Path
    fitted_static_pipeline_path: Path | None
    quality_reports_dir: Path
    summary_quality_report_path: Path
    config_path: Path
    success_marker_path: Path
    summary_quality_report: SummaryQualityReport


def split_data(df: pl.DataFrame, config: ProcessingConfig) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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

    target_cfg = config.preprocessing_config.get("target", {}) if config.preprocessing_config else {}
    target_col = target_cfg.get("column", "streamflow")
    group_id = config.group_identifier

    unique_basins = []
    if group_id in df.columns:
        unique_basins = df.select(pl.col(group_id)).unique().to_series().to_list()
    else:
        logger.warning(f"Group identifier '{group_id}' not found in DataFrame columns during split.")
        unique_basins = ["single_group"]
        df = df.with_columns(pl.lit("single_group").alias(group_id))

    for basin_id in unique_basins:
        basin_df = df.filter(pl.col(group_id) == basin_id).sort("date")
        valid_df = basin_df.filter(pl.col(target_col).is_not_null())
        n_valid = valid_df.height

        if n_valid == 0:
            logger.warning(f"Basin {basin_id} has no valid points for target '{target_col}', skipping split.")
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
) -> pl.LazyFrame:
    """
    Helper: lazily load one basin.

    Args:
        gauge_id: The gauge ID to load
        region_time_series_base_dirs: Dictionary mapping region prefixes to base directories
        required_columns: List of required column names
        group_identifier: Column name for group identification

    Returns:
        pl.LazyFrame: The loaded lazy frame

    Raises:
        FileOperationError: If file cannot be found or loaded
        DataQualityError: If required columns are missing
    """
    try:
        prefix = gauge_id.split("_", 1)[0]
        base_dir = region_time_series_base_dirs.get(prefix)
        if base_dir is None:
            raise FileOperationError(f"No base directory for region prefix '{prefix}'")

        file_path = Path(base_dir) / f"{gauge_id}.parquet"
        if not file_path.exists():
            raise FileOperationError(f"Timeseries file not found: {file_path}")

        lf = pl.scan_parquet(str(file_path))

        schema = lf.collect_schema().names()
        missing = [c for c in required_columns if c not in schema]
        if "date" not in schema:
            missing.append("date")
        if missing:
            raise DataQualityError(f"Missing required columns in basin {gauge_id}: {missing}")

        # Add group identifier and select/sort
        lf = lf.with_columns(pl.lit(gauge_id).alias(group_identifier))
        select_cols = [group_identifier, "date"] + required_columns
        select_cols = list(dict.fromkeys(select_cols))
        lf = lf.select(select_cols).sort("date")

        return lf

    except Exception as exc:
        if isinstance(exc, (FileOperationError, DataQualityError)):
            raise
        raise FileOperationError(f"Error loading basin {gauge_id}: {exc}")


def load_basins_timeseries_lazy(
    gauge_ids: list[str],
    region_time_series_base_dirs: dict[str, Path],
    required_columns: list[str],
    group_identifier: str = "gauge_id",
) -> pl.LazyFrame:
    """
    Lazily load & stack time-series for multiple basins.

    Args:
        gauge_ids: List of gauge IDs to load
        region_time_series_base_dirs: Dictionary mapping region prefixes to base directories
        required_columns: List of required column names
        group_identifier: Column name for group identification

    Returns:
        pl.LazyFrame: Combined lazy frame with all basins

    Raises:
        ConfigurationError: If no gauge IDs provided
        FileOperationError: If no basins could be loaded
        DataQualityError: If data quality issues are found
    """
    if not gauge_ids:
        raise ConfigurationError("No gauge IDs provided for loading")

    lazy_frames: list[pl.LazyFrame] = []
    for gid in gauge_ids:
        try:
            lf = _load_single_basin_lazy(gid, region_time_series_base_dirs, required_columns, group_identifier)
            lazy_frames.append(lf)
        except (FileOperationError, DataQualityError) as e:
            # Re-raise the specific error from the helper function
            raise e

    if not lazy_frames:
        raise FileOperationError("No basins were successfully loaded.")

    combined = pl.concat(lazy_frames, how="vertical").sort([group_identifier, "date"])
    return combined


def write_train_val_test_splits_to_disk(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    output_dir: str | Path,
    group_identifier: str = "gauge_id",
    compression: str = "zstd",
    basin_ids: list[str] | None = None,
) -> None:
    """
    Write train, validation, and test splits to disk as individual Parquet files per group.

    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        test_df: Test data DataFrame
        output_dir: Output directory path
        group_identifier: Column name for group identification
        compression: Parquet compression type
        basin_ids: Optional list of basin IDs to process

    Raises:
        FileOperationError: If writing to disk fails
    """
    base_path = Path(output_dir)

    # Infer basin IDs from the training data if not provided
    if not basin_ids:
        if train_df.height > 0 and group_identifier in train_df.columns:
            basin_ids = train_df[group_identifier].unique().to_list()
        else:
            # Handle case where train_df might be empty or missing the identifier
            # Maybe try inferring from val or test
            if val_df.height > 0 and group_identifier in val_df.columns:
                basin_ids = val_df[group_identifier].unique().to_list()
            elif test_df.height > 0 and group_identifier in test_df.columns:
                basin_ids = test_df[group_identifier].unique().to_list()
            else:
                logger.warning("Cannot infer basin IDs, no data found in splits.")
                basin_ids = []

    if not basin_ids:
        logger.warning("No basin IDs to process for writing splits.")
        return  # Nothing to write

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

    except Exception as exc:
        raise FileOperationError(f"Error writing splits to disk: {exc}")


def batch_process_time_series_data(
    lf: pl.LazyFrame,
    config: ProcessingConfig,
    features_pipeline: GroupedPipeline | UnifiedPipeline,
    target_pipeline: GroupedPipeline | UnifiedPipeline,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    dict[str, GroupedPipeline | UnifiedPipeline],
    dict[str, BasinQualityReport],
]:
    """
    Clean, split, fit on train, and transform time-series data.

    Args:
        lf: Polars LazyFrame containing the time-series data for a batch of basins.
        config: Configuration object with processing parameters.
        features_pipeline: GroupedPipeline or UnifiedPipeline for feature transformation.
        target_pipeline: GroupedPipeline or UnifiedPipeline for target transformation.

    Returns:
        tuple containing:
            - train_df: Eager Polars DataFrame for the training split.
            - val_df: Eager Polars DataFrame for the validation split.
            - test_df: Eager Polars DataFrame for the test split.
            - fitted_pipelines: Dictionary of fitted pipelines for this batch.
            - quality_reports: Dictionary of BasinQualityReport objects for basins in this batch.

    Raises:
        DataQualityError: If data cleaning or quality checks fail
        DataProcessingError: If pipeline fitting or transformation fails
    """
    try:
        cleaned_df, quality_reports = clean_data(lf, config)
    except Exception as e:
        raise DataQualityError(f"Data cleaning failed: {e}")

    valid_basins = [basin_id for basin_id, report in quality_reports.items() if report.passed_quality_check]
    if not valid_basins:
        raise DataQualityError("No valid basins found after quality checks in this batch.")

    cleaned_df_valid_basins = cleaned_df.filter(pl.col(config.group_identifier).is_in(valid_basins))

    if cleaned_df_valid_basins.height == 0:
        raise DataQualityError("Cleaned DataFrame is empty after filtering for valid basins.")

    train_df, val_df, test_df = split_data(cleaned_df_valid_basins, config)

    if train_df.height == 0:
        logger.warning("Training split is empty for this batch. Cannot fit pipelines.")
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), {}, quality_reports

    train_pd_df = train_df.to_pandas()

    # Handle fitting based on pipeline type
    fitted_batch_pipelines = {}

    # For grouped pipelines, clone and fit on this batch
    if isinstance(features_pipeline, GroupedPipeline):
        batch_features_pipeline = clone(features_pipeline)
    else:
        # For unified pipelines, use the already-fitted pipeline
        batch_features_pipeline = features_pipeline

    if isinstance(target_pipeline, GroupedPipeline):
        batch_target_pipeline = clone(target_pipeline)
    else:
        # For unified pipelines, use the already-fitted pipeline
        batch_target_pipeline = target_pipeline

    # Only fit grouped pipelines
    if isinstance(features_pipeline, GroupedPipeline) or isinstance(target_pipeline, GroupedPipeline):
        try:
            fitted_batch_pipelines = fit_time_series_pipelines(
                train_pd_df,
                batch_features_pipeline if isinstance(batch_features_pipeline, GroupedPipeline) else None,
                batch_target_pipeline if isinstance(batch_target_pipeline, GroupedPipeline) else None,
            )
        except Exception as e:
            raise DataProcessingError(f"Pipeline fitting failed: {e}")

    # Add unified pipelines to fitted pipelines dict (they're already fitted)
    if isinstance(features_pipeline, UnifiedPipeline):
        fitted_batch_pipelines["features"] = batch_features_pipeline
    if isinstance(target_pipeline, UnifiedPipeline):
        fitted_batch_pipelines["target"] = batch_target_pipeline

    val_pd_df = val_df.to_pandas() if val_df.height > 0 else pd.DataFrame()
    test_pd_df = test_df.to_pandas() if test_df.height > 0 else pd.DataFrame()

    try:
        train_transformed_pd = transform_time_series_data(train_pd_df, fitted_batch_pipelines)
    except Exception as e:
        raise DataProcessingError(f"Train transform failed: {e}")

    val_transformed_pd = pd.DataFrame()
    if not val_pd_df.empty:
        try:
            val_transformed_pd = transform_time_series_data(val_pd_df, fitted_batch_pipelines)
        except Exception as e:
            raise DataProcessingError(f"Validation transform failed: {e}")

    test_transformed_pd = pd.DataFrame()
    if not test_pd_df.empty:
        try:
            test_transformed_pd = transform_time_series_data(test_pd_df, fitted_batch_pipelines)
        except Exception as e:
            raise DataProcessingError(f"Test transform failed: {e}")

    final_train_df = pl.from_pandas(train_transformed_pd) if not train_transformed_pd.empty else pl.DataFrame()
    final_val_df = pl.from_pandas(val_transformed_pd) if not val_transformed_pd.empty else pl.DataFrame()
    final_test_df = pl.from_pandas(test_transformed_pd) if not test_transformed_pd.empty else pl.DataFrame()

    return (
        final_train_df,
        final_val_df,
        final_test_df,
        fitted_batch_pipelines,
        quality_reports,
    )


def run_hydro_processor(
    region_time_series_base_dirs: dict[str, Path],
    region_static_attributes_base_dirs: dict[str, Path],
    path_to_preprocessing_output_directory: str | Path,
    required_columns: list[str],
    run_uuid: str,
    datamodule_config: dict[str, Any],
    preprocessing_config: dict[str, dict[str, Any]],
    min_train_years: float = 5.0,
    max_imputation_gap_size: int = 5,
    group_identifier: str = "gauge_id",
    train_prop: float = 0.5,
    val_prop: float = 0.25,
    test_prop: float = 0.25,
    list_of_gauge_ids_to_process: list[str] | None = None,
    basin_batch_size: int = 50,
    random_seed: int | None = 42,
) -> ProcessingOutput:
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

    Args:
        region_time_series_base_dirs: Dictionary mapping region prefixes to time series base directories
        region_static_attributes_base_dirs: Dictionary mapping region prefixes to static attributes base directories
        path_to_preprocessing_output_directory: Path to output directory
        required_columns: List of required column names
        run_uuid: Unique identifier for this run
        datamodule_config: Configuration dictionary for data module
        preprocessing_config: Configuration dictionary for preprocessing pipelines
        min_train_years: Minimum training years required
        max_imputation_gap_size: Maximum gap size for imputation
        group_identifier: Column name for group identification
        train_prop: Proportion of data for training
        val_prop: Proportion of data for validation
        test_prop: Proportion of data for testing
        list_of_gauge_ids_to_process: Optional list of gauge IDs to process
        basin_batch_size: Batch size for processing basins
        random_seed: Random seed for deterministic basin selection in unified pipeline fitting.
                    If None, operations will be non-deterministic.

    Returns:
        ProcessingOutput: Object containing all output paths and artifacts

    Raises:
        ConfigurationError: If configuration is invalid
        FileOperationError: If file operations fail
        DataProcessingError: If data processing fails
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

        # Validate preprocessing configuration early
        from .preprocessing_validation import validate_preprocessing_config_comprehensive

        validate_preprocessing_config_comprehensive(
            preprocessing_config=preprocessing_config,
            required_columns=required_columns,
            group_identifier=group_identifier
        )

        # Prepare output directories
        run_output_dir = Path(path_to_preprocessing_output_directory) / run_uuid
        run_output_dir.mkdir(parents=True, exist_ok=True)

        ts_output_dir = run_output_dir / "processed_time_series"
        ts_output_dir.mkdir(parents=True, exist_ok=True)  # Will contain train/val/test subdirs

        quality_reports_dir = run_output_dir / "quality_reports"
        quality_reports_dir.mkdir(parents=True, exist_ok=True)

        config_path = run_output_dir / "config.json"
        summary_quality_report_path = run_output_dir / "quality_summary.json"
        success_marker_path = run_output_dir / "_SUCCESS"

        # Save DataModule config early
        try:
            save_config(datamodule_config, config_path)
        except Exception as e:
            raise FileOperationError(f"Failed to save config: {e}")
        logger.info(f"Config saved to {config_path}")

        static_features_path = run_output_dir / "processed_static_features.parquet"
        fitted_static_path = run_output_dir / "fitted_static_pipeline.joblib"
        static_pipeline_fitted: Pipeline | None = None  # Store the fitted static pipeline

        if "static_features" in preprocessing_config and list_of_gauge_ids_to_process:
            logger.info("Processing static features...")
            try:
                result = process_static_data(
                    region_static_attributes_base_dirs,
                    list_of_gauge_ids_to_process,
                    preprocessing_config,
                    static_features_path,
                    group_identifier,
                )

                # Handle the Result type
                if hasattr(result, "unwrap"):
                    # It's a Success Result
                    save_path_static, static_pipeline_fitted = result.unwrap()
                else:
                    # Handle Failure case
                    raise DataProcessingError(f"Static data processing failed: {result}")

                # Save the fitted pipeline
                pipeline_save_result = save_static_pipeline(static_pipeline_fitted, fitted_static_path)
                if hasattr(pipeline_save_result, "unwrap"):
                    pipeline_save_result.unwrap()  # Will raise if it's a Failure
                else:
                    raise FileOperationError(f"Failed to save static pipeline: {pipeline_save_result}")

                logger.info(f"Static features saved to {save_path_static}")
                logger.info(f"Static features pipeline saved to {fitted_static_path}")
            except Exception as e:
                logger.warning(f"Static data processing failed: {e}")
                static_features_path = None
                fitted_static_path = None
        else:
            logger.info("Skipping static feature processing (not configured or no gauge IDs).")
            static_features_path = None
            fitted_static_path = None

        fitted_ts_pipelines_path = run_output_dir / "fitted_time_series_pipelines.joblib"

        # Create pipeline instances using the factory function
        main_pipelines: dict[str, GroupedPipeline | UnifiedPipeline] = {}
        unified_pipelines: dict[str, UnifiedPipeline] = {}
        grouped_pipelines: dict[str, GroupedPipeline] = {}

        for pipeline_name in ["features", "target"]:
            if pipeline_name in preprocessing_config:
                pipeline = create_pipeline(preprocessing_config[pipeline_name])
                main_pipelines[pipeline_name] = pipeline

                if isinstance(pipeline, UnifiedPipeline):
                    unified_pipelines[pipeline_name] = pipeline
                elif isinstance(pipeline, GroupedPipeline):
                    pipeline.fitted_pipelines.clear()
                    grouped_pipelines[pipeline_name] = pipeline

        if not main_pipelines:
            raise ConfigurationError("No time series pipelines ('features' or 'target') found in preprocessing_config.")

        all_quality_reports: dict[str, BasinQualityReport] = {}
        processed_basin_count = 0
        valid_gauge_ids: set[str] = set()  # Track basins that pass quality checks

        if not list_of_gauge_ids_to_process:
            raise ConfigurationError("No gauge IDs provided for time series processing.")

        # Pre-validation Stage: Check all basins for quality
        logger.info("Running quality validation on all basins...")
        validation_batch_size = 100  # Process validation in reasonable batches
        original_basin_count = len(list_of_gauge_ids_to_process)  # Store original count

        for val_batch_num, val_batch_ids in enumerate(
            batch_basins(list_of_gauge_ids_to_process, validation_batch_size)
        ):
            logger.info(f"Validating batch {val_batch_num + 1} ({len(val_batch_ids)} basins)...")
            try:
                val_lf = load_basins_timeseries_lazy(
                    val_batch_ids,
                    region_time_series_base_dirs,
                    required_columns,
                    group_identifier,
                )

                # Run quality checks without raising on failure
                _, batch_quality_reports = clean_data(val_lf, config, raise_on_failure=False)

                # Track valid basins
                for basin_id, report in batch_quality_reports.items():
                    all_quality_reports[basin_id] = report
                    if report.passed_quality_check:
                        valid_gauge_ids.add(basin_id)
                    else:
                        logger.debug(f"Basin {basin_id} failed quality check: {report.failure_reason}")

            except Exception as e:
                logger.error(f"Failed to validate batch {val_batch_num + 1}: {e}")
                continue

        # Convert to list and sort for deterministic processing
        valid_gauge_ids_list = sorted(list(valid_gauge_ids))
        failed_count = original_basin_count - len(valid_gauge_ids_list)

        logger.info(f"Quality validation complete: {len(valid_gauge_ids_list)} valid basins, {failed_count} failed")

        # Save all quality reports from pre-validation
        for gauge_id, report in all_quality_reports.items():
            report_name = f"{gauge_id}.json"
            save_path = quality_reports_dir / report_name
            try:
                save_quality_report_to_json(report=report, path=save_path)
            except FileOperationError as e:
                logger.warning(f"Failed to save quality report for {gauge_id}: {e}")

        if not valid_gauge_ids_list:
            raise DataQualityError("No basins passed quality validation")

        # Update the list to only process valid basins
        list_of_gauge_ids_to_process = valid_gauge_ids_list

        # Unified Fit Stage: Fit unified pipelines on a subset of basins if configured
        if unified_pipelines:
            logger.info("Starting unified pipeline fitting stage...")

            # Check for fit_on_n_basins parameter in configs
            fit_on_n_basins = None
            for pipeline_name, pipeline_config in preprocessing_config.items():
                if pipeline_config.get("strategy") == "unified" and "fit_on_n_basins" in pipeline_config:
                    fit_on_n_basins = pipeline_config["fit_on_n_basins"]
                    break  # Use the first fit_on_n_basins found

            # If no fit_on_n_basins specified, use all valid basins
            if fit_on_n_basins is None:
                fit_on_n_basins = len(list_of_gauge_ids_to_process)

            # Initialize seed manager for deterministic basin selection
            seed_manager = SeedManager(random_seed)

            # Determine which basins to use for fitting
            if len(list_of_gauge_ids_to_process) < fit_on_n_basins:
                logger.warning(
                    f"Only {len(list_of_gauge_ids_to_process)} valid basins available, "
                    f"requested {fit_on_n_basins}. Using all available valid basins."
                )
                fit_basins = list_of_gauge_ids_to_process
            else:
                # Sort the valid basins for deterministic selection
                sorted_valid_basins = sorted(list_of_gauge_ids_to_process)

                # Use seed manager for deterministic sampling
                with seed_manager.temporary_seed("unified_basin_selection", "preprocessing_unified"):
                    import random
                    fit_basins = random.sample(sorted_valid_basins, fit_on_n_basins)

                # Log selected basins with seed information
                if len(fit_basins) > 5:
                    logger.info(
                        f"Selected {len(fit_basins)} basins for unified pipeline fitting "
                        f"(seed={random_seed}): {fit_basins[:5]}..."
                    )
                else:
                    logger.info(f"Selected basins for unified pipeline fitting (seed={random_seed}): {fit_basins}")

            # Load the selected basins for fitting
            try:
                fit_lf = load_basins_timeseries_lazy(
                    fit_basins,
                    region_time_series_base_dirs,
                    required_columns,
                    group_identifier,
                )

                # Clean and split the data (should pass quality checks now since we're using pre-validated basins)
                cleaned_df, quality_reports = clean_data(fit_lf, config)
                train_df, _, _ = split_data(cleaned_df, config)

                if train_df.height > 0:
                    train_pd = train_df.to_pandas()

                    # Fit each unified pipeline
                    for pipeline_name, pipeline in unified_pipelines.items():
                        logger.info(f"Fitting unified {pipeline_name} pipeline...")
                        try:
                            # For features pipeline, exclude target column
                            if pipeline_name == "features":
                                target_col = preprocessing_config.get("target", {}).get("column", "streamflow")
                                feature_cols = [
                                    col for col in train_pd.columns if col not in [group_identifier, "date", target_col]
                                ]
                                X = train_pd[feature_cols] if feature_cols else train_pd
                            else:
                                # For target pipeline, use target column
                                target_col = preprocessing_config.get("target", {}).get("column", "streamflow")
                                if target_col in train_pd.columns:
                                    X = train_pd[[target_col]]
                                else:
                                    X = train_pd

                            pipeline.fit(X)
                            logger.info(f"Successfully fitted unified {pipeline_name} pipeline")
                        except Exception as e:
                            raise DataProcessingError(f"Failed to fit unified {pipeline_name} pipeline: {e}")
                else:
                    logger.warning("No training data available for unified pipeline fitting")

            except Exception as e:
                raise DataProcessingError(f"Failed during unified pipeline fitting stage: {e}")

            logger.info("Completed unified pipeline fitting stage")

        logger.info(
            f"Starting time series batch processing for {len(list_of_gauge_ids_to_process)} basins in batches of {basin_batch_size}..."
        )
        for batch_num, batch_ids in enumerate(batch_basins(list_of_gauge_ids_to_process, basin_batch_size)):
            logger.info(f"--- Processing Batch {batch_num + 1} ({len(batch_ids)} basins) ---")

            # Load data for the batch
            try:
                lf = load_basins_timeseries_lazy(
                    batch_ids,
                    region_time_series_base_dirs,
                    required_columns,
                    group_identifier,
                )
            except Exception as e:
                logger.error(f"Failed to load batch {batch_num + 1}: {e}")
                continue  # Skip to next batch

            # Process the batch: clean, split, fit, transform
            try:
                train_df, val_df, test_df, batch_fitted_pipelines, quality_reports = batch_process_time_series_data(
                    lf,
                    config=config,
                    features_pipeline=main_pipelines.get("features"),
                    target_pipeline=main_pipelines.get("target"),
                )
            except Exception as e:
                logger.error(f"Failed processing batch {batch_num + 1}: {e}")
                continue  # Skip to next batch

            # Quality reports were already saved in pre-validation stage
            # Just update the collection if there are any differences
            for gauge_id, report in quality_reports.items():
                if gauge_id not in all_quality_reports:
                    # This shouldn't happen, but save it if it does
                    report_name = f"{gauge_id}.json"
                    save_path = quality_reports_dir / report_name
                    try:
                        save_quality_report_to_json(report=report, path=save_path)
                    except FileOperationError as e:
                        logger.warning(f"Failed to save quality report for {gauge_id}: {e}")
                    all_quality_reports[gauge_id] = report

            # Merge fitted pipelines from this batch into the main pipelines
            # Only add if the pipeline was successfully fitted for this batch
            if batch_fitted_pipelines:
                for pipeline_type in ["features", "target"]:
                    if pipeline_type in batch_fitted_pipelines and pipeline_type in main_pipelines:
                        batch_pipe = batch_fitted_pipelines[pipeline_type]
                        main_pipe = main_pipelines[pipeline_type]

                        # Only merge for GroupedPipelines
                        if isinstance(batch_pipe, GroupedPipeline) and isinstance(main_pipe, GroupedPipeline):
                            for group_id, group_pipeline in batch_pipe.fitted_pipelines.items():
                                main_pipe.add_fitted_group(group_id, group_pipeline)
                        # UnifiedPipelines don't need merging as they're already fitted

            # Write processed train/val/test splits to disk for this batch
            try:
                write_train_val_test_splits_to_disk(
                    train_df,
                    val_df,
                    test_df,
                    output_dir=ts_output_dir,  # Base directory for splits
                    group_identifier=group_identifier,
                    basin_ids=batch_ids,  # Write only for basins in this batch
                )
            except Exception as e:
                # Log error but continue to next batch if possible
                logger.error(f"Failed to write processed data for batch {batch_num + 1}: {e}")
                continue

            processed_basin_count += len(batch_ids)  # Count basins attempted in the batch
            logger.info(f"--- Finished Batch {batch_num + 1} ---")

        logger.info(f"Finished processing all time series batches. Attempted {processed_basin_count} basins.")

        if main_pipelines:
            try:
                saved_path = save_time_series_pipelines(main_pipelines, fitted_ts_pipelines_path)
                logger.info(f"Fitted time series pipelines saved to {saved_path}")
            except FileOperationError:
                raise
        else:
            # Handle case where no time series pipelines were defined/fitted
            logger.warning("No main time series pipelines were fitted or saved.")
            fitted_ts_pipelines_path = None  # Mark as unavailable

        try:
            summary_report = summarize_quality_reports_from_folder(quality_reports_dir, summary_quality_report_path)
        except Exception as e:
            raise FileOperationError(f"Failed to create summary report: {e}")
        logger.info(f"Summary quality report saved to {summary_quality_report_path}")

        success_marker_path.touch(exist_ok=True)
        logger.info(f"SUCCESS: Preprocessing completed successfully. Output at {run_output_dir}")

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

        return output

    except (ConfigurationError, FileOperationError, DataQualityError) as e:
        # Re-raise our custom exceptions without wrapping
        raise
    except Exception as e:
        import traceback

        logger.error(f"ERROR in run_hydro_processor: {e}\n{traceback.format_exc()}")
        raise DataProcessingError(f"Unexpected error during hydro processing: {e}")
