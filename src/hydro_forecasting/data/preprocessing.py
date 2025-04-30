from pathlib import Path
from dataclasses import dataclass, asdict
import polars as pl
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

    target_col = config.preprocessing_config.get("target", {}).get(
        "column", "streamflow"
    )
    group_id = config.group_identifier

    # iterate over each basin
    for basin_id in df.select(pl.col(group_id)).unique().to_series().to_list():
        basin_df = df.filter(pl.col(group_id) == basin_id).sort("date")
        valid_df = basin_df.filter(pl.col(target_col).is_not_null())
        n_valid = valid_df.height

        if n_valid == 0:
            print(f"WARNING: Basin {basin_id} has no valid points, skipping")
            continue

        # compute split sizes
        train_end = int(n_valid * config.train_prop)
        val_end = train_end + int(n_valid * config.val_prop)

        train_seg = valid_df.head(train_end)
        val_seg = valid_df.slice(train_end, val_end - train_end)
        test_seg = valid_df.slice(val_end, n_valid - val_end)

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
    `
        Args:
            basins: Sequence of basin identifiers (e.g., list of gauge_id strings).
            batch_size: Maximum number of basins per batch (must be >=1).

        Yields:
            Lists of basin identifiers, each of length <= batch_size.
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
    """
    Helper: lazily load one basin, wrapped in a Result.
    """
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

        lf = (
            lf.with_columns(pl.lit(gauge_id).alias(group_identifier))
            .select([group_identifier, "date", *required_columns])
            .sort("date")
        )

        return Success(lf)

    except Exception as exc:
        return Failure(f"Error loading basin {gauge_id}: {exc}")


def load_basins_timeseries_lazy(
    gauge_ids: list[str],
    region_time_series_base_dirs: dict[str, Path],
    required_columns: list[str],
    group_identifier: str = "gauge_id",
) -> Result[pl.LazyFrame, str]:
    """
    Lazily load & stack time‐series for multiple basins using Polars scan_parquet,
    returning a Result for Railway‐Oriented handling.

    Args:
        gauge_ids: List of basin IDs (e.g. ["EU_0001", "EU_0002"]).
        region_time_series_base_dirs: Mapping from region prefix to its base Path.
        required_columns: List of columns (besides "date") that must be present.
        group_identifier: Column name to tag each row with the gauge_id.

    Returns:
        Success(LazyFrame) on success, or Failure(error_message) on first error.
    """
    if not gauge_ids:
        return Failure("No gauge IDs provided for loading")

    lazy_frames: list[pl.LazyFrame] = []
    for gid in gauge_ids:
        result = _load_single_basin_lazy(
            gid, region_time_series_base_dirs, required_columns, group_identifier
        )
        if isinstance(result, Failure):
            # Short‐circuit on first error
            return result
        lazy_frames.append(result.unwrap())

    # concatenate and final sort
    combined = pl.concat(lazy_frames).sort([group_identifier, "date"])
    return Success(combined)


def write_train_val_test_splits_to_disk(
    train_lf: pl.LazyFrame,
    val_lf: pl.LazyFrame,
    test_lf: pl.LazyFrame,
    output_dir: Path | str,
    group_identifier: str = "gauge_id",
    compression: str = "zstd",
    basin_ids: Optional[list[str]] = None,
) -> Result[None, str]:
    """
    Write train, validation, and test splits to disk as individual Parquet files per group.

    Creates subdirectories 'train', 'val', and 'test' under `output_dir`, then for each
    unique value in `group_identifier` writes a separate Parquet file.

    Args:
        train_lf: Polars LazyFrame containing the training split.
        val_lf: Polars LazyFrame containing the validation split.
        test_lf: Polars LazyFrame containing the test split.
        output_dir: Base directory where split subfolders will be created.
        group_identifier: Column name to group by (e.g. 'gauge_id').
        compression: Parquet compression codec (default 'zstd').
        basin_ids: Optional list of specific gauge IDs to process. If None, all unique IDs are used.
    """
    base_path = Path(output_dir)

    # The basin ids are the same for all three splits. No need to infer them for each split.
    if not basin_ids:
        ids_df = train_lf.select(pl.col(group_identifier)).unique().collect()
        basin_ids: list[str] = ids_df[group_identifier].to_list()

        if not basin_ids:
            return Failure("No basin IDs found in train split")

    try:
        for split_name, lf in (
            ("train", train_lf),
            ("val", val_lf),
            ("test", test_lf),
        ):
            split_path = base_path / split_name
            split_path.mkdir(parents=True, exist_ok=True)

            for gid in basin_ids:
                sub_lf = lf.filter(pl.col(group_identifier) == gid)
                out_file = split_path / f"{gid}.parquet"
                sub_lf.collect().write_parquet(
                    out_file,
                    compression=compression,
                )

        return Success(None)
    except Exception as exc:
        return Failure(f"Error writing splits to disk: {exc}")


def batch_process_time_series_data(
    lf: pl.LazyFrame,
    config: ProcessingConfig,
    features_pipeline: GroupedPipeline,
    target_pipeline: GroupedPipeline,
) -> tuple[
    pl.LazyFrame,
    pl.LazyFrame,
    pl.LazyFrame,
    dict[str, GroupedPipeline],
    dict[str, BasinQualityReport],
]:
    """
    Clean, split, fit on train, and transform time-series data.

    Processes hydrological time-series data by:
    1. Cleaning the data using the clean_data function
    2. Splitting into train/val/test sets
    3. Fitting pipelines on the training data
    4. Transforming all sets with the fitted pipelines

    Args:
        lf: Polars LazyFrame containing the time-series data
        config: Configuration object with processing parameters
        features_pipeline: GroupedPipeline for feature transformation
        target_pipeline: GroupedPipeline for target transformation
        batch_size: Kept for API compatibility but not used

    Returns:
        Tuple of (train_df, val_df, test_df) as Polars LazyFrames and fitted pipelines
        as a dictionary of GroupedPipelines for each group.
        Also returns a dictionary of BasinQualityReport objects for each basin.
    """
    clean_result = clean_data(lf, config)

    if isinstance(clean_result, Failure):
        print(f"ERROR: {clean_result.failure()}")
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

    cleaned_df, quality_reports = clean_result.unwrap()

    # Get list of basins that passed quality checks
    valid_basins = [
        basin_id
        for basin_id, report in quality_reports.items()
        if report.passed_quality_check
    ]

    if not valid_basins:
        print("ERROR: No valid basins found after quality checks.")
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

    # Convert Polars DataFrame to pandas for compatibility with existing code
    train_df, val_df, test_df = split_data(cleaned_df, config)

    train_pd_df = train_df.to_pandas()
    val_pd_df = val_df.to_pandas()
    test_pd_df = test_df.to_pandas()

    if train_pd_df.empty:
        print(
            "ERROR: Training DataFrame is empty after splitting. Should not happen at this stage mate."
        )
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

    fit_result = fit_time_series_pipelines(
        train_pd_df,
        features_pipeline,
        target_pipeline,
    )

    if isinstance(fit_result, Failure):
        print(f"ERROR: {fit_result.failure()}")
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

    fitted_pipelines = fit_result.unwrap()

    # Transform each dataset and convert back to Polars
    train_df = pl.DataFrame()
    val_df = pl.DataFrame()
    test_df = pl.DataFrame()

    # Transform train set
    train_transformed = transform_time_series_data(train_pd_df, fitted_pipelines)
    if isinstance(train_transformed, Success):
        train_df = pl.from_pandas(train_transformed.unwrap()).lazy()

    # Transform validation set
    val_transformed = transform_time_series_data(val_pd_df, fitted_pipelines)
    if isinstance(val_transformed, Success):
        val_df = pl.from_pandas(val_transformed.unwrap()).lazy()

    # Transform test set
    test_transformed = transform_time_series_data(test_pd_df, fitted_pipelines)
    if isinstance(test_transformed, Success):
        test_df = pl.from_pandas(test_transformed.unwrap()).lazy()

    return train_df, val_df, test_df, fitted_pipelines, quality_reports


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
    1. Creates necessary directories for the processing run
    2. Processes static attributes if available
    3. Batch-processes time series data using the specified pipelines
    4. Saves quality reports and fitted pipelines
    5. Creates a summary quality report
    6. Creates a success marker file if everything succeeds

    Args:
        region_time_series_base_dirs: Mapping from region prefix to time series directory
        region_static_attributes_base_dirs: Mapping from region prefix to static attribute directory
        path_to_preprocessing_output_directory: Base directory for processed data
        required_columns: List of required data columns for quality checking
        run_uuid: Unique identifier for this processing run
        datamodule_config: Configuration dictionary for the data module
        preprocessing_config: Configuration for data preprocessing pipelines
        min_train_years: Minimum required years for training
        max_imputation_gap_size: Maximum gap length to impute with interpolation
        group_identifier: Column name identifying the basin
        train_prop: Proportion of data for training
        val_prop: Proportion of data for validation
        test_prop: Proportion of data for testing
        list_of_gauge_ids_to_process: List of basin (gauge) IDs to process
        basin_batch_size: Maximum number of basins to process in a single batch

    Returns:
        Result containing ProcessingOutput with paths to all artifacts on success,
        or error message string on failure
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
        ts_output_dir.mkdir(parents=True, exist_ok=True)

        quality_reports_dir = run_output_dir / "quality_reports"
        quality_reports_dir.mkdir(parents=True, exist_ok=True)

        config_path = run_output_dir / "config.json"
        summary_quality_report_path = run_output_dir / "quality_summary.json"
        success_marker_path = run_output_dir / "_SUCCESS"

        save_result = save_config(datamodule_config, config_path)

        if isinstance(save_result, Failure):
            return Failure(f"Failed to save config: {save_result.failure()}")
        print(f"SUCCESS: Config saved to {config_path}")

        # 1. Process static data
        static_features_path = run_output_dir / "processed_static_features.parquet"
        fitted_static_path = run_output_dir / "fitted_static_pipeline.joblib"
        static_pipeline = None

        static_processing_results = process_static_data(
            region_static_attributes_base_dirs,
            list_of_gauge_ids_to_process,
            preprocessing_config,
            static_features_path,
            group_identifier,
        )

        if isinstance(static_processing_results, Success):
            save_path_static, static_pipeline = static_processing_results.unwrap()

            save_results = save_static_pipeline(static_pipeline, fitted_static_path)

            if isinstance(save_results, Failure):
                return Failure(
                    f"Failed to save static pipeline: {save_results.failure()}"
                )

            print(f"SUCCESS: Static features saved to {save_path_static}")
            print(f"SUCCESS: Static features pipeline saved to {fitted_static_path}")
        else:
            print(f"WARNING: {static_processing_results.failure()}")
            static_features_path = None
            fitted_static_path = None

        # 2. Batch process time series data
        fitted_ts_pipelines_path = (
            run_output_dir / "fitted_time_series_pipelines.joblib"
        )

        main_pipelines: dict[str, GroupedPipeline] = {
            "features": clone(preprocessing_config["features"].get("pipeline")),
            "target": clone(preprocessing_config["target"].get("pipeline")),
        }

        main_pipelines["features"].fitted_pipelines.clear()
        main_pipelines["target"].fitted_pipelines.clear()

        all_quality_reports: dict[str, BasinQualityReport] = {}

        for batch in batch_basins(list_of_gauge_ids_to_process, basin_batch_size):
            loading_results = load_basins_timeseries_lazy(
                batch,
                region_time_series_base_dirs,
                required_columns,
                group_identifier,
            )

            if isinstance(loading_results, Failure):
                return Failure(f"Failed to load batch: {loading_results.failure()}")

            lf = loading_results.unwrap()

            train_lf, val_lf, test_lf, batch_result_pipelines, quality_reports = (
                batch_process_time_series_data(
                    lf,
                    config=config,
                    features_pipeline=main_pipelines["features"],
                    target_pipeline=main_pipelines["target"],
                )
            )

            # Save the quality reports
            for gauge_id, report in quality_reports.items():
                report_name = f"{gauge_id}.json"
                save_path = quality_reports_dir / report_name
                succ_flag, path, error = save_quality_report_to_json(
                    report=report,
                    path=save_path,
                )

                if not succ_flag:
                    print(
                        f"WARNING: Failed to save quality report for {gauge_id}: {error}"
                    )

                # Collect reports for summary
                all_quality_reports[gauge_id] = report

            # Merge fitted pipelines from this batch into main pipelines
            for pipeline_type in ["features", "target"]:
                for group_id, group_pipeline in batch_result_pipelines[
                    pipeline_type
                ].fitted_pipelines.items():
                    main_pipelines[pipeline_type].add_fitted_group(
                        group_id, group_pipeline
                    )

            # Write the processed data to disk
            write_results = write_train_val_test_splits_to_disk(
                train_lf,
                val_lf,
                test_lf,
                output_dir=ts_output_dir,
                group_identifier=group_identifier,
                basin_ids=batch,
            )

            if isinstance(write_results, Failure):
                return Failure(
                    f"Failed to write processed data: {write_results.failure()}"
                )

        # 3. Save the fitted time series pipelines
        save_results = save_time_series_pipelines(
            main_pipelines, fitted_ts_pipelines_path
        )

        if isinstance(save_results, Failure):
            return Failure(
                f"Failed to save time series pipelines: {save_results.failure()}"
            )

        # 4. Create summary quality report
        summary_result = summarize_quality_reports_from_folder(
            quality_reports_dir,
            summary_quality_report_path,
        )

        if isinstance(summary_result, Failure):
            return Failure(
                f"Failed to create summary report: {summary_result.failure()}"
            )

        summary_report = summary_result.unwrap()

        # 5. Create success marker
        success_marker_path.touch(exist_ok=True)

        print(
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

        print(f"ERROR in run_hydro_processor: {e}\n{traceback.format_exc()}")
        return Failure(f"Unexpected error during hydro processing: {e}")
