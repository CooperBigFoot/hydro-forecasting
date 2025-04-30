from pathlib import Path
from dataclasses import dataclass
import polars as pl
from typing import Optional, Any, Union, Iterator
from returns.result import Failure, Success, Result
from sklearn.pipeline import Pipeline, clone
from .clean_data import BasinQualityReport, clean_data
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

        schema = lf.schema
        missing = [c for c in required_columns if c not in schema]
        if "date" not in schema:
            missing.append("date")
        if missing:
            return Failure(f"Missing required columns in basin {gauge_id}: {missing}")

        lf = (
            lf.with_column(pl.lit(gauge_id).alias(group_identifier))
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
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, dict[str, GroupedPipeline]]:
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
        as a dictionary of GroupedPipelines.
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

    return train_df, val_df, test_df, fitted_pipelines


def run_hydro_processor(
    region_time_series_base_dirs: dict[str, Path],
    region_static_attributes_base_dirs: dict[str, Path],
    path_to_preprocessing_output_directory: Union[str, Path],
    required_columns: list[str],
    config: ProcessingConfig,
    preprocessing_config: dict[str, dict[str, GroupedPipeline | Pipeline]],
    run_uuid: str,
    group_identifier: str = "gauge_id",
    list_of_gauge_ids_to_process: Optional[list[str]] = None,
    basin_batch_size: int = 50,
) -> tuple[bool, Optional[ProcessingConfig], Optional[str]]:
    """
    1. Process static data
    2. Batch process time series data:
        - Define Batches
            - Read Batch Data
            - Call batch_process_time_series_data
            - Write Data to Disk
            - Next Batch
    3. If successful, write _SUCCESS file
    """
    path_to_preprocessing_output_directory = Path(
        path_to_preprocessing_output_directory
    )

    # 1. Process static data
    static_name = "processed_static_features.parquet"
    static_features_path = path_to_preprocessing_output_directory / static_name

    static_features_path.parent.mkdir(parents=True, exist_ok=True)

    static_processing_results = process_static_data(
        region_static_attributes_base_dirs,
        list_of_gauge_ids_to_process,
        preprocessing_config,
        static_features_path,
        group_identifier,
    )

    if isinstance(static_processing_results, Failure):
        print(f"ERROR: {static_processing_results.failure()}")
        return False, None, static_processing_results.failure()

    # Save the static features pipeline
    fitted_static_name = "fitted_static_pipeline.joblib"
    fitted_static_path = path_to_preprocessing_output_directory / fitted_static_name
    save_path_static, static_features_pipeline = static_processing_results.unwrap()

    save_results = save_static_pipeline(
        static_features_pipeline,
        fitted_static_path,
    )

    if isinstance(save_results, Failure):
        print(f"ERROR: {save_results.failure()}")
        return False, None, save_results.failure()

    print(f"SUCCESS: Static features saved to {save_path_static}")
    print(f"SUCCESS: Static features pipeline saved to {fitted_static_path}")

    # 2. Batch process time series data
    ts_folder = "processed_time_series"
    ts_output_dir = path_to_preprocessing_output_directory / ts_folder

    main_pipelines: dict[str, GroupedPipeline] = {
        "features": clone(preprocessing_config["features"].get("pipeline")),
        "target": clone(preprocessing_config["target"].get("pipeline")),
    }

    main_pipelines["features"].fitted_pipelines.clear()
    main_pipelines["target"].fitted_pipelines.clear()

    for batch in batch_basins(list_of_gauge_ids_to_process, basin_batch_size):
        loading_results = load_basins_timeseries_lazy(
            batch,
            region_time_series_base_dirs,
            required_columns,
            group_identifier,
        )

        if isinstance(loading_results, Failure):
            print(f"ERROR: {loading_results.failure()}")
            return False, None, loading_results.failure()

        lf = loading_results.unwrap()

        train_lf, val_lf, test_lf, batch_result_pipelines = (
            batch_process_time_series_data(
                lf,
                config=config,
                features_pipeline=main_pipelines["features"],
                target_pipeline=main_pipelines["target"],
            )
        )

        for pipeline_type in ["features", "target"]:
            for group_id, group_pipeline in batch_result_pipelines[
                pipeline_type
            ].fitted_pipelines.items():
                main_pipelines[pipeline_type].add_fitted_group(group_id, group_pipeline)

        write_results = write_train_val_test_splits_to_disk(
            train_lf,
            val_lf,
            test_lf,
            output_dir=ts_output_dir,
            group_identifier=group_identifier,
            basin_ids=batch,
        )

        if isinstance(write_results, Failure):
            print(f"ERROR: {write_results.failure()}")
            return False, None, write_results.failure()

    # Save the fitted pipelines to disk
    fitted_ts_pipelines_name = "fitted_time_series_pipelines.joblib"

    save_results = save_time_series_pipelines(main_pipelines, fitted_ts_pipelines_name)

    if isinstance(save_results, Failure):
        print(f"ERROR: {save_results.failure()}")
        return False, None, save_results.failure()

    print(f"SUCCESS: Fitted pipelines saved to {save_results.unwrap()}")
    print(f"SUCCESS: Time series data processed and saved to {ts_output_dir}")

    # 3. If successful, write _SUCCESS file
    success_file = path_to_preprocessing_output_directory / "_SUCCESS"
    success_file.touch(exist_ok=True)
    print(
        f"SUCCESS: Preprocessing completed successfully. Output at {path_to_preprocessing_output_directory}"
    )

    return True, config, None
