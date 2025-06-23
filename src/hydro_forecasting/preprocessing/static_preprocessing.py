import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from ..exceptions import ConfigurationError, DataProcessingError, FileOperationError

logger = logging.getLogger(__name__)


def fit_static_pipeline(
    static_df: pd.DataFrame,
    preprocessing_config: dict[str, dict[str, Pipeline | str]],
) -> Pipeline:
    """
    Fit and return the static features pipeline.

    Args:
        static_df: DataFrame containing static features.
        preprocessing_config: Configuration dict with keys:
            - "pipeline": an unfitted sklearn Pipeline
            - "columns": list of column names to fit on

    Returns:
        Fitted Pipeline instance.
        
    Raises:
        DataProcessingError: If static DataFrame is empty or columns are missing.
        ConfigurationError: If configuration is invalid.
    """
    if static_df is None or static_df.empty:
        raise DataProcessingError("Static DataFrame is empty or None")

    try:
        cols: list[str] = preprocessing_config["static_features"]["columns"]
    except KeyError as e:
        raise ConfigurationError(f"Missing configuration key: {e}")
    
    missing = [c for c in cols if c not in static_df.columns]
    if missing:
        raise DataProcessingError(f"Missing columns in static df: {missing}")

    try:
        pipeline: Pipeline = clone(preprocessing_config["static_features"]["pipeline"])
        pipeline.fit(static_df[cols])
        return pipeline
    except Exception as e:
        raise DataProcessingError(f"Error fitting static pipeline: {e}") from e


def transform_static_data(
    static_df: pd.DataFrame,
    preprocessing_config: dict[str, dict[str, Pipeline | str]],
    fitted_pipeline: Pipeline,
) -> pd.DataFrame:
    """
    Transform static data using a fitted static features pipeline.

    Args:
        static_df: DataFrame with static features to transform.
        preprocessing_config: Configuration dict with "columns" key specifying feature columns.
        fitted_pipeline: Fitted sklearn Pipeline for static features.

    Returns:
        DataFrame with transformed static features.
        
    Raises:
        DataProcessingError: If static DataFrame is empty, pipeline is None, or columns are missing.
        ConfigurationError: If configuration is invalid.
    """
    if static_df is None or static_df.empty:
        raise DataProcessingError("Static DataFrame is empty or None")

    if fitted_pipeline is None:
        raise DataProcessingError("Fitted static pipeline is required for transformation")

    try:
        cols: list[str] = preprocessing_config["static_features"]["columns"]
    except KeyError as e:
        raise ConfigurationError(f"Missing configuration key: {e}")
    
    missing = [c for c in cols if c not in static_df.columns]
    if missing:
        raise DataProcessingError(f"Missing columns in static df for transform: {missing}")

    try:
        transformed_df = static_df.copy()
        transformed = fitted_pipeline.transform(static_df[cols])

        if isinstance(transformed, np.ndarray):
            for idx, col in enumerate(cols):
                transformed_df[col] = transformed[:, idx]
        else:
            for col in cols:
                transformed_df[col] = transformed[col]

        return transformed_df
    except Exception as e:
        raise DataProcessingError(f"Error transforming static data: {e}") from e


def read_static_data_from_disk(
    region_static_attributes_base_dirs: dict[str, Path],
    list_of_gauge_ids: list[str],
) -> pd.DataFrame:
    """
    Load and merge static attribute data from multiple regions on disk.

    Args:
        region_static_attributes_base_dirs: Mapping from region prefix to static attribute directory
        list_of_gauge_ids: List of gauge IDs to load

    Returns:
        DataFrame with merged static attributes.
        
    Raises:
        ConfigurationError: If no gauge IDs are provided.
        FileOperationError: If file operations fail.
        DataProcessingError: If no static attribute files are found.
    """
    if not list_of_gauge_ids:
        raise ConfigurationError("No gauge IDs provided for static attribute loading")

    region_to_gauges: dict[str, list[str]] = {}
    for gid in list_of_gauge_ids:
        prefix = gid.split("_")[0]
        region_to_gauges.setdefault(prefix, []).append(gid)

    region_merged_dfs: list[pd.DataFrame] = []
    for region, gauges in region_to_gauges.items():
        static_dir = region_static_attributes_base_dirs.get(region)
        if static_dir is None:
            logger.warning("No static attribute directory for region '%s'", region)
            continue

        region_type_dfs: list[pd.DataFrame] = []
        for file_type in ["caravan", "hydroatlas", "other"]:
            file_path = static_dir / f"attributes_{file_type}_{region}.parquet"
            if not file_path.exists():
                continue
            try:
                df_type = pd.read_parquet(file_path)
                if "gauge_id" not in df_type.columns:
                    logger.warning("'gauge_id' missing in %s, skipping", file_path)
                    continue
                df_type["gauge_id"] = df_type["gauge_id"].astype(str)
                filtered = df_type[df_type["gauge_id"].isin(gauges)].copy()
                if not filtered.empty:
                    region_type_dfs.append(filtered)
            except Exception as e:
                logger.error("Error loading %s: %s", file_path, str(e))

        if region_type_dfs:
            merged_region_df = region_type_dfs[0]
            for i, df_to_merge in enumerate(region_type_dfs[1:], start=1):
                overlap = set(merged_region_df.columns) & set(df_to_merge.columns) - {"gauge_id"}
                if overlap:
                    logger.info("Overlapping columns during merge: %s", ', '.join(overlap))
                merged_region_df = pd.merge(
                    merged_region_df,
                    df_to_merge,
                    on="gauge_id",
                    how="outer",
                    suffixes=("", f"_{i}"),
                )
            region_merged_dfs.append(merged_region_df)

    if region_merged_dfs:
        final_df = pd.concat(region_merged_dfs, ignore_index=True, join="outer")
        final_df = final_df.drop_duplicates(subset=["gauge_id"]).reset_index(drop=True)
        return final_df

    raise DataProcessingError("No static attribute files found or loaded for any region")


def write_static_data_to_disk(
    df: pd.DataFrame,
    output_path: Path,
    columns_to_save: list[str],
    group_identifier: str = "gauge_id",
) -> Path:
    """
    Save specified columns of the static attribute DataFrame to disk as a Parquet file.

    Args:
        df: DataFrame of static features to save.
        output_path: Path to save the static attributes Parquet file.
        columns_to_save: List of column names to include in the saved file.
        group_identifier: Name of the column that identifies the gauges (default: "gauge_id").

    Returns:
        Path to the saved Parquet file.
        
    Raises:
        DataProcessingError: If group identifier is not found in DataFrame.
        FileOperationError: If file operations fail.
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise FileOperationError(f"Failed to create output directory: {e}") from e

    # Verify group identifier is in the DataFrame
    if group_identifier not in df.columns:
        raise DataProcessingError(f"Group identifier '{group_identifier}' not found in DataFrame")

    # Create list of columns to keep
    final_columns = [group_identifier] + [col for col in columns_to_save if col != group_identifier]

    # Filter columns that exist in the DataFrame
    existing_columns = [col for col in final_columns if col in df.columns]
    missing_columns = set(final_columns) - set(existing_columns)

    if missing_columns:
        logger.warning("Columns not found in DataFrame: %s", missing_columns)

    try:
        # Save only the specified columns
        df[existing_columns].to_parquet(output_path)
        return output_path
    except Exception as e:
        raise FileOperationError(f"Failed to write static data to disk: {e}") from e


def process_static_data(
    region_static_attributes_base_dirs: dict[str, Path | str],
    list_of_gauge_ids: list[str],
    preprocessing_config: dict[str, dict[str, Pipeline | str]],
    output_path: Path | str,
    group_identifier: str = "gauge_id",
) -> tuple[Path, Pipeline]:
    """
    Complete static data processing pipeline: read, fit, transform, and write.

    Args:
        region_static_attributes_base_dirs: Mapping from region prefix to static attribute directory.
        list_of_gauge_ids: List of gauge IDs to load.
        preprocessing_config: Configuration dict for static features (see fit_static_pipeline).
        output_path: Path to save the processed static attributes Parquet file.
        group_identifier: Name of the column that identifies the gauges (default: "gauge_id").

    Returns:
        Tuple of (Path to saved file, fitted Pipeline).
        
    Raises:
        ConfigurationError: If configuration is invalid.
        DataProcessingError: If data processing fails.
        FileOperationError: If file operations fail.
    """
    # Cast all paths to Path objects
    region_static_attributes_base_dirs = {
        k: Path(v) if isinstance(v, str) else v for k, v in region_static_attributes_base_dirs.items()
    }

    if isinstance(output_path, str):
        output_path = Path(output_path)

    try:
        # Get columns to transform from config
        columns_to_save = preprocessing_config["static_features"]["columns"]
    except KeyError as e:
        raise ConfigurationError(f"Missing configuration key: {e}")

    # Process data step by step
    static_df = read_static_data_from_disk(region_static_attributes_base_dirs, list_of_gauge_ids)
    pipeline = fit_static_pipeline(static_df, preprocessing_config)
    transformed_df = transform_static_data(static_df, preprocessing_config, pipeline)
    saved_path = write_static_data_to_disk(
        transformed_df,
        output_path,
        columns_to_save=columns_to_save,
        group_identifier=group_identifier,
    )
    
    return saved_path, pipeline


def save_static_pipeline(pipeline: Pipeline, filepath: Path | str) -> Path:
    """
    Save a fitted static features Pipeline object to disk.

    Args:
        pipeline: Fitted Pipeline object for static features
        filepath: Path where the pipeline will be saved

    Returns:
        Path where the pipeline was saved.
        
    Raises:
        FileOperationError: If file operations fail.
    """
    filepath = Path(filepath)
    try:
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Save pipeline using joblib
        joblib.dump(pipeline, filepath)
        return filepath
    except Exception as e:
        raise FileOperationError(f"Failed to save static pipeline: {e}") from e


def load_static_pipeline(filepath: Path | str) -> Pipeline:
    """
    Load a fitted static features Pipeline object from disk.

    Args:
        filepath: Path where the pipeline is saved

    Returns:
        Loaded Pipeline object.
        
    Raises:
        FileOperationError: If file operations fail.
    """
    filepath = Path(filepath)
    try:
        # Load pipeline using joblib
        pipeline = joblib.load(filepath)
        return pipeline
    except Exception as e:
        raise FileOperationError(f"Failed to load static pipeline: {e}") from e
