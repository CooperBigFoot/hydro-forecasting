from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from returns.result import Result, Success, Failure
from returns.pipeline import pipe
from returns.pointfree import bind
from typing import Union


def fit_static_pipeline(
    static_df: pd.DataFrame,
    preprocessing_config: dict[str, dict[str, Union[Pipeline, str]]],
) -> Result[Pipeline, str]:
    """
    Fit and return the static features pipeline.

    Args:
        static_df: DataFrame containing static features.
        preprocessing_config: Configuration dict with keys:
            - "pipeline": an unfitted sklearn Pipeline
            - "columns": list of column names to fit on

    Returns:
        Result containing fitted Pipeline instance or error message.
    """
    if static_df is None or static_df.empty:
        return Failure("Static DataFrame is empty or None")

    try:
        cols: list[str] = preprocessing_config["static_features"]["columns"]
        missing = [c for c in cols if c not in static_df.columns]

        if missing:
            return Failure(f"Missing columns in static df: {missing}")

        pipe = clone(preprocessing_config["static_features"]["pipeline"])
        pipe.fit(static_df[cols])

        return Success(pipe)

    except Exception as e:
        return Failure(f"Error fitting static pipeline: {e}")


def transform_static_data(
    static_df: pd.DataFrame,
    preprocessing_config: dict[str, dict[str, Union[Pipeline, str]]],
    fitted_pipeline: Pipeline,
) -> Result[pd.DataFrame, str]:
    """
    Transform static data using a fitted static features pipeline.

    Args:
        static_df: DataFrame with static features to transform.
        preprocessing_config: Configuration dict with "columns" key specifying feature columns.
        fitted_pipeline: Fitted sklearn Pipeline for static features.

    Returns:
        Result containing DataFrame with transformed static features or error message.
    """
    if static_df is None or static_df.empty:
        return Failure("Static DataFrame is empty or None")

    if fitted_pipeline is None:
        return Failure("Fitted static pipeline is required for transformation")

    try:
        cols: list[str] = preprocessing_config["static_features"]["columns"]
        missing = [c for c in cols if c not in static_df.columns]

        if missing:
            return Failure(f"Missing columns in static df for transform: {missing}")

        transformed_df = static_df.copy()

        transformed = fitted_pipeline.transform(static_df[cols])

        if isinstance(transformed, np.ndarray):
            for idx, col in enumerate(cols):
                transformed_df[col] = transformed[:, idx]
        else:
            for col in cols:
                transformed_df[col] = transformed[col]

        return Success(transformed_df)

    except Exception as e:
        return Failure(f"Error transforming static data: {e}")


def read_static_data_from_disk(
    region_static_attributes_base_dirs: dict[str, Path],
    list_of_gauge_ids: list[str],
) -> Result[pd.DataFrame, str]:
    """
    Load and merge static attribute data from multiple regions on disk.

    Args:
        region_static_attributes_base_dirs: Mapping from region prefix to static attribute directory
        list_of_gauge_ids: List of gauge IDs to load

    Returns:
        Result containing DataFrame with merged static attributes or error message.
    """
    if not list_of_gauge_ids:
        return Failure("No gauge IDs provided for static attribute loading")

    region_to_gauges: dict[str, list[str]] = {}
    for gid in list_of_gauge_ids:
        prefix = gid.split("_")[0]
        region_to_gauges.setdefault(prefix, []).append(gid)

    region_merged_dfs: list[pd.DataFrame] = []
    for region, gauges in region_to_gauges.items():
        static_dir = region_static_attributes_base_dirs.get(region)
        if static_dir is None:
            print(f"WARNING: No static attribute directory for region '{region}'")
            continue

        region_type_dfs: list[pd.DataFrame] = []
        for file_type in ["caravan", "hydroatlas", "other"]:
            file_path = static_dir / f"attributes_{file_type}_{region}.parquet"
            if not file_path.exists():
                continue
            try:
                df_type = pd.read_parquet(file_path)
                if "gauge_id" not in df_type.columns:
                    print(f"WARNING: 'gauge_id' missing in {file_path}, skipping.")
                    continue
                df_type["gauge_id"] = df_type["gauge_id"].astype(str)
                filtered = df_type[df_type["gauge_id"].isin(gauges)].copy()
                if not filtered.empty:
                    region_type_dfs.append(filtered)
            except Exception as e:
                print(f"ERROR: Error loading {file_path}: {str(e)}")

        if region_type_dfs:
            merged_region_df = region_type_dfs[0]
            for i, df_to_merge in enumerate(region_type_dfs[1:], start=1):
                overlap = set(merged_region_df.columns) & set(df_to_merge.columns) - {
                    "gauge_id"
                }
                if overlap:
                    print(
                        f"INFO: Overlapping columns during merge: {', '.join(overlap)}"
                    )
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
        return Success(final_df)

    return Failure("No static attribute files found or loaded for any region")


def write_static_data_to_disk(
    df: pd.DataFrame,
    output_path: Path,
    columns_to_save: list[str],
    group_identifier: str = "gauge_id",
) -> Result[Path, str]:
    """
    Save specified columns of the static attribute DataFrame to disk as a Parquet file.

    Args:
        df: DataFrame of static features to save.
        output_path: Path to save the static attributes Parquet file.
        columns_to_save: List of column names to include in the saved file.
        group_identifier: Name of the column that identifies the gauges (default: "gauge_id").

    Returns:
        Result containing Path to the saved Parquet file or error message.
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Verify group identifier is in the DataFrame
        if group_identifier not in df.columns:
            return Failure(
                f"Group identifier '{group_identifier}' not found in DataFrame"
            )

        # Create list of columns to keep
        final_columns = [group_identifier] + [
            col for col in columns_to_save if col != group_identifier
        ]

        # Filter columns that exist in the DataFrame
        existing_columns = [col for col in final_columns if col in df.columns]
        missing_columns = set(final_columns) - set(existing_columns)

        if missing_columns:
            print(f"WARNING: Columns not found in DataFrame: {missing_columns}")

        # Save only the specified columns
        df[existing_columns].to_parquet(output_path)

        return Success(output_path)
    except Exception as e:
        return Failure(f"Failed to write static data to disk: {e}")


def process_static_data(
    region_static_attributes_base_dirs: dict[str, Path | str],
    list_of_gauge_ids: list[str],
    preprocessing_config: dict[str, dict[str, Union[Pipeline, str]]],
    output_path: Path | str,
    group_identifier: str = "gauge_id",
) -> Result[Path, str]:
    """
    Complete static data processing pipeline: read, fit, transform, and write.

    Args:
        region_static_attributes_base_dirs: Mapping from region prefix to static attribute directory.
        list_of_gauge_ids: List of gauge IDs to load.
        preprocessing_config: Configuration dict for static features (see fit_static_pipeline).
        output_path: Path to save the processed static attributes Parquet file.
        group_identifier: Name of the column that identifies the gauges (default: "gauge_id").

    Returns:
        Result container with Path to saved file on success, or error message on failure.
    """
    # Cast all paths to Path objects
    region_static_attributes_base_dirs = {
        k: Path(v) if isinstance(v, str) else v
        for k, v in region_static_attributes_base_dirs.items()
    }

    if isinstance(output_path, str):
        output_path = Path(output_path)

    # Get columns to transform from config
    columns_to_save = preprocessing_config["static_features"]["columns"]

    # Helper functions for pipeline. I know, it's ugly AF but it works.
    def _read(_: object) -> Result[pd.DataFrame, str]:
        return read_static_data_from_disk(
            region_static_attributes_base_dirs, list_of_gauge_ids
        )

    def _fit(static_df: pd.DataFrame) -> Result[Pipeline, str]:
        return fit_static_pipeline(static_df, preprocessing_config)

    def _transform(pipeline: Pipeline) -> Result[pd.DataFrame, str]:
        # Re-read data to transform (could cache if needed)
        df = read_static_data_from_disk(
            region_static_attributes_base_dirs, list_of_gauge_ids
        ).unwrap()
        return transform_static_data(df, preprocessing_config, pipeline)

    def _write(transformed_df: pd.DataFrame) -> Result[Path, str]:
        return write_static_data_to_disk(
            transformed_df,
            output_path,
            columns_to_save=columns_to_save,
            group_identifier=group_identifier,
        )

    # Chain operations using pipe and bind
    return pipe(
        _read,
        bind(_fit),
        bind(_transform),
        bind(_write),
    )(None)
