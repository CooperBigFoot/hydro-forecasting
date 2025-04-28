from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from returns.result import Result, Success, Failure
from returns.pipeline import flow
from typing import Union


def fit_static_pipeline(
    static_df: pd.DataFrame,
    static_cfg: dict[str, dict[str, Union[Pipeline, str]]],
) -> Result[Pipeline, str]:
    """
    Fit and return the static features pipeline.

    Args:
        static_df: DataFrame containing static features.
        static_cfg: Configuration dict with keys:
            - "pipeline": an unfitted sklearn Pipeline
            - "columns": list of column names to fit on

    Returns:
        Result containing fitted Pipeline instance or error message.
    """
    try:
        pipeline = clone(static_cfg["static_features"].get("pipeline", None))
        pipeline.fit(static_df)

        return Success(pipeline)
    except Exception as e:
        return Failure(f"Error fitting static pipeline: {e}")


def transform_static_data(
    static_df: pd.DataFrame,
    static_cfg: dict[str, dict[str, Union[Pipeline, str]]],
    fitted_pipeline: Pipeline,
) -> Result[pd.DataFrame, str]:
    """
    Transform static data using a fitted static features pipeline.

    Args:
        df: DataFrame with static features to transform.
        static_cfg: Configuration dict with "columns" key specifying feature columns.
        fitted_pipeline: Fitted sklearn Pipeline for static features.

    Returns:
        Result containing DataFrame with transformed static features or error message.
    """

    if fitted_pipeline is None:
        return Failure("Fitted static pipeline is required for transformation")

    columns = static_df.columns

    try:
        transformed_df = static_df.copy()
        transformed = fitted_pipeline.transform(transformed_df)

        if isinstance(transformed, np.ndarray):
            for idx, col in enumerate(columns):
                transformed_df[col] = transformed[:, idx]
        else:
            for col in columns:
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
    loaded_files_count = 0
    processed_regions_count = 0

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
                    loaded_files_count += 1

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
            processed_regions_count += 1

    if region_merged_dfs:
        final_df = pd.concat(region_merged_dfs, ignore_index=True, join="outer")
        final_df = final_df.drop_duplicates(subset=["gauge_id"]).reset_index(drop=True)

        return Success(final_df)

    return Failure("No static attribute files found or loaded for any region")


def write_static_data_to_disk(
    df: pd.DataFrame,
    output_path: Path,
) -> Result[Path, str]:
    """
    Save the static attribute DataFrame to disk as a Parquet file.

    Args:
        df: DataFrame of static features ready for saving.
        output_path: Path to save the static attributes Parquet file.

    Returns:
        Result containing Path to the saved Parquet file or error message.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)

        return Success(output_path)

    except Exception as e:
        return Failure(f"Failed to write static data to disk: {e}")


def process_static_data(
    region_static_attributes_base_dirs: dict[str, Path],
    list_of_gauge_ids: list[str],
    static_cfg: dict,
    output_path: Path,
) -> Result[Path, str]:
    """
    Complete static data processing pipeline: read, fit, transform, and write.

    Args:
        region_static_attributes_base_dirs: Mapping from region prefix to static attribute directory.
        list_of_gauge_ids: List of gauge IDs to load.
        static_cfg: Configuration dict for static features (see fit_static_pipeline).
        output_path: Path to save the processed static attributes Parquet file.

    Returns:
        Result container with Path to saved file on success, or error message on failure.
    """
    return flow(
        read_static_data_from_disk(
            region_static_attributes_base_dirs, list_of_gauge_ids
        ),
        lambda static_df: fit_static_pipeline(static_df, static_cfg),
        lambda pipeline: transform_static_data(
            read_static_data_from_disk(
                region_static_attributes_base_dirs, list_of_gauge_ids
            ).unwrap(),
            static_cfg,
            pipeline,
        ),
        lambda transformed_df: write_static_data_to_disk(transformed_df, output_path),
    )
