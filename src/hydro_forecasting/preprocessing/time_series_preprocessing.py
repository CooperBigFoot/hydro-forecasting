from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from ..exceptions import DataProcessingError, FileOperationError
from ..preprocessing.grouped import GroupedPipeline
from ..preprocessing.unified import UnifiedPipeline


def fit_time_series_pipelines(
    df: pd.DataFrame,
    features_pipeline: GroupedPipeline | None,
    target_pipeline: GroupedPipeline | None,
) -> dict[str, GroupedPipeline]:
    """
    Clone and fit separate GroupedPipelines for features and target
    on the provided DataFrame (which must include the group_identifier col).

    Note: This function only handles GroupedPipelines. UnifiedPipelines should
    be fitted separately before batch processing.

    Args:
        df: DataFrame with time series data
        features_pipeline: GroupedPipeline for features (can be None)
        target_pipeline: GroupedPipeline for target (can be None)

    Returns:
        Dictionary of fitted pipelines

    Raises:
        DataProcessingError: If fitting fails
    """
    if df is None or df.empty:
        raise DataProcessingError("DataFrame is empty or None")

    fitted_pipelines = {}

    try:
        if features_pipeline is not None:
            feat_gp = clone(features_pipeline)
            feat_gp.fit(df)
            fitted_pipelines["features"] = feat_gp

        if target_pipeline is not None:
            targ_gp = clone(target_pipeline)
            targ_gp.fit(df)
            fitted_pipelines["target"] = targ_gp

        return fitted_pipelines
    except Exception as e:
        raise DataProcessingError(f"Failed to fit time-series pipelines: {e}") from e


def transform_time_series_data(
    df: pd.DataFrame,
    fitted_pipelines: dict[str, GroupedPipeline | UnifiedPipeline],
) -> pd.DataFrame:
    """
    Apply the pre-fitted 'features' and 'target' pipelines to df.
    Supports both GroupedPipeline and UnifiedPipeline.

    Args:
        df: DataFrame to transform
        fitted_pipelines: Dictionary of fitted pipelines

    Returns:
        Transformed DataFrame

    Raises:
        DataProcessingError: If transformation fails
    """
    if df is None or df.empty:
        raise DataProcessingError("DataFrame is empty or None")

    if fitted_pipelines is None:
        raise DataProcessingError("Fitted pipelines are None")

    try:
        out = df.copy()

        # Apply features pipeline if present
        if "features" in fitted_pipelines:
            feat_pipeline = fitted_pipelines["features"]
            if feat_pipeline is not None:
                out = feat_pipeline.transform(out)

        # Apply target pipeline if present
        if "target" in fitted_pipelines:
            targ_pipeline = fitted_pipelines["target"]
            if targ_pipeline is not None:
                out = targ_pipeline.transform(out)

        return out
    except Exception as e:
        raise DataProcessingError(f"Failed to transform time-series data: {e}") from e


def save_time_series_pipelines(
    pipelines: dict[str, GroupedPipeline | UnifiedPipeline | Pipeline],
    path: Path | str,
) -> Path:
    """
    Save a dict of fitted pipelines to a .joblib file on disk.
    Supports GroupedPipeline, UnifiedPipeline, and sklearn Pipeline.

    Args:
        pipelines: Dictionary of fitted pipeline objects.
        path: Path to save the pipelines.

    Returns:
        The save path

    Raises:
        FileOperationError: If saving fails
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipelines, path)
        return path
    except Exception as e:
        raise FileOperationError(f"Failed to save pipelines to {path}: {e}") from e


def load_time_series_pipelines(
    path: Path | str,
) -> dict[str, GroupedPipeline | UnifiedPipeline | Pipeline]:
    """
    Load back the dict of pipelines previously saved with save_time_series_pipelines.

    Args:
        path: Path to the saved pipelines file.

    Returns:
        Dictionary of loaded pipelines

    Raises:
        FileOperationError: If loading fails
    """
    try:
        path = Path(path)
        pipelines = joblib.load(path)
        return pipelines
    except Exception as e:
        raise FileOperationError(f"Failed to load pipelines from {path}: {e}") from e
