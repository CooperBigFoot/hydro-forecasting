from pathlib import Path

import joblib
import pandas as pd
from returns.result import Failure, Result, Success
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from ..preprocessing.grouped import GroupedPipeline


def fit_time_series_pipelines(
    df: pd.DataFrame,
    features_pipeline: GroupedPipeline,
    target_pipeline: GroupedPipeline,
) -> Result[dict[str, GroupedPipeline], str]:
    """
    Clone and fit separate GroupedPipelines for features and target
    on the provided DataFrame (which must include the group_identifier col).
    """

    if df is None or df.empty:
        return Failure("DataFrame is empty or None")

    try:
        feat_gp: GroupedPipeline = clone(features_pipeline)
        targ_gp: GroupedPipeline = clone(target_pipeline)

        feat_gp.fit(df)
        targ_gp.fit(df)

        return Success({"features": feat_gp, "target": targ_gp})
    except Exception as e:
        return Failure(f"Failed to fit time‐series pipelines: {e}")


def transform_time_series_data(
    df: pd.DataFrame,
    fitted_pipelines: dict[str, GroupedPipeline],
) -> Result[pd.DataFrame, str]:
    """
    Apply the pre‐fitted 'features' and 'target' GroupedPipelines to df.
    Returns Success(transformed_df) or Failure(error_message).
    """

    if df is None or df.empty:
        return Failure("DataFrame is empty or None")

    if fitted_pipelines is None:
        return Failure("Fitted pipelines are None")

    # Retrieve exactly the pipelines we fit
    feat_gp = fitted_pipelines.get("features")
    targ_gp = fitted_pipelines.get("target")

    if feat_gp is None or targ_gp is None:
        return Failure("Must supply both 'features' and 'target' fitted pipelines")

    try:
        out = df.copy()

        # GroupedPipeline.transform expects the group_identifier column to be present,
        # and only overwrites the specified feature columns.
        out = feat_gp.transform(out)
        out = targ_gp.transform(out)

        return Success(out)
    except Exception as e:
        return Failure(f"Failed to transform time‐series data: {e}")


def save_time_series_pipelines(
    pipelines: dict[str, GroupedPipeline | Pipeline],
    path: Path | str,
) -> Result[Path, str]:
    """
    Save a dict of fitted GroupedPipelines (e.g. {"features": feat_gp, "target": targ_gp})
    to a .joblib file on disk.

    Args:
        pipelines: Dictionary of fitted GroupedPipeline or Pipeline objects.
        path: Path to save the pipelines.

    Returns:
        Success with the save path if successful, or Failure with error message.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipelines, path)
        return Success(path)
    except Exception as e:
        return Failure(f"Failed to save pipelines to {path}: {e}")


def load_time_series_pipelines(
    path: Path | str,
) -> Result[dict[str, GroupedPipeline | Pipeline], str]:
    """
    Load back the dict of GroupedPipelines previously saved with save_time_series_pipelines.

    Args:
        path: Path to the saved pipelines file.

    Returns:
        Success with the loaded pipelines dict if successful, or Failure with error message.
    """
    try:
        path = Path(path)
        pipelines = joblib.load(path)
        return Success(pipelines)
    except Exception as e:
        return Failure(f"Failed to load pipelines from {path}: {e}")
