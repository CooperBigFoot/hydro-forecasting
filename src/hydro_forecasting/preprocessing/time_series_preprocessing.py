from typing import Dict, Optional, Union
import joblib
from pathlib import Path
import pandas as pd
from sklearn.base import clone
from returns.result import Result, Success, Failure
from ..preprocessing.grouped import GroupedPipeline


def fit_time_series_pipelines(
    df: pd.DataFrame,
    features_pipeline: GroupedPipeline,
    target_pipeline: GroupedPipeline,
) -> Result[Dict[str, GroupedPipeline], str]:
    """
    Clone and fit separate GroupedPipelines for features and target
    on the provided DataFrame (which must include the group_identifier col).
    """
    try:
        feat_gp = clone(features_pipeline)
        targ_gp = clone(target_pipeline)

        feat_gp.fit(df)
        targ_gp.fit(df)

        return Success({"features": feat_gp, "target": targ_gp})
    except Exception as e:
        return Failure(f"Failed to fit time‐series pipelines: {e}")


def transform_time_series_data(
    df: pd.DataFrame,
    fitted_pipelines: Dict[str, GroupedPipeline],
) -> Result[pd.DataFrame, str]:
    """
    Apply the pre‐fitted 'features' and 'target' GroupedPipelines to df.
    Returns Success(transformed_df) or Failure(error_message).
    """
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
    pipelines: Dict[str, GroupedPipeline],
    path: Union[Path, str],
) -> tuple[bool, Optional[Path], Optional[str]]:
    """
    Save a dict of fitted GroupedPipelines (e.g. {"features": feat_gp, "target": targ_gp})
    to a .joblib file on disk.

    Returns:
      (success, path if success else None, error message if failure else None)
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipelines, path)
        return True, path, None
    except Exception as e:
        return False, None, f"Failed to save pipelines to {path}: {e}"


def load_time_series_pipelines(
    path: Union[Path, str],
) -> tuple[bool, Optional[Dict[str, GroupedPipeline]], Optional[str]]:
    """
    Load back the dict of GroupedPipelines previously saved with save_time_series_pipelines.

    Returns:
      (success, pipelines dict if success else None, error message if failure else None)
    """
    try:
        path = Path(path)
        pipelines = joblib.load(path)
        return True, pipelines, None
    except Exception as e:
        return False, None, f"Failed to load pipelines from {path}: {e}"
