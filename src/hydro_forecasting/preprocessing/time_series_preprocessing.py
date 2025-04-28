import pandas as pd
from sklearn.base import clone
from returns.result import Result, Success, Failure
from ..preprocessing.grouped import GroupedPipeline


def fit_time_series_pipelines(
    df: pd.DataFrame,
    features_pipeline: GroupedPipeline,
    target_pipeline: GroupedPipeline,
) -> dict[str, GroupedPipeline]:
    """
    Fit and return grouped pipelines for time-series features and target on the given dataframe.

    Args:
        df: DataFrame containing training data for all groups.
        features_pipeline: GroupedPipeline for features.
        target_pipeline: GroupedPipeline for target.
    Returns:
        Dict with keys 'features' and 'target' mapping to the fitted pipelines.
    """

    try:
        features_gp = clone(features_pipeline)
        target_gp = clone(target_pipeline)

        # Fit on entire DataFrame
        features_gp.fit(df)
        target_gp.fit(df)

        return Success(
            {
                "features": features_gp,
                "target": target_gp,
            }
        )
    except Exception as e:
        return Failure(f"Error fitting time series pipelines: {e}")


def transform_time_series_data(
    df: pd.DataFrame,
    time_series_cfg: dict[str, dict[str, GroupedPipeline]],
    fitted_pipelines: dict[str, GroupedPipeline],
) -> Result[pd.DataFrame, str]:
    """
    Transform a time series DataFrame using pre-fitted grouped pipelines.

    Args:
        df: Input time series DataFrame to transform.
        time_series_cfg: Configuration dictionary containing preprocessing settings.
        fitted_pipelines: Dict mapping 'features' and/or 'target' to fitted Pipeline or GroupedPipeline.
        basin_id: Optional basin identifier; used if group_identifier column is missing in df.

    Returns:
        Success(DataFrame) with transformed data, or Failure(str) with error message.
    """

    feature_pipeline = time_series_cfg["features"].get("pipeline", None)
    target_pipeline = time_series_cfg["target"].get("pipeline", None)

    if feature_pipeline is None or target_pipeline is None:
        return Failure("Both feature and target pipelines are required for transformation")


    try:
        transformed_df = df.copy()


        # Feature transformation
        if "features" in fitted_pipelines:
            feature_cols = feature_pipeline.columns
            data_to_transform = transformed_df[feature_cols]

            transformed_features = feature_pipeline.transform(data_to_transform)

            for col in feature_cols:
                transformed_df[col] = transformed_features[col]

        # Target transformation
        if "target" in fitted_pipelines:
            target_col = target_pipeline.columns

            data_to_transform = transformed_df[target_col]

            transformed_target = target_pipeline.transform(data_to_transform)

            transformed_df[target_col] = transformed_target[target_col]

        return Success(transformed_df)

    except Exception as e:
        return Failure(f"Error transforming time series data: {e}")
