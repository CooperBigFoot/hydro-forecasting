from dataclasses import dataclass
import json
import duckdb
import pandas as pd
import numpy as np
import gc
import joblib
from pathlib import Path
from typing import Callable, Optional, Union, Any
import multiprocessing as mp
from tqdm import tqdm
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from returns.result import Failure
from .clean_data import BasinQualityReport, clean_data
from ..preprocessing.grouped import GroupedPipeline
from ..preprocessing.time_series_preprocessing import (
    fit_time_series_pipelines,
    transform_time_series_data,
    save_time_series_pipelines,
)


@dataclass
class QualityReport:
    original_basins: int
    retained_basins: int
    excluded_basins: dict[str, str]
    basins: dict[str, BasinQualityReport]
    split_method: str


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
    df: pd.DataFrame, config: ProcessingConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training, validation, and test sets based on proportions.
    Filters out NaN values first to ensure splits contain only valid data points.

    Args:
        df: DataFrame with data
        config: Configuration object

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_data, val_data, test_data = [], [], []
    target_col = config.preprocessing_config.get("target", {}).get(
        "column", "streamflow"
    )

    for gauge_id, basin_data in df.groupby(config.group_identifier):
        basin_data = basin_data.sort_values("date").reset_index(drop=True)
        valid_mask = ~basin_data[target_col].isna()
        valid_data = basin_data[valid_mask].reset_index(drop=True)
        n_valid = len(valid_data)

        if n_valid == 0:
            print(f"WARNING: Basin {gauge_id} has no valid points, skipping")
            continue

        train_size = int(n_valid * config.train_prop)
        val_size = int(n_valid * config.val_prop)
        train_valid = valid_data.iloc[:train_size]
        val_valid = valid_data.iloc[train_size : train_size + val_size]
        test_valid = valid_data.iloc[train_size + val_size :]
        train_data.append(train_valid)
        val_data.append(val_valid)
        test_data.append(test_valid)

    return (
        pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame(),
        pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame(),
        pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame(),
    )


# TODO: Migrate to Polars
def batch_process_time_series_data(
    df: pd.DataFrame,
    config: ProcessingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean, split, fit on train, and transform time-series data.

    Args:
        df: Raw DataFrame (must include group_identifier & date).
        config: ProcessingConfig with 'features' and 'target' pipelines.

    Returns:
        (train_df, val_df, test_df) all transformed.

    Raises:
        ValueError on any step failure.
    """
    cleaned_batches: list[pd.DataFrame] = []
    for basin_id, basin_df in df.groupby(config.group_identifier):
        result = clean_data(basin_df, config)

        if isinstance(result, Failure):
            print(f"WARNING: Basin {basin_id} cleaning failed, skipping")
            continue

        cleaned_df, _ = result.unwrap()
        cleaned_batches.append(cleaned_df)

    if not cleaned_batches:
        raise ValueError("No basins passed cleaning")

    cleaned_df = pd.concat(cleaned_batches, ignore_index=True)

    train_df, val_df, test_df = split_data(cleaned_df, config)

    pcfg = config.preprocessing_config or {}
    feat_cfg = pcfg.get("features")
    targ_cfg = pcfg.get("target")

    if not feat_cfg or not targ_cfg:
        raise ValueError(
            "Must define both 'features' and 'target' in preprocessing_config"
        )

    feat_pipe = feat_cfg["pipeline"]
    targ_pipe = targ_cfg["pipeline"]

    fit_res = fit_time_series_pipelines(train_df, feat_pipe, targ_pipe)

    if isinstance(fit_res, Failure):
        raise ValueError(f"Pipeline fitting failed: {fit_res.failure()}")
    fitted_pipelines = fit_res.unwrap()

    def _apply(split: pd.DataFrame) -> pd.DataFrame:
        tr = transform_time_series_data(split, fitted_pipelines)

        if isinstance(tr, Failure):
            raise ValueError(f"Transformation failed: {tr.failure()}")

        return tr.unwrap()

    train_t = _apply(train_df)
    val_t = _apply(val_df)
    test_t = _apply(test_df)

    return train_t, val_t, test_t
