from dataclasses import dataclass
import json
import duckdb
import pandas as pd
import numpy as np
import gc
import joblib
from pathlib import Path
from typing import Callable, Optional, Tuple, TypedDict, Union, Any, cast
import multiprocessing as mp
from tqdm import tqdm
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from .clean_data import BasinQualityReport, clean_data
from ..preprocessing.grouped import GroupedPipeline
from ..preprocessing.time_series_preprocessing import (
    fit_time_series_pipelines,
    transform_time_series_data,
    save_time_series_pipelines,
)


@dataclass
class QualityReport():
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


# def split_data(
#     df: pd.DataFrame, config: Config
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """
#     Split data into training, validation, and test sets based on proportions.
#     Filters out NaN values first to ensure splits contain only valid data points.

#     Args:
#         df: DataFrame with data
#         config: Configuration object

#     Returns:
#         Tuple of (train_df, val_df, test_df)
#     """
#     train_data, val_data, test_data = [], [], []
#     target_col = config.preprocessing_config.get("target", {}).get(
#         "column", "streamflow"
#     )

#     for gauge_id, basin_data in df.groupby(config.group_identifier):
#         basin_data = basin_data.sort_values("date").reset_index(drop=True)
#         valid_mask = ~basin_data[target_col].isna()
#         valid_data = basin_data[valid_mask].reset_index(drop=True)
#         n_valid = len(valid_data)

#         if n_valid == 0:
#             print(f"WARNING: Basin {gauge_id} has no valid points, skipping")
#             continue

#         train_size = int(n_valid * config.train_prop)
#         val_size = int(n_valid * config.val_prop)
#         train_valid = valid_data.iloc[:train_size]
#         val_valid = valid_data.iloc[train_size : train_size + val_size]
#         test_valid = valid_data.iloc[train_size + val_size :]
#         train_data.append(train_valid)
#         val_data.append(val_valid)
#         test_data.append(test_valid)

#     return (
#         pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame(),
#         pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame(),
#         pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame(),
#     )





