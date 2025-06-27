from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from .file_cache import FileCache


class HydroLazyDataset(Dataset):
    """
    Lazy loading dataset for hydrological time series with autoregressive option.

    Args:
        index_file_path: Path to main index Parquet file (sorted by file_path)
        index_meta_file_path: Path to index metadata Parquet file (for efficient length calculation)
        target: Name of the target variable.
        forcing_features: List of forcing features (alphabetically sorted).
        static_features: List of static features (alphabetically sorted).
        input_length: Number of timesteps for model input.
        output_length: Number of timesteps for model output.
        group_identifier: Column name for grouping (e.g., 'gauge_id').
        domain_id: Domain identifier (e.g., 'CH', 'US').
        domain_type: 'source' or 'target'.
        is_autoregressive: If True, X includes past target values as first feature.
    """

    def __init__(
        self,
        index_file_path: Path,
        index_meta_file_path: Path,
        target: str,
        forcing_features: list[str],
        static_features: list[str],
        input_length: int,
        output_length: int,
        group_identifier: str = "gauge_id",
        domain_id: str = "source",
        domain_type: str = "source",
        is_autoregressive: bool = False,
        index_cache_size: int = 2048,
    ):
        self.index_file_path = Path(index_file_path)
        self.index_meta_file_path = Path(index_meta_file_path)
        self.target = target
        self.forcing_features = sorted(forcing_features)
        self.static_features = sorted(static_features)
        self.input_length = input_length
        self.output_length = output_length
        self.group_identifier = group_identifier
        self.domain_id = domain_id
        self.domain_type = domain_type
        self.is_autoregressive = is_autoregressive

        # Efficiently get row count using meta file (defer to __len__)
        self._cached_len: int | None = None

        # LRU cache for index rows
        self._index_cache_size = index_cache_size

        # define ordered feature lists for X and future
        if is_autoregressive:
            self.input_features = [target] + [f for f in self.forcing_features if f != target]
        else:
            self.input_features = self.forcing_features
        self.future_features = self.forcing_features

        self.file_cache = FileCache(max_files=50)
        self._preload_static_data()

    def _preload_static_data(self) -> None:
        # Scan index file for unique static_file_path and group_identifier
        df = (
            pl.scan_parquet(self.index_file_path).select([self.group_identifier, "static_file_path"]).unique().collect()
        )
        static_paths = set(df["static_file_path"].to_list())
        dfs = []
        for path in static_paths:
            try:
                df_static = pl.read_parquet(path, columns=[self.group_identifier] + self.static_features)
            except (FileNotFoundError, pl.ColumnNotFoundError) as e:
                print(f"Skipping static file {path}: {e}")
                continue
            dfs.append(df_static)
        combined = pl.concat(dfs).unique(subset=self.group_identifier) if dfs else pl.DataFrame()
        self.static_data_cache = {
            row[self.group_identifier]: np.array([row.get(f, 0.0) for f in self.static_features], dtype=np.float32)
            for row in combined.iter_rows(named=True)
        }

    def __len__(self):
        if self._cached_len is not None:
            return self._cached_len
        path_to_metadata_file = self.index_meta_file_path
        con = duckdb.connect()
        query = f"""
        WITH meta AS (
          SELECT * FROM read_parquet('{str(path_to_metadata_file)}')
        )
        SELECT
          (count + start_row_index) AS total_length
        FROM meta
        ORDER BY start_row_index DESC
        LIMIT 1;
        """
        try:
            result = con.execute(query).fetchone()
            calculated_length = 0 if result is None or len(result) == 0 else result[0]
        except Exception as e:
            print(f"Error querying metadata file {path_to_metadata_file}: {e}")
            raise
        finally:
            con.close()
        self._cached_len = calculated_length
        return self._cached_len

    @lru_cache(maxsize=2048)
    def _get_index_entry(self, idx: int) -> dict:
        # Efficiently read a single row from Parquet index file
        df = pl.scan_parquet(self.index_file_path).slice(idx, 1).collect()
        if df.height == 0:
            raise IndexError(f"Index {idx} out of range in index file")
        row = df.to_dicts()[0]
        return row

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range")
        entry = self._get_index_entry(idx)
        try:
            df_full = self.file_cache.get_file(
                entry["file_path"],
                columns=["date", self.target, *self.forcing_features],
            )
            ts = df_full.slice(entry["start_idx"], self.input_length + self.output_length)
            inp = ts.slice(0, self.input_length)
            out = ts.slice(self.input_length, self.output_length)
        except Exception as e:
            raise OSError(f"Error loading/slicing data for idx {idx}, path {entry['file_path']}: {e}") from e
        try:
            X_np = inp.select(self.input_features).to_numpy().astype(np.float32)
            X = torch.tensor(X_np, dtype=torch.float32)
            y_np = out.select(self.target).to_numpy().squeeze().astype(np.float32)
            y = torch.tensor(y_np, dtype=torch.float32)
            future_np = out.select(self.future_features).to_numpy().astype(np.float32)
            future = torch.tensor(future_np, dtype=torch.float32)

            if torch.isnan(X).any():
                print(f"NaNs found in input tensor X for index {idx}")
            if torch.isnan(y).any():
                print(f"NaNs found in target tensor y for index {idx}")
            if torch.isnan(future).any():
                print(f"NaNs found in future tensor for index {idx}")

        except Exception as e:
            raise ValueError(f"Error constructing tensors for idx {idx}: {e}") from e

        static_arr = self.static_data_cache.get(
            entry[self.group_identifier],
            np.zeros(len(self.static_features), dtype=np.float32),
        )

        static = torch.tensor(static_arr, dtype=torch.float32)
        input_end = entry["input_end_date"]

        if input_end is None:
            raise ValueError(f"input_end_date is None for index {idx}")
        try:
            import pandas as pd

            input_end_ts = int(pd.to_datetime(input_end).to_datetime64().astype(np.int64))
        except Exception as e:
            raise ValueError(f"Could not convert input_end_date '{input_end}' to int64 for index {idx}: {e}") from e
        return {
            "X": X,
            "y": y,
            "static": static,
            "future": future,
            self.group_identifier: entry[self.group_identifier],
            "input_end_date": input_end_ts,
        }
