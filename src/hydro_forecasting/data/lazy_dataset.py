import torch
from torch.utils.data import Dataset
from typing import Union
import numpy as np
import polars as pl
from pathlib import Path
from .file_cache import FileCache
import warnings


class HydroLazyDataset(Dataset):
    """Lazy loading dataset for hydrological time series with autoregressive option.

    Args:
        batch_index_entries: list[dict[str, Path | np.int64 | np.datetime64 | str | bool]]
            Precomputed index entries including file paths, slice indices, and metadata.
        target: Name of the target variable.
        forcing_features: List of forcing features (alphabetically sorted).
        static_features: List of static features (alphabetically sorted).
        input_length: Number of timesteps for model input.
        output_length: Number of timesteps for model output.
        group_identifier: Column name for grouping (e.g., 'gauge_id').
        domain_id: Domain identifier (e.g., 'CH', 'US').
        domain_type: 'source' or 'target'.
        is_autoregressive: If True, X includes past target values as first feature.

    Attributes:
        input_features: Ordered list of features used for X tensor.
        future_features: Ordered list of features used for future tensor.
        static_data_cache: Mapping from group_identifier to numpy array of static features.
    """
    def __init__(
        self,
        batch_index_entries: list[dict[str, Path | np.int64 | np.datetime64 | str | bool]],
        target: str,
        forcing_features: list[str],
        static_features: list[str],
        input_length: int,
        output_length: int,
        group_identifier: str = "gauge_id",
        domain_id: str = "source",
        domain_type: str = "source",
        is_autoregressive: bool = False,
    ):
        """Initialize the lazy dataset with precomputed index entries.

        Args:
            batch_index_entries: List of dictionaries containing file paths and indices
            target: Name of target variable
            forcing_features: List of forcing feature names
            static_features: List of static feature names
            input_length: Length of input sequences
            output_length: Length of output sequences (prediction horizon)
            group_identifier: Column name identifying the grouping variable
            domain_id: Specific identifier for the domain (e.g., "CH", "US")
            domain_type: General type of domain - "source" or "target"
            is_autoregressive: If True, include past target values in input features
        """
        self.batch_index_entries = batch_index_entries
        self.target = target
        self.forcing_features = sorted(forcing_features)
        self.static_features = sorted(static_features)
        self.input_length = input_length
        self.output_length = output_length
        self.group_identifier = group_identifier
        self.domain_id = domain_id
        self.domain_type = domain_type
        self.is_autoregressive = is_autoregressive

        # Determine forcing indices (all features except target) if not autoregressive
        if not self.is_autoregressive:
            self.features = [self.target] + [
                f for f in forcing_features if f != self.target
            ]
            self.forcing_indices = [
                i for i, f in enumerate(self.features) if f != target
            ]
        else:
            # For autoregressive models, include all features including target
            self.features = [self.target] + [
                f for f in forcing_features if f != self.target
            ]
            self.forcing_indices = list(range(len(self.features)))

        # define ordered feature lists for X and future
        if is_autoregressive:
            self.input_features = [target] + [f for f in self.forcing_features if f != target]
        else:
            self.input_features = self.forcing_features
        self.future_features = self.forcing_features

        self.file_cache = FileCache(max_files=50)
        self._preload_static_data()

    def _preload_static_data(self) -> None:
        """Load and cache static features from all unique static_file_path entries."""
        if not self.batch_index_entries:
            self.static_data_cache = {}
            return

        unique_paths = {Path(e["static_file_path"]) for e in self.batch_index_entries}
        dfs = []
        for path in unique_paths:
            try:
                df = pl.read_parquet(path, columns=[self.group_identifier] + self.static_features)
            except (FileNotFoundError, pl.ColumnNotFoundError) as e:
                warnings.warn(f"Skipping static file {path}: {e}")
                continue
            dfs.append(df)
        if dfs:
            combined = pl.concat(dfs).unique(subset=self.group_identifier)
        else:
            combined = pl.DataFrame()
        self.static_data_cache = {
            row[self.group_identifier]: np.array(
                [row.get(f, 0.0) for f in self.static_features], dtype=np.float32
            )
            for row in combined.iter_rows(named=True)
        }

    def __len__(self):
        return len(self.batch_index_entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieve a single time series sample.

        Returns:
            dict with keys:
            - X: Tensor[input_length, len(input_features)]
            - y: Tensor[output_length]
            - static: Tensor[len(static_features)]
            - future: Tensor[output_length, len(future_features)]
            - gauge_id: str
            - input_end_date: int (nanoseconds since epoch)

        Raises:
            IndexError: If idx is out of bounds.
            IOError: If file cannot be read or sliced.
            ValueError: If tensor construction fails.
        """
        if not (0 <= idx < len(self.batch_index_entries)):
            raise IndexError(f"Index {idx} out of range")
        entry = self.batch_index_entries[idx]
        try:
            df_full = self.file_cache.get_file(
                entry["file_path"],
                columns=["date", self.target, *self.forcing_features],
            )
            ts = df_full.slice(entry["start_idx"], self.input_length + self.output_length)
            inp = ts.slice(0, self.input_length)
            out = ts.slice(self.input_length, self.output_length)
        except Exception as e:
            raise IOError(f"Error loading/slicing data for idx {idx}, path {entry['file_path']}: {e}")
        try:
            X_np = inp.select(self.input_features).to_numpy().astype(np.float32)
            X = torch.tensor(X_np, dtype=torch.float32)
            y_np = out.select(self.target).to_numpy().squeeze().astype(np.float32)
            y = torch.tensor(y_np, dtype=torch.float32)
            future_np = out.select(self.future_features).to_numpy().astype(np.float32)
            future = torch.tensor(future_np, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"Error constructing tensors for idx {idx}: {e}")
        static_arr = self.static_data_cache.get(
            entry[self.group_identifier],
            np.zeros(len(self.static_features), dtype=np.float32),
        )
        static = torch.tensor(static_arr, dtype=torch.float32)
        input_end = entry["input_end_date"]
        input_end_ts = (
            input_end.astype(np.int64)
            if isinstance(input_end, np.datetime64)
            else np.datetime64(input_end).astype(np.int64)
        )
        return {
            "X": X,
            "y": y,
            "static": static,
            "future": future,
            self.group_identifier: entry[self.group_identifier],
            "input_end_date": int(input_end_ts),
        }
