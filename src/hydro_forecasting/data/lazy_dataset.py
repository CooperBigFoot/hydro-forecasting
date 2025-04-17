import torch
from torch.utils.data import Dataset
from typing import Union
import numpy as np
import polars as pl
from pathlib import Path
from .file_cache import FileCache


class HydroLazyDataset(Dataset):
    def __init__(
        self,
        batch_index_entries: list[
            dict[str, Union[Path, np.int64, np.datetime64, str, bool]]
        ],
        target: str,
        forcing_features: list[str],
        static_features: list[str],
        input_length: int,
        output_length: int,
        group_identifier: str = "gauge_id",
        domain_id: str = "source",
        domain_type: str = "source",  # Either 'source' or 'target'
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
        self.forcing_features = forcing_features
        self.static_features = static_features
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

        # Phase 1: preload static and init file cache
        self.file_cache = FileCache(max_files=50)
        self._preload_static_data()

    def _preload_static_data(self) -> None:
        """
        Read all static attributes once and cache by gauge_id.
        """
        if not self.batch_index_entries:
            self.static_data_cache = {}
            return

        path0 = self.batch_index_entries[0]["static_file_path"]
        gids = {e["gauge_id"] for e in self.batch_index_entries}
        cols = list({self.group_identifier, *self.static_features})

        df = pl.read_parquet(path0, columns=cols)

        self.static_data_cache = {}

        # Create lookup for static features
        for gid in gids:
            sub = df.filter(pl.col(self.group_identifier) == gid)
            if not sub.is_empty():
                arr = (
                    sub.select(
                        [c for c in self.static_features if c != self.group_identifier]
                    )
                    .to_numpy()
                    .flatten()
                    .astype(np.float32)
                )

            else:
                arr = np.zeros(len(self.static_features), dtype=np.float32)
            self.static_data_cache[gid] = arr

    def __len__(self):
        return len(self.batch_index_entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing tensors for X, y, future, static features, and metadata.
        """
        # Get the batch index entry for this idx
        entry = self.batch_index_entries[idx]

        df = self.file_cache.get_file(
            entry["file_path"], columns=["date", self.target, *self.forcing_features]
        )
        # slice input/output
        ts = df.slice(entry["start_idx"], entry["end_idx"] - entry["start_idx"])
        inp = ts.slice(0, self.input_length)
        out = ts.slice(self.input_length, self.output_length)

        # static
        static_arr = self.static_data_cache.get(
            entry["gauge_id"], np.zeros(len(self.static_features), dtype=np.float32)
        )

        feats = []
        if self.target in self.forcing_features:
            feats.append(inp[self.target].to_numpy())

        for f in self.forcing_features:
            if f != self.target or self.target not in self.forcing_features:
                feats.append(inp[f].to_numpy())

        X = torch.tensor(np.stack(feats, axis=1), dtype=torch.float32)
        y = torch.tensor(out[self.target].to_numpy(), dtype=torch.float32)

        # build future forcing features for decoder
        future_arr = out.select(self.forcing_features).to_numpy().astype(np.float32)
        future = torch.tensor(future_arr, dtype=torch.float32)

        # convert input_end_date -> nanoseconds since epoch as Python int
        input_end = entry["input_end_date"]
        input_end_ts = (
            input_end.astype(np.int64)
            if isinstance(input_end, np.datetime64)
            else np.datetime64(input_end).astype(np.int64)
        )
        return {
            "X": X,
            "y": y,
            "static": torch.tensor(static_arr, dtype=torch.float32),
            "future": future,
            "gauge_id": entry["gauge_id"],
            "input_end_date": int(input_end_ts),
        }
