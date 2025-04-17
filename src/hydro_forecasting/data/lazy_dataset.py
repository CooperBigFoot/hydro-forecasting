import torch
from torch.utils.data import Dataset
from typing import Union
import numpy as np
import polars as pl
from pathlib import Path


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
        is_autoregressive: bool = False,  # New parameter
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
        self.is_autoregressive = is_autoregressive  # Store the new parameter

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

    def __len__(self):
        return len(self.batch_index_entries)

    def _read_parquet_range(
        self, file_path: Union[Path, str], start_idx: int, end_idx: int
    ) -> pl.DataFrame:
        """
        Efficiently read a row slice from a Parquet file using Polars.

        Args:
            file_path: Path to the Parquet file.
            start_idx: Start row index (inclusive).
            end_idx: End row index (exclusive).

        Returns:
            Polars DataFrame with the selected rows.
        """
        # Include all required columns including target and forcing features
        columns = ["date"] + [self.target] + self.forcing_features
        # Remove duplicates while preserving order
        columns = list(dict.fromkeys(columns))

        return pl.read_parquet(
            file_path, columns=columns, row_count_name=None, use_pyarrow=False
        ).slice(start_idx, end_idx - start_idx)

    def _get_static_attributes_from_id(
        self, path_to_static: Union[Path, str], gauge_id: str, static_columns: list[str]
    ) -> pl.DataFrame:
        """
        Retrieve static attributes for a specific gauge ID from a Polars DataFrame.

        Args:
            path_to_static: Path to the static attributes parquet file.
            gauge_id: The gauge ID to filter for.
            static_columns: List of static attributes to retrieve.

        Returns:
            Polars DataFrame with static attributes for the specified gauge.
        """
        # Ensure gauge_id column is included in the columns to read
        columns_to_read = list(set([self.group_identifier] + static_columns))
        static_df = pl.read_parquet(str(path_to_static), columns=columns_to_read)
        filtered_df = static_df.filter(pl.col(self.group_identifier) == gauge_id)
        return filtered_df

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
        gauge_id = entry["gauge_id"]
        start_idx = entry["start_idx"]
        end_idx = entry["end_idx"]
        input_end_date = entry["input_end_date"]

        # Read time series data: this is where the lazy magic happens
        time_series = self._read_parquet_range(
            file_path=entry["file_path"],
            start_idx=start_idx,
            end_idx=end_idx,
        )

        # Split into input and output periods
        input_data = time_series.slice(0, self.input_length)
        output_data = time_series.slice(self.input_length, self.output_length)

        # Get static attributes: more lazy magic
        static_df = self._get_static_attributes_from_id(
            path_to_static=entry["static_file_path"],
            gauge_id=gauge_id,
            static_columns=self.static_features,
        )

        # Prepare input tensor X (with target in first position if in forcing_features)
        features_list = []

        # First add target if it's in the forcing features
        if self.target in self.forcing_features:
            features_list.append(input_data[self.target].to_numpy())

        # Then add all other forcing features
        for feat in self.forcing_features:
            if feat != self.target or self.target not in self.forcing_features:
                features_list.append(input_data[feat].to_numpy())

        # Stack all features along the column axis
        X = np.column_stack(features_list)

        # Prepare target tensor y for the output period
        y = output_data[self.target].to_numpy()

        # Prepare future forcing features (excluding target)
        future_features = []
        for feat in self.forcing_features:
            if feat != self.target:
                future_features.append(output_data[feat].to_numpy())

        # Stack future features along the column axis
        future = np.column_stack(future_features) if future_features else np.array([])

        # Prepare static features
        if not static_df.is_empty():
            # Extract only the features, not the gauge_id
            static_values = (
                static_df.select(
                    [
                        col
                        for col in self.static_features
                        if col != self.group_identifier
                    ]
                )
                .to_numpy()
                .flatten()
            )
        else:
            # Empty static tensor if no data available
            static_values = np.zeros(len(self.static_features), dtype=np.float32)

        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        future_tensor = torch.tensor(future, dtype=torch.float32)
        static_tensor = torch.tensor(static_values, dtype=torch.float32)

        # Create domain tensor (1.0 for target, 0.0 for source)
        domain_tensor = torch.tensor(
            [1.0 if self.domain_type == "target" else 0.0], dtype=torch.float32
        )

        return {
            "X": X_tensor,
            "y": y_tensor,
            "future": future_tensor,
            "static": static_tensor,
            "domain_id": domain_tensor,
            "domain_name": self.domain_id,
            self.group_identifier: gauge_id,
            "input_end_date": str(input_end_date),
        }
