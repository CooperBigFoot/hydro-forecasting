import plars as pl
from torch.utils.data import Dataset
from typing import Union
import numpy as np
from pathlib import Path


class HydroLazyDataset(Dataset):
    def __init__(
        self,
        batch_index_entries: list[
            dict[str, Union[Path, np.int64, np.datetime64, str, bool]]
        ],
        static_features: list[str],
    ):
        self.batch_index_entries = batch_index_entries
        self.static_features = static_features

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
        columns = [
            "date",
            "snow_depth_water_equivalent_mean",
            "surface_net_solar_radiation_mean",
            "surface_net_thermal_radiation_mean",
            "potential_evaporation_sum_ERA5_LAND",
            "potential_evaporation_sum_FAO_PENMAN_MONTEITH",
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
            "total_precipitation_sum",
        ]
        return pl.read_parquet(
            file_path, columns=columns, row_count_name=None, use_pyarrow=False
        ).slice(start_idx, end_idx - start_idx)

    def _get_static_attributes_from_id(
        self, path_to_static: Union[Path, str], gauge_id: str, static_columns: list[str]
    ) -> pl.DataFrame:
        """
        Retrieve static attributes for a specific gauge ID from a Polars DataFrame.

        Args:
            static_df: Polars DataFrame containing static attributes for all gauges.
            gauge_id: The gauge ID to filter for.

        Returns:
            Polars DataFrame with static attributes for the specified gauge.
        """
        static_df = pl.read_parquet(str(path_to_static), columns=static_columns)
        filtered_df = static_df.filter(pl.col("gauge_id") == gauge_id)
        return filtered_df

    def __getitem__(self, idx):
        time_series = self._read_parquet_range(
            file_path=self.batch_index_entries[idx]["file_path"],
            start_idx=self.batch_index_entries[idx]["start_idx"],
            end_idx=self.batch_index_entries[idx]["end_idx"],
        )

        static = self._get_static_attributes_from_id(
            path_to_static=self.batch_index_entries[idx]["path_to_static"],
            gauge_id=self.batch_index_entries[idx]["gauge_id"],
            static_columns=self.static_features,
        )

        pass
