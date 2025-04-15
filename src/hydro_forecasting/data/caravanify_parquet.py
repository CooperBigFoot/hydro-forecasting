from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


@dataclass
class CaravanifyParquetConfig:
    """
    Configuration for loading Caravan-formatted datasets.

    Attributes:
        attributes_dir: Directory containing attribute parquet files.
        timeseries_dir: Directory containing timeseries parquet files.
        gauge_id_prefix: Prefix used to identify gauge IDs.
        shapefile_dir: Optional directory containing shapefile data.
        use_caravan_attributes: Flag to load Caravan attributes.
        use_hydroatlas_attributes: Flag to load HydroAtlas attributes.
        use_other_attributes: Flag to load other attributes.
        human_influence_path: Path to human influence classification parquet (with gauge_id and human_influence_category columns)
    """

    attributes_dir: Union[str, Path]
    timeseries_dir: Union[str, Path]
    gauge_id_prefix: str
    shapefile_dir: Optional[Union[str, Path]] = None

    human_influence_path: Optional[Union[str, Path]] = None

    use_caravan_attributes: bool = True
    use_hydroatlas_attributes: bool = False
    use_other_attributes: bool = False

    def __post_init__(self):
        """
        Convert directory paths provided as strings to Path objects.
        """
        self.attributes_dir = Path(self.attributes_dir)
        self.timeseries_dir = Path(self.timeseries_dir)
        if self.shapefile_dir:
            self.shapefile_dir = Path(self.shapefile_dir)
        if self.human_influence_path:
            self.human_influence_path = Path(self.human_influence_path)


class CaravanifyParquet:
    def __init__(self, config: CaravanifyParquetConfig):
        """
        Initialize a CaravanifyParquet instance with the provided configuration.

        Args:
            config: A CaravanifyParquetConfig object containing dataset directories, gauge ID prefix,
                    and attribute settings.

        Attributes:
            time_series: Dictionary mapping gauge_id to its timeseries DataFrame.
            static_attributes: DataFrame containing merged static attribute data.
        """
        self.config = config
        self.time_series: dict[str, pd.DataFrame] = {}  # {gauge_id: DataFrame}
        self.static_attributes = pd.DataFrame()  # Combined static attributes

    def get_all_gauge_ids(self) -> list[str]:
        """
        Retrieve all gauge IDs from the timeseries directory based on the configured prefix.

        Returns:
            A sorted list of gauge ID strings.

        Raises:
            FileNotFoundError: If the timeseries directory does not exist.
            ValueError: If any gauge IDs in the directory do not match the expected prefix.
        """
        ts_dir = self.config.timeseries_dir / self.config.gauge_id_prefix

        if not ts_dir.exists():
            raise FileNotFoundError(
                f"Timeseries directory not found for prefix {self.config.gauge_id_prefix}: {ts_dir}"
            )

        gauge_ids = [f.stem for f in ts_dir.glob("*.parquet")]
        prefix = f"{self.config.gauge_id_prefix}_"
        invalid_ids = [gid for gid in gauge_ids if not gid.startswith(prefix)]
        if invalid_ids:
            raise ValueError(
                f"Found gauge IDs that don't match prefix {prefix}: {invalid_ids}"
            )

        return sorted(gauge_ids)

    def load_stations(self, gauge_ids: list[str]) -> None:
        """
        Load station data for the specified gauge IDs.

        This method validates the provided gauge IDs and loads both the timeseries and static attribute data.

        Args:
            gauge_ids: List of gauge ID strings to load.

        Raises:
            ValueError: If any gauge ID does not conform to the expected format.
            FileNotFoundError: If required timeseries files are missing.
        """
        self._validate_gauge_ids(gauge_ids)
        self._load_timeseries(gauge_ids)
        self._load_static_attributes(gauge_ids)

    def _load_timeseries(self, gauge_ids: list[str]) -> None:
        """
        Load timeseries data for the specified gauge IDs from parquet files in parallel using multithreading.

        Each parquet file is expected to have a 'date' column which will be parsed as dates.
        The gauge ID is inferred from the file name.

        Args:
            gauge_ids: List of gauge ID strings for which to load timeseries data.

        Raises:
            FileNotFoundError: If a required timeseries file is not found.
        """
        ts_dir = self.config.timeseries_dir / self.config.gauge_id_prefix
        file_paths = []
        for gauge_id in gauge_ids:
            fp = ts_dir / f"{gauge_id}.parquet"
            if not fp.exists():
                raise FileNotFoundError(f"Timeseries file {fp} not found")
            file_paths.append(fp)

        def read_single(fp: Path) -> pd.DataFrame:
            df = pd.read_parquet(fp, engine="pyarrow")
            if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(
                df["date"]
            ):
                df["date"] = pd.to_datetime(df["date"])
            df["gauge_id"] = fp.stem
            return df

        with ThreadPoolExecutor() as executor:
            dfs = list(executor.map(read_single, file_paths))

        for df in dfs:
            self.time_series[df["gauge_id"].iloc[0]] = df

    def _load_static_attributes(self, gauge_ids: list[str]) -> None:
        """
        Load and merge static attribute data for the specified gauge IDs.

        This method reads various attribute parquet files based on the enabled attribute flags in the configuration,
        filters the data to include only rows with gauge IDs from the provided list, and merges them horizontally.

        Args:
            gauge_ids: List of gauge ID strings for which to load static attributes.
        """
        attr_dir = self.config.attributes_dir / self.config.gauge_id_prefix
        gauge_ids_set = set(gauge_ids)
        dfs = []

        def load_attributes(file_name: str) -> Union[pd.DataFrame, None]:
            """
            Load attribute data from a parquet file, filter by gauge IDs, and set 'gauge_id' as the index.

            Args:
                file_name: Name of the parquet file to load.

            Returns:
                A DataFrame with filtered attribute data, or None if the file does not exist.
            """
            file_path = attr_dir / file_name
            if not file_path.exists():
                return None

            df = pd.read_parquet(file_path, engine="pyarrow")
            # Ensure gauge_id is treated as string
            if "gauge_id" in df.columns:
                df["gauge_id"] = df["gauge_id"].astype(str)
            df = df[df["gauge_id"].isin(gauge_ids_set)]
            df.set_index("gauge_id", inplace=True)
            return df

        # Load enabled attribute types based on configuration flags
        if self.config.use_other_attributes:
            other_df = load_attributes(
                f"attributes_other_{self.config.gauge_id_prefix}.parquet"
            )
            if other_df is not None:
                dfs.append(other_df)

        if self.config.use_hydroatlas_attributes:
            hydro_df = load_attributes(
                f"attributes_hydroatlas_{self.config.gauge_id_prefix}.parquet"
            )
            if hydro_df is not None:
                dfs.append(hydro_df)

        if self.config.use_caravan_attributes:
            caravan_df = load_attributes(
                f"attributes_caravan_{self.config.gauge_id_prefix}.parquet"
            )
            if caravan_df is not None:
                dfs.append(caravan_df)

        # Concatenate all DataFrames horizontally if any were loaded
        if dfs:
            self.static_attributes = pd.concat(dfs, axis=1, join="outer").reset_index()

    def _validate_gauge_ids(self, gauge_ids: list[str]) -> None:
        """
        Validate that each gauge ID in the provided list starts with the configured prefix.

        Args:
            gauge_ids: List of gauge ID strings to validate.

        Raises:
            ValueError: If any gauge ID does not start with the expected prefix.
        """
        prefix = f"{self.config.gauge_id_prefix}_"
        for gid in gauge_ids:
            if not gid.startswith(prefix):
                raise ValueError(f"Gauge ID {gid} must start with '{prefix}'")

    def get_time_series(self) -> pd.DataFrame:
        """
        Concatenate and return all loaded timeseries data as a single DataFrame.

        The returned DataFrame includes the 'gauge_id' and 'date' columns along with all other available columns.

        Returns:
            A pandas DataFrame containing the combined timeseries data.
        """
        if not self.time_series:
            return pd.DataFrame()
        df = pd.concat(self.time_series.values(), ignore_index=True)
        return df[
            ["gauge_id", "date"]
            + [c for c in df.columns if c not in ("gauge_id", "date")]
        ]

    def get_static_attributes(self) -> pd.DataFrame:
        """
        Return a copy of the merged static attributes DataFrame.

        Returns:
            A pandas DataFrame containing the static attributes.
        """
        return self.static_attributes.copy()
