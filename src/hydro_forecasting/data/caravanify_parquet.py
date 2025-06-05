from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd


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

    attributes_dir: str | Path
    timeseries_dir: str | Path
    gauge_id_prefix: str
    shapefile_dir: str | Path | None = None

    human_influence_path: str | Path | None = None

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
            raise ValueError(f"Found gauge IDs that don't match prefix {prefix}: {invalid_ids}")

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
            if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
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

        def load_attributes(file_name: str) -> pd.DataFrame | None:
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
            other_df = load_attributes(f"attributes_other_{self.config.gauge_id_prefix}.parquet")
            if other_df is not None:
                dfs.append(other_df)

        if self.config.use_hydroatlas_attributes:
            hydro_df = load_attributes(f"attributes_hydroatlas_{self.config.gauge_id_prefix}.parquet")
            if hydro_df is not None:
                dfs.append(hydro_df)

        if self.config.use_caravan_attributes:
            caravan_df = load_attributes(f"attributes_caravan_{self.config.gauge_id_prefix}.parquet")
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
        return df[["gauge_id", "date"] + [c for c in df.columns if c not in ("gauge_id", "date")]]

    def get_static_attributes(self) -> pd.DataFrame:
        """
        Return a copy of the merged static attributes DataFrame.

        Returns:
            A pandas DataFrame containing the static attributes.
        """
        return self.static_attributes.copy()

    def get_shapefiles(self) -> gpd.GeoDataFrame:
        """
        Load and return shapefile data as a GeoDataFrame.

        Constructs the shapefile path using the configured shapefile directory and gauge ID prefix,
        and uses geopandas to read the shapefile.

        Returns:
            A GeoDataFrame containing the shapefile data.

        Raises:
            FileNotFoundError: If the shapefile is not found at the constructed path.
        """
        shapefile_path = (
            self.config.shapefile_dir / self.config.gauge_id_prefix / f"{self.config.gauge_id_prefix}_basin_shapes.shp"
        )
        if not shapefile_path.exists():
            raise FileNotFoundError(f"Shapefile {shapefile_path} not found")

        gdf = gpd.read_file(shapefile_path)
        return gdf

    def filter_gauge_ids_by_human_influence(
        self,
        gauge_ids: list[str],
        categories: str | list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Filter a list of gauge IDs by human influence category.

        Args:
            gauge_ids: List of gauge IDs to filter
            categories: String or List of human influence categories to keep
                        (e.g., 'High', 'Medium', 'Low')

        Returns:
            Filtered list of gauge IDs matching the specified influence categories
            and a list of discarded gauge IDs that did not match the criteria.

        Raises:
            IOError: If the human influence data file cannot be loaded.
            ValueError: If the specified categories are not found in the data.
        """
        # Load human influence data
        try:
            human_influence_df = pd.read_parquet(self.config.human_influence_path, engine="pyarrow")
        except Exception as e:
            raise OSError(f"Failed to load human influence Parquet data: {e}") from e

        # Verify human influence data has required columns
        required_cols = ["gauge_id", "human_influence_category"]
        missing_cols = [col for col in required_cols if col not in human_influence_df.columns]
        if missing_cols:
            raise ValueError(f"Human influence data missing required columns: {missing_cols}")

        # Convert categories to list if a string was provided
        if isinstance(categories, str):
            categories = [categories]

        # Check if specified categories exist in the data
        available_categories = human_influence_df["human_influence_category"].unique()
        invalid_categories = [cat for cat in categories if cat not in available_categories]
        if invalid_categories:
            raise ValueError(
                f"Invalid categories: {invalid_categories}. Available categories: {available_categories.tolist()}"
            )

        # Filter human influence data to include only specified categories and gauge IDs
        filtered_hi = human_influence_df[
            (human_influence_df["human_influence_category"].isin(categories))
            & (human_influence_df["gauge_id"].isin(gauge_ids))
        ]

        # Get list of gauge_ids that match the criteria
        filtered_gauge_ids = filtered_hi["gauge_id"].unique().tolist()

        print(f"Original gauge_ids: {len(gauge_ids)}")
        print(f"Filtered gauge_ids: {len(filtered_gauge_ids)}")

        if not filtered_gauge_ids:
            print("No gauge_ids matched the specified human influence categories.")
            return [], list(gauge_ids)

        discarded_gauge_ids = list(set(gauge_ids) - set(filtered_gauge_ids))

        return filtered_gauge_ids, discarded_gauge_ids
