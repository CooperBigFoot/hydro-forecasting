# TODO: Implement Railway Oriented Programming
# TODO: Fix Print Statements

import json
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Union, Any
import multiprocessing as mp
from tqdm import tqdm
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from ..preprocessing.grouped import GroupedPipeline


class BasinQualityReport(TypedDict):
    valid_period: Dict[str, Dict[str, Optional[str]]]
    processing_steps: List[str]
    imputation_info: Dict[str, Dict]


class QualityReport(TypedDict):
    original_basins: int
    retained_basins: int
    excluded_basins: Dict[str, str]
    basins: Dict[str, BasinQualityReport]
    split_method: str


class ProcessingResult(TypedDict):
    """Result of the hydro processor execution."""

    quality_report: QualityReport
    fitted_pipelines: Dict[str, Any]  # Pipeline or GroupedPipeline instances
    processed_dir: Path  # Directory with processed files
    processed_path_to_static_attributes_directory: Optional[
        Path
    ]  # Directory with processed static data


class Config:
    def __init__(
        self,
        required_columns: List[str],
        preprocessing_config: Optional[Dict[str, Dict[str, Any]]] = None,
        min_train_years: float = 5.0,
        max_imputation_gap_size: int = 5,
        group_identifier: str = "gauge_id",
        train_prop: float = 0.6,
        val_prop: float = 0.2,
        test_prop: float = 0.2,
    ):
        """
        Configuration for hydro data processing.

        Args:
            required_columns: List of required data columns for quality checking
            preprocessing_config: Configuration for data preprocessing pipelines
            min_train_years: Minimum required years for training
            max_imputation_gap_size: Maximum gap length to impute with interpolation
            group_identifier: Column name identifying the basin
            train_prop: Proportion of data for training
            val_prop: Proportion of data for validation
            test_prop: Proportion of data for testing
        """
        self.required_columns = required_columns
        self.preprocessing_config = preprocessing_config or {}
        self.min_train_years = min_train_years
        self.max_imputation_gap_size = max_imputation_gap_size
        self.group_identifier = group_identifier
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.test_prop = test_prop


def find_gaps(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find start and end indices of gaps in time series data.

    Args:
        series: Input time series

    Returns:
        Tuple containing arrays with gap start and end indices
    """
    is_missing = series.isna()
    if not is_missing.any():
        return np.array([]), np.array([])

    nan_indices = np.where(is_missing)[0]
    if len(nan_indices) == 0:
        return np.array([]), np.array([])

    gap_boundaries = np.where(np.diff(nan_indices) > 1)[0]

    gap_starts = np.array([nan_indices[0]])
    if len(gap_boundaries) > 0:
        gap_starts = np.append(gap_starts, nan_indices[gap_boundaries + 1])

    gap_ends = np.array([])
    if len(gap_boundaries) > 0:
        gap_ends = np.append(gap_ends, nan_indices[gap_boundaries] + 1)
    gap_ends = np.append(gap_ends, nan_indices[-1] + 1)

    return gap_starts, gap_ends


def impute_short_gaps(
    df: pd.DataFrame,
    columns: List[str],
    max_imputation_gap_size: int,
    basin_report: BasinQualityReport,
) -> Tuple[pd.DataFrame, BasinQualityReport]:
    """
    Linearly impute short gaps in time series data.

    Args:
        df: DataFrame with basin data
        columns: Columns to impute
        max_imputation_gap_size: Maximum gap length to impute
        basin_report: Quality report for the basin

    Returns:
        Tuple of imputed DataFrame and updated basin report
    """
    imputed_df = df.copy()

    for column in columns:
        series = imputed_df[column]
        is_nan = series.isna()

        if not is_nan.any():
            basin_report["imputation_info"][column] = {
                "short_gaps_count": 0,
                "imputed_values_count": 0,
            }
            continue

        gap_starts, gap_ends = find_gaps(series)
        short_gaps = []
        short_gap_indices = []

        for start_idx, end_idx in zip(gap_starts, gap_ends):
            gap_length = end_idx - start_idx
            if gap_length <= max_imputation_gap_size:
                short_gaps.append((start_idx, end_idx))
                short_gap_indices.extend(range(int(start_idx), int(end_idx)))

        clean_series = series.dropna()

        if not clean_series.empty and short_gap_indices:
            temp_series = series.copy()
            interpolate_mask = pd.Series(False, index=temp_series.index)
            for idx in short_gap_indices:
                if idx < len(interpolate_mask):
                    interpolate_mask.iloc[idx] = True

            if interpolate_mask.any():
                temp_series_interp = temp_series.interpolate(method="linear")
                temp_series.loc[interpolate_mask] = temp_series_interp.loc[
                    interpolate_mask
                ]
                imputed_df[column] = temp_series

        basin_report["imputation_info"][column] = {
            "short_gaps_count": len(short_gaps),
            "imputed_values_count": len(short_gap_indices),
        }

    basin_report["processing_steps"].append("Applied imputation to short gaps")
    return imputed_df, basin_report


def split_data(
    df: pd.DataFrame, config: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
            print(f"Warning: Basin {gauge_id} has no valid points, skipping")
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


def fit_pipelines(
    train_df: pd.DataFrame,
    all_data: pd.DataFrame,
    static_df: Optional[pd.DataFrame],
    config: Config,
) -> Dict[str, Union[Pipeline, GroupedPipeline]]:
    """
    Fit preprocessing pipelines on training data to prevent data leakage.

    Args:
        train_df: Training data
        all_data: All data (train + val + test)
        static_df: Static features data (if available)
        config: Configuration object

    Returns:
        Dictionary of fitted pipeline instances
    """
    fitted_pipelines = {}

    # Process features
    if "features" in config.preprocessing_config:
        pipeline_config = config.preprocessing_config["features"]
        pipeline = clone(pipeline_config["pipeline"])

        if isinstance(pipeline, GroupedPipeline):
            feature_cols = pipeline.columns
            train_features = train_df[feature_cols + [config.group_identifier]]
            pipeline.fit(train_features)
            fitted_pipelines["features"] = pipeline
        else:
            target_col = config.preprocessing_config.get("target", {}).get(
                "column", "streamflow"
            )
            feature_cols = [col for col in config.required_columns if col != target_col]
            train_features = train_df[feature_cols]
            pipeline.fit(train_features)
            fitted_pipelines["features"] = pipeline

    # Process target
    if "target" in config.preprocessing_config:
        pipeline_config = config.preprocessing_config["target"]
        pipeline = clone(pipeline_config["pipeline"])
        target_col = pipeline_config.get("column", "streamflow")

        if isinstance(pipeline, GroupedPipeline):
            train_target = train_df[pipeline.columns + [config.group_identifier]]
        else:
            train_target = train_df[[target_col]]
        pipeline.fit(train_target)
        fitted_pipelines["target"] = pipeline

    # Process static features (if static data is available)
    if "static_features" in config.preprocessing_config and static_df is not None:
        static_cfg = config.preprocessing_config["static_features"]
        static_pipeline = static_cfg["pipeline"]
        static_columns = static_cfg.get("columns")
        if static_columns is None:
            raise ValueError(
                "You must specify 'columns' for static_features in preprocessing_config."
            )
        missing_cols = [col for col in static_columns if col not in static_df.columns]
        if missing_cols:
            raise ValueError(f"Static columns not found in static_df: {missing_cols}")
        # Fit only on the specified columns
        static_pipeline.fit(static_df[static_columns])
        fitted_pipelines["static"] = static_pipeline

    return fitted_pipelines


def apply_transformations(
    df: pd.DataFrame,
    config: Config,
    fitted_pipelines: Dict[str, Union[Pipeline, GroupedPipeline]],
    basin_id: Optional[str] = None,
    static_data: bool = False,
) -> pd.DataFrame:
    """
    Apply fitted transformations to a DataFrame.

    Args:
        df: DataFrame to transform.
        config: Configuration object.
        fitted_pipelines: Fitted pipeline instances.
        basin_id: Basin identifier (if applicable).
        static_data: Flag to indicate whether processing static features.

    Returns:
        Transformed DataFrame.
    """
    transformed_df = df.copy()

    if static_data:
        # Only static_features uses the "columns" key
        static_cfg = config.preprocessing_config.get("static_features", {})
        pipeline = fitted_pipelines.get("static")
        static_columns = static_cfg.get("columns")
        if pipeline is None or static_columns is None:
            print(
                "Static transformation pipeline or columns not available; returning data unchanged."
            )
            return transformed_df
        transformed_static = pipeline.transform(transformed_df[static_columns])
        if isinstance(transformed_static, np.ndarray):
            for i, col in enumerate(static_columns):
                transformed_df[col] = transformed_static[:, i]
        else:
            for col in static_columns:
                transformed_df[col] = transformed_static[col]
        return transformed_df

    # For time series, keep existing logic (no "columns" key)
    if config.group_identifier not in transformed_df.columns and basin_id is not None:
        transformed_df[config.group_identifier] = basin_id

    # Features
    if "features" in fitted_pipelines:
        pipeline = fitted_pipelines["features"]
        target_col = config.preprocessing_config.get("target", {}).get(
            "column", "streamflow"
        )
        feature_cols = [col for col in config.required_columns if col != target_col]
        if feature_cols:
            if isinstance(pipeline, GroupedPipeline):
                features_data = transformed_df[feature_cols + [config.group_identifier]]
                transformed_features = pipeline.transform(features_data)
            else:
                features_data = transformed_df[feature_cols]
                transformed_features = pipeline.transform(features_data)
            if isinstance(transformed_features, np.ndarray):
                for i, col in enumerate(feature_cols):
                    transformed_df[col] = transformed_features[:, i]
            else:
                for col in feature_cols:
                    transformed_df[col] = transformed_features[col]

    # Target
    if "target" in fitted_pipelines:
        pipeline = fitted_pipelines["target"]
        target_col = config.preprocessing_config.get("target", {}).get(
            "column", "streamflow"
        )
        if isinstance(pipeline, GroupedPipeline):
            target_data = transformed_df[[target_col, config.group_identifier]]
            transformed_target = pipeline.transform(target_data)
        else:
            target_data = transformed_df[[target_col]]
            transformed_target = pipeline.transform(target_data)
        if isinstance(transformed_target, np.ndarray):
            transformed_df[target_col] = transformed_target[:, 0]
        else:
            transformed_df[target_col] = transformed_target[target_col]

    return transformed_df


def process_basin(
    basin_file: Path,
    config: Config,
    path_to_preprocessing_output_directory: Path,
    reports_dir: Path,
    fitted_pipelines: Optional[Dict[str, Union[Pipeline, GroupedPipeline]]] = None,
) -> Tuple[bool, Optional[str], Optional[BasinQualityReport]]:
    """
    Process a single basin file using DuckDB and pandas.

    Args:
        basin_file: Path to the basin parquet file
        config: Configuration parameters
        path_to_preprocessing_output_directory: Directory to save processed data
        reports_dir: Directory to save quality reports
        fitted_pipelines: Dictionary of fitted pipeline instances

    Returns:
        Tuple containing:
        - Success flag (True if basin passed quality checks)
        - Error message (if any)
        - Quality report for the basin
    """
    basin_id = basin_file.stem  # Get gauge_id from filename

    basin_report: BasinQualityReport = {
        "valid_period": {},
        "processing_steps": [],
        "imputation_info": {},
    }

    try:
        con = duckdb.connect(database=":memory:")
        con.execute(f"CREATE TABLE basin AS SELECT * FROM read_parquet('{basin_file}')")

        columns = con.execute("PRAGMA table_info(basin)").fetchall()
        column_names = [col[1] for col in columns]

        missing_cols = [
            col for col in config.required_columns if col not in column_names
        ]
        if missing_cols or "date" not in column_names:
            error_msg = f"Missing required columns: {missing_cols + (['date'] if 'date' not in column_names else [])}"
            return False, error_msg, basin_report

        basin_report["processing_steps"].append("Loaded basin data")

        for column in config.required_columns:
            result = con.execute(f"""
                SELECT 
                    MIN(date) FILTER (WHERE {column} IS NOT NULL) as start_date,
                    MAX(date) FILTER (WHERE {column} IS NOT NULL) as end_date
                FROM basin
            """).fetchone()

            start_date, end_date = result

            basin_report["valid_period"][column] = {
                "start": start_date.strftime("%Y-%m-%d") if start_date else None,
                "end": end_date.strftime("%Y-%m-%d") if end_date else None,
            }

        valid_starts = [
            val["start"]
            for col, val in basin_report["valid_period"].items()
            if val["start"] is not None
        ]
        valid_ends = [
            val["end"]
            for col, val in basin_report["valid_period"].items()
            if val["end"] is not None
        ]

        if not valid_starts or not valid_ends:
            return False, "No valid data period found", basin_report

        start_dates = [pd.to_datetime(date) for date in valid_starts]
        end_dates = [pd.to_datetime(date) for date in valid_ends]
        overall_start = max(start_dates)
        overall_end = min(end_dates)

        total_days = (overall_end - overall_start).days + 1
        total_years = total_days / 365.25
        train_days = int(total_days * config.train_prop)
        train_years = train_days / 365.25

        min_valid_days = 1 if config.min_train_years < 0.1 else 365

        if train_years < config.min_train_years or total_days < min_valid_days:
            required_total_years = config.min_train_years / config.train_prop
            error_msg = (
                f"Insufficient data period ({total_days} days, {train_years:.2f} training years). "
                f"Need {required_total_years:.2f} total years with current proportions."
            )
            return False, error_msg, basin_report

        con.execute(f"""
            CREATE TABLE filtered_basin AS
            SELECT * FROM basin
            WHERE date >= '{overall_start}' AND date <= '{overall_end}'
            ORDER BY date
        """)

        basin_report["processing_steps"].append("Filtered to valid period")
        df = con.execute("SELECT * FROM filtered_basin").df()
        basin_report["processing_steps"].append("Transferred to pandas for imputation")

        for column in config.required_columns:
            basin_report["imputation_info"][column] = {
                "short_gaps_count": 0,
                "imputed_values_count": 0,
            }

        imputed_df, basin_report = impute_short_gaps(
            df, config.required_columns, config.max_imputation_gap_size, basin_report
        )

        transformed_df = imputed_df
        if fitted_pipelines:
            basin_report["processing_steps"].append("Applying transformations")
            transformed_df = apply_transformations(
                imputed_df, config, fitted_pipelines, basin_id
            )
            basin_report["processing_steps"].append("Transformations applied")

        output_file = path_to_preprocessing_output_directory / f"{basin_id}.parquet"
        transformed_df.to_parquet(output_file)

        report_file = reports_dir / f"{basin_id}_report.json"
        with open(report_file, "w") as f:
            json.dump(basin_report, f, indent=2, default=str)

        basin_report["processing_steps"].append("Basin processed successfully")
        return True, None, basin_report

    except Exception as e:
        error_msg = f"Error processing basin {basin_id}: {str(e)}"
        return False, error_msg, basin_report
    finally:
        if "con" in locals():
            con.close()


def process_basin_worker(
    args: Tuple[Path, Config, Path, Path, Dict],
) -> Tuple[str, bool, Optional[str], Optional[BasinQualityReport]]:
    """
    Worker function for parallel basin processing.

    Args:
        args: Tuple containing (basin_file, config, path_to_preprocessing_output_directory, reports_dir, fitted_pipelines)

    Returns:
        Tuple containing basin_id, success flag, error message, and basin report
    """
    (
        basin_file,
        config,
        path_to_preprocessing_output_directory,
        reports_dir,
        fitted_pipelines,
    ) = args
    success, error_msg, basin_report = process_basin(
        basin_file,
        config,
        path_to_preprocessing_output_directory,
        reports_dir,
        fitted_pipelines,
    )
    basin_id = basin_file.stem
    return basin_id, success, error_msg, basin_report


def process_basins_parallel(
    basin_files: List[Path],
    config: Config,
    path_to_preprocessing_output_directory: Path,
    reports_dir: Path,
    fitted_pipelines: Dict[str, Union[Pipeline, GroupedPipeline]],
    num_processes: int,
) -> QualityReport:
    """
    Process multiple basin files in parallel.

    Args:
        basin_files: List of paths to basin parquet files
        config: Configuration parameters
        path_to_preprocessing_output_directory: Directory to save processed data
        reports_dir: Directory to save quality reports
        fitted_pipelines: Dictionary of fitted pipeline instances
        num_processes: Number of parallel processes to use

    Returns:
        Overall quality report
    """
    quality_report: QualityReport = {
        "original_basins": len(basin_files),
        "retained_basins": 0,
        "excluded_basins": {},
        "basins": {},
        "split_method": "proportional",
    }

    args_list = [
        (
            basin_file,
            config,
            path_to_preprocessing_output_directory,
            reports_dir,
            fitted_pipelines,
        )
        for basin_file in basin_files
    ]

    with mp.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_basin_worker, args_list),
                total=len(basin_files),
                desc="Processing basins",
            )
        )

    for basin_id, success, error_msg, basin_report in results:
        quality_report["basins"][basin_id] = basin_report
        if not success:
            quality_report["excluded_basins"][basin_id] = error_msg
        else:
            quality_report["retained_basins"] += 1

    return quality_report


def load_all_basins(
    path_to_time_series_directory: Path, config: Config
) -> pd.DataFrame:
    """
    Load all basin files to create training data for fitting pipelines.

    Args:
        path_to_time_series_directory: Directory containing parquet files
        config: Configuration parameters

    Returns:
        DataFrame containing data from all basins
    """
    basin_files = list(path_to_time_series_directory.glob("*.parquet"))
    if not basin_files:
        raise ValueError("No parquet files found in input directory")

    all_data = []
    for basin_file in tqdm(basin_files, desc="Loading basin data for fitting"):
        try:
            df = pd.read_parquet(basin_file)
            df[config.group_identifier] = basin_file.stem  # Ensure gauge_id is set
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {basin_file}: {str(e)}")

    if not all_data:
        raise ValueError("Failed to load any basin files")

    return pd.concat(all_data, ignore_index=True)


def load_static_attributes(
    path_to_static_attributes_directory: Path,
    gauge_id_prefix: str = None,
    list_of_gauge_ids_to_process: List[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Load static attribute data from multiple parquet files.

    Looks for three types of attribute files:
    - attributes_caravan_{gauge_id_prefix}.parquet
    - attributes_hydroatlas_{gauge_id_prefix}.parquet
    - attributes_other_{gauge_id_prefix}.parquet

    Args:
        path_to_static_attributes_directory: Directory containing static attribute files
        gauge_id_prefix: Prefix for gauge IDs (e.g., "CA" for Canadian basins)
        list_of_gauge_ids_to_process: Optional list of basin IDs to filter attributes for

    Returns:
        DataFrame with merged static attributes or None if no files found
    """
    if not path_to_static_attributes_directory.exists():
        print(f"Static directory {path_to_static_attributes_directory} not found")
        return None

    if gauge_id_prefix is None and list_of_gauge_ids_to_process:
        first_id = list_of_gauge_ids_to_process[0]
        if "_" in first_id:
            gauge_id_prefix = first_id.split("_")[0]
            print(f"Extracted gauge ID prefix: {gauge_id_prefix}")
        else:
            print("Could not extract gauge ID prefix from basin IDs")
            return None

    if gauge_id_prefix is None:
        parquet_files = list(
            path_to_static_attributes_directory.glob("attributes_*.parquet")
        )
        if parquet_files:
            filename = parquet_files[0].stem
            parts = filename.split("_")
            if len(parts) >= 3:
                gauge_id_prefix = parts[2]
                print(f"Inferred gauge ID prefix: {gauge_id_prefix}")

    basin_id_set = (
        set(list_of_gauge_ids_to_process) if list_of_gauge_ids_to_process else None
    )
    dfs = []

    def load_attribute_file(file_type: str) -> Optional[pd.DataFrame]:
        filename = f"attributes_{file_type}_{gauge_id_prefix}.parquet"
        file_path = path_to_static_attributes_directory / filename

        if not file_path.exists():
            print(f"Attribute file {filename} not found")
            return None

        try:
            print(f"Loading {file_type} attributes from {file_path}")
            df = pd.read_parquet(file_path, engine="pyarrow")
            if "gauge_id" in df.columns:
                df["gauge_id"] = df["gauge_id"].astype(str)
            if basin_id_set:
                df = df[df["gauge_id"].isin(basin_id_set)]
            df.set_index("gauge_id", inplace=True)
            return df
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            return None

    caravan_df = load_attribute_file("caravan")
    if caravan_df is not None:
        dfs.append(caravan_df)
    hydroatlas_df = load_attribute_file("hydroatlas")
    if hydroatlas_df is not None:
        dfs.append(hydroatlas_df)
    other_df = load_attribute_file("other")
    if other_df is not None:
        dfs.append(other_df)

    if dfs:
        print(f"Merging {len(dfs)} attribute DataFrames")
        merged_df = pd.concat(dfs, axis=1, join="outer").reset_index()
        print(f"Loaded static attributes for {len(merged_df)} basins")
        return merged_df
    else:
        print("No static attribute files found or loaded")
        return None


def run_hydro_processor(
    path_to_time_series_directory: Union[str, Path],
    path_to_preprocessing_output_directory: Union[str, Path],
    required_columns: List[str],
    preprocessing_config: Optional[Dict[str, Dict[str, Any]]] = None,
    path_to_static_attributes_directory: Optional[str] = None,
    min_train_years: float = 5.0,
    max_imputation_gap_size: int = 5,
    group_identifier: str = "gauge_id",
    train_prop: float = 0.5,
    val_prop: float = 0.25,
    test_prop: float = 0.25,
    processes: int = 6,
    list_of_gauge_ids_to_process: Optional[List[str]] = None,
) -> ProcessingResult:
    """
    Main function to run the hydrological data processor with pipeline fitting.

    Args:
        path_to_time_series_directory: Directory containing parquet files
        path_to_preprocessing_output_directory: Directory for processed parquet files and reports
        required_columns: List of required data columns for quality checking
        preprocessing_config: Configuration for data preprocessing pipelines
        path_to_static_attributes_directory: Optional directory containing static attribute files
        min_train_years: Minimum required years for training
        max_imputation_gap_size: Maximum gap length to impute with interpolation
        group_identifier: Column name identifying the basin
        train_prop: Proportion of data for training
        val_prop: Proportion of data for validation
        test_prop: Proportion of data for testing
        processes: Number of parallel processes to use
        list_of_gauge_ids_to_process: Optional list of basin (gauge) IDs to process. If None, process all found.
    Returns:
        ProcessingResult containing quality report, fitted pipelines, and processed directories
    """
    config = Config(
        required_columns=required_columns,
        preprocessing_config=preprocessing_config,
        min_train_years=min_train_years,
        max_imputation_gap_size=max_imputation_gap_size,
        group_identifier=group_identifier,
        train_prop=train_prop,
        val_prop=val_prop,
        test_prop=test_prop,
    )

    path_to_preprocessing_output_directory_path = Path(
        path_to_preprocessing_output_directory
    )
    path_to_preprocessing_output_directory_path.mkdir(parents=True, exist_ok=True)

    processed_dir = path_to_preprocessing_output_directory_path / "processed_data"
    processed_path_to_static_attributes_directory = (
        path_to_preprocessing_output_directory_path / "processed_static_data"
    )
    reports_dir = path_to_preprocessing_output_directory_path / "quality_reports"
    processed_dir.mkdir(exist_ok=True)
    processed_path_to_static_attributes_directory.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    path_to_time_series_directory_path = Path(path_to_time_series_directory)
    basin_files = list(path_to_time_series_directory_path.glob("*.parquet"))
    if list_of_gauge_ids_to_process:
        basin_files = [f for f in basin_files if f.stem in list_of_gauge_ids_to_process]
    print(f"Found {len(basin_files)} basin files")

    if not basin_files:
        print("No parquet files found in input directory")
        return None

    list_of_gauge_ids_to_process = [basin_file.stem for basin_file in basin_files]

    static_df = None
    if path_to_static_attributes_directory:
        static_path = Path(path_to_static_attributes_directory)
        gauge_id_prefix = (
            list_of_gauge_ids_to_process[0].split("_")[0]
            if "_" in list_of_gauge_ids_to_process[0]
            else None
        )
        static_df = load_static_attributes(
            static_path, gauge_id_prefix, list_of_gauge_ids_to_process
        )

    fitted_pipelines = {}

    if preprocessing_config:
        print("Fitting preprocessing pipelines on all basins...")
        try:
            sample_data = load_all_basins(path_to_time_series_directory_path, config)

            # DEBUG
            print(f"Found following uniquwe columns in sample data: {sample_data.columns.tolist()}")

            print(f"Loaded {len(basin_files)} basins for pipeline fitting")
            train_df, val_df, test_df = split_data(sample_data, config)
            print(
                f"Split data into train ({len(train_df)} rows), val ({len(val_df)} rows), test ({len(test_df)} rows)"
            )
            fitted_pipelines = fit_pipelines(train_df, sample_data, static_df, config)
            print(f"Fitted {len(fitted_pipelines)} pipelines")
        except Exception as e:
            print(f"Error during pipeline fitting: {str(e)}")

    quality_report = process_basins_parallel(
        basin_files, config, processed_dir, reports_dir, fitted_pipelines, processes
    )

    # Process static attributes using the extended apply_transformations function
    if static_df is not None and "static" in fitted_pipelines:
        print("Processing static attributes...")
        # Only keep static attributes for retained basins
        retained_basins = [
            basin_id
            for basin_id in quality_report["basins"]
            if basin_id not in quality_report["excluded_basins"]
        ]
        filtered_static_df = static_df[static_df["gauge_id"].isin(retained_basins)]
        transformed_static = apply_transformations(
            filtered_static_df, config, fitted_pipelines, static_data=True
        )
        output_file = (
            processed_path_to_static_attributes_directory / "static_attributes.parquet"
        )
        transformed_static.to_parquet(output_file)
        print(
            f"Saved transformed static attributes for {len(transformed_static)} basins to {output_file}"
        )

    with open(
        path_to_preprocessing_output_directory_path / "quality_summary.json", "w"
    ) as f:
        json.dump(quality_report, f, indent=2, default=str)

    pipeline_info = {
        name: f"{type(pipeline).__name__}"
        for name, pipeline in fitted_pipelines.items()
    }
    with open(
        path_to_preprocessing_output_directory_path / "pipeline_info.json", "w"
    ) as f:
        json.dump(pipeline_info, f, indent=2)

    print(
        f"Processing complete. {quality_report['retained_basins']} basins retained out of {quality_report['original_basins']}."
    )

    if quality_report["excluded_basins"]:
        print(
            f"{len(quality_report['excluded_basins'])} basins excluded due to quality issues."
        )

    return {
        "quality_report": quality_report,
        "fitted_pipelines": fitted_pipelines,
        "processed_time_series_dir": processed_dir,
        "processed_static_attributes_dir": processed_path_to_static_attributes_directory
        if static_df is not None
        else None,
    }
