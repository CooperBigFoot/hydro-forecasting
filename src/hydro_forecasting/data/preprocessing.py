import argparse
import json
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict
import multiprocessing as mp
from tqdm import tqdm


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


class Config:
    def __init__(
        self,
        required_columns: List[str],
        min_train_years: float = 5.0,
        max_imputation_gap_size: int = 5,
        group_identifier: str = "gauge_id",
        train_prop: float = 0.6,
        val_prop: float = 0.2,
        test_prop: float = 0.2,
    ):
        self.required_columns = required_columns
        self.min_train_years = min_train_years
        self.max_imputation_gap_size = max_imputation_gap_size
        self.group_identifier = group_identifier
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.test_prop = test_prop


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process hydrological time series data with DuckDB"
    )

    # Required arguments
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for processed parquet files and reports",
    )
    parser.add_argument(
        "--required-columns",
        type=str,
        required=True,
        nargs="+",
        help="List of required data columns for quality checking",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--min-train-years",
        type=float,
        default=5.0,
        help="Minimum required years for training",
    )
    parser.add_argument(
        "--max-imputation-gap-size",
        type=int,
        default=5,
        help="Maximum gap length to impute with interpolation",
    )
    parser.add_argument(
        "--group-identifier",
        type=str,
        default="gauge_id",
        help="Column name identifying the basin",
    )
    parser.add_argument(
        "--train-prop", type=float, default=0.6, help="Proportion of data for training"
    )
    parser.add_argument(
        "--val-prop", type=float, default=0.2, help="Proportion of data for validation"
    )
    parser.add_argument(
        "--test-prop", type=float, default=0.2, help="Proportion of data for testing"
    )
    parser.add_argument(
        "--processes", type=int, default=4, help="Number of parallel processes to use"
    )

    args = parser.parse_args()

    # Create config object
    config = Config(
        required_columns=args.required_columns,
        min_train_years=args.min_train_years,
        max_imputation_gap_size=args.max_imputation_gap_size,
        group_identifier=args.group_identifier,
        train_prop=args.train_prop,
        val_prop=args.val_prop,
        test_prop=args.test_prop,
    )

    return args, config


def find_gaps(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find start and end indices of gaps in time series data.

    Args:
        series: Input time series

    Returns:
        Tuple containing arrays with gap start and end indices
    """
    # Find missing value runs
    is_missing = series.isna()

    # Edge case: no gaps
    if not is_missing.any():
        return np.array([]), np.array([])

    # Get indices of all NaN values
    nan_indices = np.where(is_missing)[0]

    if len(nan_indices) == 0:
        return np.array([]), np.array([])

    # Find discontinuities in the sequence of indices
    # which indicate separate gap regions
    gap_boundaries = np.where(np.diff(nan_indices) > 1)[0]

    # Create gap start indices
    gap_starts = np.array([nan_indices[0]])
    if len(gap_boundaries) > 0:
        # Add starts of new gaps (after each discontinuity)
        gap_starts = np.append(gap_starts, nan_indices[gap_boundaries + 1])

    # Create gap end indices (exclusive, so add 1)
    gap_ends = np.array([])
    if len(gap_boundaries) > 0:
        # Add ends of gaps before new ones start
        gap_ends = np.append(gap_ends, nan_indices[gap_boundaries] + 1)
    # Add the end of the last gap
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
    # Create a copy to avoid modifying the input
    imputed_df = df.copy()

    for column in columns:
        # Current column data
        series = imputed_df[column]
        is_nan = series.isna()

        if not is_nan.any():
            basin_report["imputation_info"][column] = {
                "short_gaps_count": 0,
                "imputed_values_count": 0,
            }
            continue

        # Find gaps
        gap_starts, gap_ends = find_gaps(series)

        # Track which gaps are short vs long
        short_gaps = []
        short_gap_indices = []

        for start_idx, end_idx in zip(gap_starts, gap_ends):
            # Calculate gap length
            gap_length = end_idx - start_idx

            if gap_length <= max_imputation_gap_size:
                short_gaps.append((start_idx, end_idx))
                # Add all indices in this gap
                short_gap_indices.extend(range(int(start_idx), int(end_idx)))

        # Create sorted series without NaNs for interpolation reference
        clean_series = series.dropna()

        if not clean_series.empty and short_gap_indices:
            # Apply interpolation to short gaps only
            temp_series = series.copy()

            # Create a mask for values we want to interpolate
            interpolate_mask = pd.Series(False, index=temp_series.index)
            for idx in short_gap_indices:
                if idx < len(interpolate_mask):
                    interpolate_mask.iloc[idx] = True

            # Apply interpolation method='linear' only where our mask is True
            if interpolate_mask.any():
                temp_series_interp = temp_series.interpolate(method="linear")
                temp_series.loc[interpolate_mask] = temp_series_interp.loc[
                    interpolate_mask
                ]

                # Update the original DataFrame with our interpolated values
                imputed_df[column] = temp_series

        # Record imputation statistics
        basin_report["imputation_info"][column] = {
            "short_gaps_count": len(short_gaps),
            "imputed_values_count": len(short_gap_indices),
        }

    basin_report["processing_steps"].append("Applied imputation to short gaps")

    return imputed_df, basin_report


def process_basin(
    basin_file: Path, config: Config, output_dir: Path, reports_dir: Path
) -> Tuple[bool, Optional[str], Optional[BasinQualityReport]]:
    """
    Process a single basin file using DuckDB and pandas.

    Args:
        basin_file: Path to the basin parquet file
        config: Configuration parameters
        output_dir: Directory to save processed data
        reports_dir: Directory to save quality reports

    Returns:
        Tuple containing:
        - Success flag (True if basin passed quality checks)
        - Error message (if any)
        - Quality report for the basin
    """
    basin_id = basin_file.stem  # Get gauge_id from filename

    # Initialize quality report for this basin
    basin_report: BasinQualityReport = {
        "valid_period": {},
        "processing_steps": [],
        "imputation_info": {},
    }

    try:
        # Step 1: Load basin data using DuckDB
        con = duckdb.connect(database=":memory:")
        con.execute(f"CREATE TABLE basin AS SELECT * FROM read_parquet('{basin_file}')")

        # Make sure required columns exist
        columns = con.execute("PRAGMA table_info(basin)").fetchall()
        column_names = [col[1] for col in columns]

        missing_cols = [
            col for col in config.required_columns if col not in column_names
        ]
        if missing_cols or "date" not in column_names:
            error_msg = f"Missing required columns: {missing_cols + (['date'] if 'date' not in column_names else [])}"
            return False, error_msg, basin_report

        # Step 2: Find valid periods for each required column (first/last non-NULL value)
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

        # Find overall valid period (overlap of all required columns)
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

        # Convert string dates back to datetime for comparison
        start_dates = [pd.to_datetime(date) for date in valid_starts]
        end_dates = [pd.to_datetime(date) for date in valid_ends]

        overall_start = max(start_dates)
        overall_end = min(end_dates)

        # Check if period meets minimum requirements
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

        # Step 3: Filter to valid period with SQL
        con.execute(f"""
            CREATE TABLE filtered_basin AS
            SELECT * FROM basin
            WHERE date >= '{overall_start}' AND date <= '{overall_end}'
            ORDER BY date
        """)

        basin_report["processing_steps"].append("Filtered to valid period")

        # Step 4: Transfer to pandas for imputation
        df = con.execute("SELECT * FROM filtered_basin").df()
        basin_report["processing_steps"].append("Transferred to pandas for imputation")

        # Initialize imputation info
        for column in config.required_columns:
            basin_report["imputation_info"][column] = {
                "short_gaps_count": 0,
                "imputed_values_count": 0,
            }

        # Apply imputation to the DataFrame
        imputed_df, basin_report = impute_short_gaps(
            df,
            config.required_columns,
            config.max_imputation_gap_size,
            basin_report,
        )

        # Save processed basin data
        output_file = output_dir / f"{basin_id}.parquet"
        imputed_df.to_parquet(output_file)

        # Save basin quality report
        report_file = reports_dir / f"{basin_id}_report.json"
        with open(report_file, "w") as f:
            json.dump(basin_report, f, indent=2, default=str)

        basin_report["processing_steps"].append("Basin processed successfully")
        return True, None, basin_report

    except Exception as e:
        error_msg = f"Error processing basin {basin_id}: {str(e)}"
        return False, error_msg, basin_report
    finally:
        # Close DuckDB connection
        if "con" in locals():
            con.close()


def process_basin_worker(args: Tuple[Path, Config, Path, Path]) -> Tuple[str, bool, Optional[str], Optional[BasinQualityReport]]:
    """
    Worker function for parallel basin processing.
    
    Args:
        args: Tuple containing (basin_file, config, output_dir, reports_dir)
        
    Returns:
        Tuple containing basin_id, success flag, error message, and basin report
    """
    basin_file, config, output_dir, reports_dir = args
    success, error_msg, basin_report = process_basin(
        basin_file, config, output_dir, reports_dir
    )
    basin_id = basin_file.stem
    return basin_id, success, error_msg, basin_report


def process_basins_parallel(
    basin_files: List[Path],
    config: Config,
    output_dir: Path,
    reports_dir: Path,
    num_processes: int,
) -> QualityReport:
    """
    Process multiple basin files in parallel.

    Args:
        basin_files: List of paths to basin parquet files
        config: Configuration parameters
        output_dir: Directory to save processed data
        reports_dir: Directory to save quality reports
        num_processes: Number of parallel processes to use

    Returns:
        Overall quality report
    """
    # Initialize overall quality report
    quality_report: QualityReport = {
        "original_basins": len(basin_files),
        "retained_basins": 0,
        "excluded_basins": {},
        "basins": {},
        "split_method": "proportional",
    }

    # Create arguments list for the worker function
    args_list = [(basin_file, config, output_dir, reports_dir) for basin_file in basin_files]

    # Process basins in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_basin_worker, args_list),
                total=len(basin_files),
                desc="Processing basins",
            )
        )

    # Combine results into overall quality report
    for basin_id, success, error_msg, basin_report in results:
        quality_report["basins"][basin_id] = basin_report

        if not success:
            quality_report["excluded_basins"][basin_id] = error_msg
        else:
            quality_report["retained_basins"] += 1

    return quality_report


def main():
    args, config = parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for processed data and reports
    processed_dir = output_dir / "processed_data"
    reports_dir = output_dir / "quality_reports"
    processed_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    # Scan input directory for parquet files
    input_dir = Path(args.input_dir)
    basin_files = list(input_dir.glob("*.parquet"))
    print(f"Found {len(basin_files)} basin files")

    if not basin_files:
        print("No parquet files found in input directory")
        return

    # Process basins in parallel
    quality_report = process_basins_parallel(
        basin_files, config, processed_dir, reports_dir, args.processes
    )

    # Save overall quality report
    with open(output_dir / "quality_summary.json", "w") as f:
        json.dump(
            quality_report, f, indent=2, default=str
        )  # default=str handles datetime objects

    print(
        f"Processing complete. {quality_report['retained_basins']} basins retained out of {quality_report['original_basins']}."
    )

    if quality_report["excluded_basins"]:
        print(
            f"{len(quality_report['excluded_basins'])} basins excluded due to quality issues."
        )


if __name__ == "__main__":
    main()
