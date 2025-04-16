import numpy as np
import pandas as pd
from pathlib import Path
from .preprocessing import split_data, Config

# Config is only needed for splitting data. Not the cleanest approach but such is life
SPLIT_CONFIG = Config(
    required_columns=[""],  # Can stay empty
    preprocessing_config={},  # Can stay empty
    train_prop=-1,
    val_prop=-1,
    test_prop=-1,
)

if (
    SPLIT_CONFIG.train_prop == -1
    or SPLIT_CONFIG.val_prop == -1
    or SPLIT_CONFIG.test_prop == -1
):
    raise ValueError(
        "The train, val, and test proportions must be set in the SPLIT_CONFIG."
    )


def load_gauge_parquet(
    gauge_ids: list[str], time_series_base_dir: Path
) -> pd.DataFrame:
    """
    Loads the .parquet file for a given list of gauge_ids.

    Args:
        gauge_ids (list[str]): Gauge IDs with the 'USA_' prefix.
        time_series_base_dir (Path): Path to the directory containing the parquet files.

    Returns:
        pd.DataFrame: Combined data from the corresponding parquet files.
    """
    data = []

    for gauge_id in gauge_ids:
        file_path = time_series_base_dir / f"{gauge_id}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(
                f"No parquet file found for gauge ID {gauge_id} at {file_path}"
            )
        try:
            df = pd.read_parquet(file_path)
            df["gauge_id"] = gauge_id  # Assign here
            data.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    combined_data = pd.concat(data, ignore_index=True)
    return combined_data


def get_split_boundaries(train, val, test, gauge_ids):
    """
    Determine the date boundaries between train/val/test splits for each gauge ID.

    Args:
        train: Training DataFrame
        val: Validation DataFrame
        test: Test DataFrame
        gauge_ids: List of gauge IDs to process

    Returns:
        Dictionary mapping gauge_ids to their split boundary dates
    """
    split_boundaries = {}

    for gauge_id in gauge_ids:
        # Get min dates for val and test splits for this gauge
        gauge_val = val[val["gauge_id"] == gauge_id]
        gauge_test = test[test["gauge_id"] == gauge_id]

        val_start = gauge_val["date"].min() if not gauge_val.empty else None
        test_start = gauge_test["date"].min() if not gauge_test.empty else None

        split_boundaries[gauge_id] = {"val_start": val_start, "test_start": test_start}

    return split_boundaries


def find_valid_sequences(basin_data, input_length, output_length, cols_to_check=None):
    """
    Find valid sequence starting positions in the basin data.

    Args:
        basin_data: DataFrame containing basin time series data
        input_length: Length of input sequence
        output_length: Length of output sequence
        cols_to_check: Columns to check for NaN values

    Returns:
        Tuple of (valid_positions, dates) arrays
    """
    if cols_to_check is None:
        cols_to_check = ["streamflow", "total_precipitation_sum"]

    total_seq_length = input_length + output_length

    if len(basin_data) < total_seq_length:
        return np.array([]), np.array([])

    # Extract needed data as arrays
    basin_values = basin_data[cols_to_check].to_numpy()
    dates = basin_data["date"].to_numpy()

    # Combined valid mask: 1 if all cols not NaN, 0 otherwise
    combined_valid = (~np.isnan(basin_values).any(axis=1)).astype(int)

    # Convolve to find valid input sequences
    input_conv = np.convolve(
        combined_valid, np.ones(input_length, dtype=int), mode="valid"
    )
    input_valid = input_conv == input_length

    # Convolve for output sequences, shifted by input_length
    output_conv = np.convolve(
        combined_valid, np.ones(output_length, dtype=int), mode="valid"
    )
    output_valid = output_conv == output_length
    output_valid_shifted = np.pad(
        output_valid, (input_length, 0), constant_values=False
    )[: len(input_valid)]

    # Find valid sequence starts
    valid_mask = input_valid & output_valid_shifted
    valid_positions = np.where(valid_mask)[0]

    return valid_positions, dates


def determine_stage(input_end_date, boundaries):
    """
    Determine which stage (train/val/test) a sequence belongs to based on its end date.

    Args:
        input_end_date: End date of the input sequence
        boundaries: Dictionary with val_start and test_start dates

    Returns:
        String: 'train', 'val', or 'test'
    """
    val_start = boundaries["val_start"]
    test_start = boundaries["test_start"]

    if test_start is not None and input_end_date >= test_start:
        return "test"
    elif val_start is not None and input_end_date >= val_start:
        return "val"
    else:
        return "train"


def create_index_entries(
    gauge_ids: list[str],
    time_series_base_dir: Path,
    input_length: int,
    output_length: int,
):
    """
    Create index entries for valid sequences, identifying which stage (train/val/test) each sequence belongs to.

    Args:
        gauge_ids: List of gauge IDs to process
        time_series_base_dir: Base directory containing parquet files
        static_file_path: Path to static attributes file
        input_length: Length of input sequence
        output_length: Length of forecast horizon

    Returns:
        List of index entries with stage identification
    """
    valid_data = load_gauge_parquet(gauge_ids, time_series_base_dir)

    train, val, test = split_data(df=valid_data, config=SPLIT_CONFIG)

    # Get split boundaries for each gauge
    split_boundaries = get_split_boundaries(train, val, test, gauge_ids)

    all_index_entries = []
    total_seq_length = input_length + output_length

    # Process each basin
    for gauge_id, basin_data in valid_data.groupby("gauge_id"):
        # Create actual file path for this gauge
        ts_file_path = time_series_base_dir / f"{gauge_id}.parquet"

        # Get split boundaries for this gauge
        gauge_bounds = split_boundaries.get(
            gauge_id, {"val_start": None, "test_start": None}
        )

        # Find valid sequences in this basin's data
        valid_positions, dates = find_valid_sequences(
            basin_data, input_length, output_length
        )

        # Create index entries with stage identification
        for idx in valid_positions:
            if idx + total_seq_length > len(basin_data):
                continue

            # Get the input_end_date for this sequence
            input_end_date = dates[idx + input_length - 1]

            # Determine stage based on input_end_date
            stage = determine_stage(input_end_date, gauge_bounds)

            # Create entry with stage information
            entry = {
                "file_path": str(ts_file_path),
                "gauge_id": gauge_id,
                "start_idx": idx,
                "end_idx": idx + total_seq_length,
                "input_end_date": input_end_date,
                "valid_sequence": True,
                "stage": stage,
            }

            all_index_entries.append(entry)

    return all_index_entries


def split_index_entries_by_stage(
    index_entries: list[dict],
) -> dict[str, list[dict]]:
    """
    Split index entries into train, val, and test sets based on their stage.

    Args:
        index_entries: List of index entries with stage information

    Returns:
        Dictionary with keys 'train', 'val', and 'test' mapping to lists of index entries
    """
    split_entries = {"train": [], "val": [], "test": []}

    for entry in index_entries:
        stage = entry["stage"]
        if stage in split_entries:
            split_entries[stage].append(entry)

    return split_entries
