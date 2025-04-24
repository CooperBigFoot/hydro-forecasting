import numpy as np
import pandas as pd
from pathlib import Path
from returns.result import Result, Success, Failure
from tqdm import tqdm  # Add tqdm for progress bar
from .preprocessing import split_data, Config

BATCH_SIZE = 100  # Maximum number of basins to process per batch

# Using a default config without immediate validation
# We'll validate within the functions when they're actually called
SPLIT_CONFIG = Config(
    required_columns=[""],  # Can stay empty
    preprocessing_config={},  # Can stay empty
    train_prop=0.6,  # Default values that will be overridden
    val_prop=0.2,
    test_prop=0.2,
)


def load_batch_parquet(
    batch_gauge_ids: list[str], time_series_base_dir: Path
) -> Result[pd.DataFrame, str]:
    """
    Loads .parquet files for a batch of gauge_ids.

    Args:
        batch_gauge_ids: List of gauge IDs for this batch.
        time_series_base_dir: Directory containing the parquet files.

    Returns:
        Result containing the concatenated DataFrame for the batch, or Failure with error message.

    Raises:
        Does not raise; returns Failure on critical errors.

    Example:
        >>> load_batch_parquet(['USA_001', 'USA_002'], Path('/data'))
    """
    data: list[pd.DataFrame] = []
    missing: list[str] = []
    for gauge_id in batch_gauge_ids:
        file_path = time_series_base_dir / f"{gauge_id}.parquet"
        if not file_path.exists():
            missing.append(gauge_id)
            continue
        try:
            df = pd.read_parquet(file_path)
            df["gauge_id"] = gauge_id
            data.append(df)
        except Exception as e:
            missing.append(gauge_id)
            continue
    if not data:
        return Failure(
            f"No files loaded for batch: {batch_gauge_ids}. Missing: {missing}"
        )
    batch_df = pd.concat(data, ignore_index=True)
    return Success(batch_df)


def get_split_boundaries(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, gauge_ids: list[str]
) -> dict[str, dict[str, pd.Timestamp | None]]:
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


def find_valid_sequences(
    basin_data: pd.DataFrame,
    input_length: int,
    output_length: int,
    cols_to_check: list[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find valid sequence starting positions in the basin data.

    Args:
        basin_data: DataFrame containing basin time series data.
        input_length: Length of input sequence.
        output_length: Length of output sequence.
        cols_to_check: List of column names to check for NaN values in both input and output windows.
            All columns in this list must be present and non-NaN for a sequence to be considered valid.

    Returns:
        Tuple of (valid_positions, dates) arrays.
    """

    total_seq_length = input_length + output_length

    if len(basin_data) < total_seq_length:
        return np.array([]), np.array([])

    # Extract needed data as arrays
    basin_values = basin_data[cols_to_check].to_numpy()
    dates = basin_data["date"].to_numpy()

    # Combined valid mask: 1 if all cols not NaN, 0 otherwise
    combined_valid = (~np.isnan(basin_values).any(axis=1)).astype(int)

    # Input window must be valid for all features
    input_conv = np.convolve(
        combined_valid, np.ones(input_length, dtype=int), mode="valid"
    )
    input_valid = input_conv == input_length

    # Output window must also be valid for all features
    output_conv = np.convolve(
        combined_valid, np.ones(output_length, dtype=int), mode="valid"
    )
    output_valid = output_conv == output_length

    # Align the output validity check with input positions
    padded = np.pad(output_valid, (0, input_length), constant_values=False)
    output_valid_shifted = padded[input_length : input_length + len(input_valid)]

    # Sequence is valid only if both input and output windows are valid
    valid_mask = input_valid & output_valid_shifted
    valid_positions = np.where(valid_mask)[0]

    return valid_positions, dates


def determine_stage(input_end_date, boundaries: dict[str, pd.Timestamp | None]) -> str:
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
    static_file_path: Path,
    input_length: int,
    output_length: int,
    train_prop: float = None,
    val_prop: float = None,
    test_prop: float = None,
    cols_to_check: list[str] = None,
) -> list[dict]:
    """
    Create index entries for valid sequences, identifying which stage (train/val/test) each sequence belongs to.
    Processes basins in batches for memory efficiency.

    Args:
        gauge_ids: List of gauge IDs to process.
        time_series_base_dir: Base directory containing parquet files.
        static_file_path: Path to the unified processed static attributes file.
        input_length: Length of input sequence.
        output_length: Length of forecast horizon.
        train_prop: Proportion of data for training (optional).
        val_prop: Proportion of data for validation (optional).
        test_prop: Proportion of data for testing (optional).
        cols_to_check: List of column names to check for NaN values in sequence validity checks.

    Returns:
        List of index entries with stage identification.

    Raises:
        ValueError: If split proportions are invalid.

    Example:
        >>> create_index_entries(['USA_001', 'USA_002'], Path('/data'), Path('/static.parquet'), 30, 7)
    """
    # Validate split proportions once
    local_config = Config(
        required_columns=[],  # Not used here
        preprocessing_config={},
        train_prop=train_prop if train_prop is not None else 0.6,
        val_prop=val_prop if val_prop is not None else 0.2,
        test_prop=test_prop if test_prop is not None else 0.2,
    )
    if (
        local_config.train_prop <= 0
        or local_config.val_prop <= 0
        or local_config.test_prop <= 0
    ):
        raise ValueError(
            "The train, val, and test proportions must be positive values."
        )
    sum_props = local_config.train_prop + local_config.val_prop + local_config.test_prop
    if abs(sum_props - 1.0) > 1e-10:
        raise ValueError(
            f"The sum of train, val, and test proportions must be 1.0, got {sum_props:.10f}"
        )

    all_index_entries: list[dict] = []
    total_seq_length = input_length + output_length

    num_batches = (len(gauge_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_start in tqdm(
        range(0, len(gauge_ids), BATCH_SIZE),
        total=num_batches,
        desc="Creating index entries",
    ):
        batch_gauge_ids = gauge_ids[batch_start : batch_start + BATCH_SIZE]
        batch_result = load_batch_parquet(batch_gauge_ids, time_series_base_dir)
        if isinstance(batch_result, Failure):
            # Optionally log batch_result.failure()
            continue
        batch_data = batch_result.unwrap()

        train, val, test = split_data(df=batch_data, config=local_config)
        batch_boundaries = get_split_boundaries(train, val, test, batch_gauge_ids)

        for gauge_id in batch_gauge_ids:
            ts_file_path = time_series_base_dir / f"{gauge_id}.parquet"
            basin_data = batch_data[batch_data["gauge_id"] == gauge_id]
            if basin_data.empty:
                continue
            gauge_bounds = batch_boundaries.get(
                gauge_id, {"val_start": None, "test_start": None}
            )
            valid_positions, dates = find_valid_sequences(
                basin_data, input_length, output_length, cols_to_check=cols_to_check
            )
            if valid_positions.size == 0:
                continue
            for idx in valid_positions:
                if idx + total_seq_length > len(basin_data):
                    continue
                input_end_date = dates[idx + input_length - 1]
                stage = determine_stage(input_end_date, gauge_bounds)
                entry = {
                    "file_path": str(ts_file_path),
                    "gauge_id": gauge_id,
                    "start_idx": idx,
                    "end_idx": idx + total_seq_length,
                    "input_end_date": input_end_date,
                    "valid_sequence": True,
                    "stage": stage,
                    "static_file_path": str(static_file_path),
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
