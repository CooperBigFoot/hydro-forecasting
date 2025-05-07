import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np
import datetime
from pathlib import Path
from typing import Optional, Union, List, Tuple
from returns.result import Result, Success, Failure, safe
import logging
import time
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@safe
def _calculate_valid_sequences(
    basin_data: pl.DataFrame,
    input_length: int,
    output_length: int,
    total_length: int,
    value_columns: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function containing the core numpy logic, wrapped by @safe.

    Args:
        basin_data: Polars DataFrame for a single basin.
        input_length: Length of the input sequence.
        output_length: Length of the output sequence.
        total_length: Total sequence length (input + output).
        value_columns: List of columns to check for NaNs.

    Returns:
        Tuple containing an array of valid start indices and an array of dates.
    """
    # Ensure all value_columns exist in basin_data before selecting
    existing_value_columns = [col for col in value_columns if col in basin_data.columns]
    if not existing_value_columns:
        # This case should ideally be caught before calling this function,
        # but as a safeguard:
        logger.warning(
            f"No value columns found in basin_data for NaN checking. Columns expected: {value_columns}, Columns present: {basin_data.columns}"
        )
        return np.array([], dtype=int), basin_data.select(
            "date"
        ).to_numpy().flatten() if "date" in basin_data.columns else np.array([])

    values = basin_data.select(existing_value_columns).to_numpy()
    dates = basin_data.select("date").to_numpy().flatten()

    # Create a mask where True indicates a valid (non-NaN) row across all value_columns
    valid_row_mask = ~np.isnan(values).any(axis=1)

    # Use convolution to efficiently check for consecutive valid rows
    # Input window check
    input_valid_sums = np.convolve(
        valid_row_mask, np.ones(input_length, dtype=int), mode="valid"
    )
    is_input_window_valid = input_valid_sums == input_length

    # Output window check
    output_valid_sums = np.convolve(
        valid_row_mask, np.ones(output_length, dtype=int), mode="valid"
    )
    is_output_window_valid = output_valid_sums == output_length

    # Align the validity checks for the combined sequence
    num_possible_starts = len(valid_row_mask) - total_length + 1
    if num_possible_starts <= 0:
        return np.array([], dtype=int), dates  # No possible sequences

    # Check if input window starting at `i` is valid
    input_ok = is_input_window_valid[:num_possible_starts]
    # Check if output window starting at `i + input_length` is valid
    output_ok = is_output_window_valid[
        input_length : input_length + num_possible_starts
    ]

    # A sequence starting at index `i` is valid if both input and output windows are valid
    valid_sequence_starts_mask = input_ok & output_ok
    positions = np.where(valid_sequence_starts_mask)[0]

    return positions, dates


def find_valid_sequences(
    basin_data: pl.DataFrame,
    input_length: int,
    output_length: int,
) -> Result[Tuple[np.ndarray, np.ndarray], str]:
    """
    Identify start indices of valid sequences within a time series DataFrame using ROP.

    A sequence is valid if both the input window (length `input_length`) and
    the subsequent output window (length `output_length`) contain no null (NaN)
    values in any column except the 'date' column.

    Args:
        basin_data: A Polars DataFrame containing time series data for a single
                    basin/group. It *should not* contain the group_identifier column
                    at this point, only 'date' and value columns.
        input_length: The required length of the input sequence.
        output_length: The required length of the output sequence.

    Returns:
        Success containing a tuple:
            - np.ndarray: An array of integer start indices for valid sequences.
            - np.ndarray: An array of dates corresponding to each row in basin_data.
        Failure containing an error message string if validation fails or calculation errors occur.
    """
    total_length = input_length + output_length

    initial_result: Result[pl.DataFrame, str] = Success(basin_data)

    return (
        initial_result.bind(
            lambda df: Success(df)
            if df.height >= total_length
            else Failure(
                f"Data height ({df.height}) is less than total sequence length ({total_length})"
            )
        )
        .bind(
            lambda df: Success(df)
            if "date" in df.columns
            else Failure("DataFrame missing 'date' column")
        )
        .bind(
            lambda df: Success((df, [col for col in df.columns if col != "date"]))
        )  # value_columns are all non-date columns
        .bind(
            lambda df_cols: Success(df_cols)
            if df_cols[1]  # Check if value_columns list is not empty
            else Failure("No value columns found for null checking (besides 'date')")
        )
        .bind(
            lambda df_cols: _calculate_valid_sequences(
                basin_data=df_cols[0],
                input_length=input_length,
                output_length=output_length,
                total_length=total_length,
                value_columns=df_cols[1],  # Pass the identified value columns
            ).alt(lambda exc: f"NumPy calculation error: {exc}")
        )
    )


class InMemoryChunkDataset(Dataset):
    """
    Dataset that loads a chunk of basin data into memory and serves samples.

    Args:
        basin_ids: List of basin identifiers for this chunk.
        processed_data_dir: Base directory containing processed Parquet files.
        stage: Data split stage ('train', 'val', 'test').
        static_data_cache: Preloaded dictionary mapping basin_id to static features array.
        input_length: Number of timesteps for model input.
        output_length: Number of timesteps for model output.
        target: Name of the target variable column.
        forcing_features: List of forcing feature column names.
        static_features: List of static feature column names.
        group_identifier: Column name identifying basins (e.g., 'gauge_id').
        is_autoregressive: If True, include past target in input features.
    """

    def __init__(
        self,
        basin_ids: list[str],
        processed_data_dir: Union[str, Path],
        stage: str,
        static_data_cache: dict[str, np.ndarray],
        input_length: int,
        output_length: int,
        target: str,
        forcing_features: list[str],
        static_features: list[str],
        group_identifier: str = "gauge_id",
        is_autoregressive: bool = False,
        load_engine: str = "polars",  # 'polars' or 'pyarrow'
    ):
        super().__init__()
        self.basin_ids = basin_ids
        self.processed_data_dir = Path(processed_data_dir)
        self.stage = stage
        self.static_data_cache = static_data_cache
        self.input_length = input_length
        self.output_length = output_length
        self.target = target
        self.forcing_features = sorted(forcing_features)
        self.static_features = sorted(static_features)
        self.group_identifier = group_identifier
        self.is_autoregressive = is_autoregressive
        self.load_engine = load_engine

        # Feature definitions
        if is_autoregressive:
            self.input_features = [target] + [
                f for f in self.forcing_features if f != target
            ]
        else:
            self.input_features = self.forcing_features
        self.future_features = self.forcing_features

        # Internal state
        self.chunk_data: Optional[pl.DataFrame] = None
        self.index_entries: list[
            tuple[str, int, int]
        ] = []  # (basin_id, start_row_in_original_basin_df, end_row_in_original_basin_df)
        self.basin_row_map: dict[
            str, tuple[int, int]
        ] = {}  # basin_id -> (start_row_in_chunk_data, num_rows)

        # Load data and precompute index
        self._load_and_index_chunk()

    def _load_and_index_chunk(self) -> None:
        """Loads Parquet files for the chunk and computes the sample index."""
        start_time = time.time()
        logger.info(
            f"Loading chunk for stage '{self.stage}' with {len(self.basin_ids)} basins..."
        )

        dfs_to_concat: list[pl.DataFrame] = []
        processed_count = 0
        skipped_count = 0

        stage_dir = self.processed_data_dir / self.stage
        required_ts_cols = ["date", self.target] + self.forcing_features
        required_ts_cols = sorted(list(set(required_ts_cols)))

        for basin_id in self.basin_ids:
            file_path = stage_dir / f"{basin_id}.parquet"
            if file_path.exists():
                try:
                    lf = (
                        pl.scan_parquet(str(file_path))
                        .select(required_ts_cols)  # Select only necessary TS columns
                        .sort("date")
                    )
                    df = lf.collect()

                    dfs_to_concat.append(
                        df.with_columns(pl.lit(basin_id).alias(self.group_identifier))
                    )
                    processed_count += 1
                except Exception as e:
                    logger.warning(f"Could not load or process file {file_path}: {e}")
                    skipped_count += 1
            else:
                logger.warning(f"File not found, skipping: {file_path}")
                skipped_count += 1

        if not dfs_to_concat:
            logger.error(f"No valid data loaded for chunk in stage '{self.stage}'.")
            self.chunk_data = pl.DataFrame()  # Ensure it's an empty DF
            self.index_entries = []
            return

        try:
            self.chunk_data = pl.concat(dfs_to_concat, how="vertical")
            logger.info(
                f"Chunk data loaded. Shape: {self.chunk_data.shape}. Memory usage: {self.chunk_data.estimated_size('mb'):.2f} MB"
            )
        except Exception as e:
            logger.error(f"Failed to concatenate dataframes for chunk: {e}")
            self.chunk_data = pl.DataFrame()  # Ensure it's an empty DF
            self.index_entries = []
            return

        del dfs_to_concat
        gc.collect()

        logger.info("Precomputing index for in-memory chunk...")
        current_absolute_start_row = 0
        for basin_id in (
            self.basin_ids
        ):  # Iterate using the original list of basin_ids for this chunk
            basin_df_with_id = self.chunk_data.filter(
                pl.col(self.group_identifier) == basin_id
            )

            if basin_df_with_id.height == 0:
                logger.warning(
                    f"No data found for basin {basin_id} in concatenated chunk, skipping indexing."
                )
                continue

            # Store the mapping from basin_id to its absolute start row and length in the concatenated chunk_data
            self.basin_row_map[basin_id] = (
                current_absolute_start_row,
                basin_df_with_id.height,
            )

            # For indexing, create a DataFrame without the group_identifier
            # This ensures find_valid_sequences works correctly as it expects only date + value columns
            if self.group_identifier in basin_df_with_id.columns:
                basin_df_for_indexing = basin_df_with_id.drop(self.group_identifier)
            else:
                basin_df_for_indexing = (
                    basin_df_with_id  # Should not happen if logic above is correct
                )

            find_result = find_valid_sequences(
                basin_df_for_indexing, self.input_length, self.output_length
            )

            if isinstance(find_result, Success):
                positions, _ = (
                    find_result.unwrap()
                )  # dates_from_find are from basin_df_for_indexing
                for start_idx_relative_to_basin in positions:
                    # start_idx is relative to the beginning of basin_df_for_indexing
                    self.index_entries.append(
                        (
                            basin_id,  # Store the basin_id
                            int(
                                start_idx_relative_to_basin
                            ),  # Store relative start index
                            int(
                                start_idx_relative_to_basin
                                + self.input_length
                                + self.output_length
                            ),  # Store relative end index
                        )
                    )
            else:
                logger.warning(
                    f"Could not find valid sequences for {basin_id}: {find_result.failure()}"
                )

            current_absolute_start_row += basin_df_with_id.height

        if self.stage == "train":
            np.random.shuffle(self.index_entries)

        end_time = time.time()
        logger.info(
            f"Chunk loading and indexing complete for stage '{self.stage}'. "
            f"Processed {processed_count}, Skipped {skipped_count}. "
            f"Found {len(self.index_entries)} valid samples. Time: {end_time - start_time:.2f}s"
        )

    def __len__(self) -> int:
        return len(self.index_entries)

    def __getitem__(
        self, idx: int
    ) -> dict[
        str, Union[torch.Tensor, str, int, None]
    ]:  # input_end_date is int or None
        if not (0 <= idx < len(self.index_entries)):
            raise IndexError(
                f"Index {idx} out of range for chunk size {len(self.index_entries)}"
            )

        basin_id, basin_start_row_relative, basin_end_row_relative = self.index_entries[
            idx
        ]

        chunk_basin_start_abs, basin_len = self.basin_row_map.get(
            basin_id, (None, None)
        )
        if chunk_basin_start_abs is None or basin_len is None:
            raise RuntimeError(
                f"Basin {basin_id} missing from basin_row_map or invalid length recorded."
            )

        start_abs = chunk_basin_start_abs + basin_start_row_relative
        duration = basin_end_row_relative - basin_start_row_relative
        end_abs = start_abs + duration

        if end_abs > chunk_basin_start_abs + basin_len:
            logger.error(
                f"Slicing error for basin {basin_id} at index {idx}: "
                f"Attempting to slice from absolute start {start_abs} for duration {duration} (ends at {end_abs}). "
                f"Basin data in chunk starts at {chunk_basin_start_abs} and has length {basin_len} "
                f"(ends at {chunk_basin_start_abs + basin_len}). "
            )
            raise IndexError(
                f"Calculated end index {end_abs} exceeds concatenated data length for basin {basin_id}."
            )

        try:
            sequence_df_with_id = self.chunk_data.slice(start_abs, duration)
            relevant_ts_cols = sorted(
                list(set(["date", self.target] + self.forcing_features))
            )
            sequence_df_numeric = sequence_df_with_id.select(relevant_ts_cols)

            if sequence_df_numeric.height != (self.input_length + self.output_length):
                logger.error(
                    f"Sliced sequence for index {idx}, basin {basin_id} has incorrect length: {sequence_df_numeric.height}. "
                    f"Expected: {self.input_length + self.output_length}."
                )
                raise ValueError(
                    f"Sliced sequence for index {idx}, basin {basin_id} has incorrect length."
                )
        except Exception as e:
            logger.error(
                f"Error during data slicing for index {idx}, basin {basin_id}: {e}"
            )
            raise RuntimeError(f"Failed to slice data for index {idx}") from e

        inp_df = sequence_df_numeric.slice(0, self.input_length)
        out_df = sequence_df_numeric.slice(self.input_length, self.output_length)

        try:
            X_np = inp_df.select(self.input_features).to_numpy().astype(np.float32)
            y_np = out_df.select(self.target).to_numpy().squeeze().astype(np.float32)
            future_np = (
                out_df.select(self.future_features).to_numpy().astype(np.float32)
            )
        except Exception as e:
            logger.error(
                f"Error converting sliced data to NumPy arrays for index {idx}, basin {basin_id}: {e}"
            )
            raise RuntimeError(
                f"Failed to convert data to NumPy for index {idx}"
            ) from e

        static_arr = self.static_data_cache.get(basin_id)
        if static_arr is None:
            static_arr = np.zeros(len(self.static_features), dtype=np.float32)
        elif static_arr.shape[0] != len(self.static_features):
            static_arr = np.zeros(len(self.static_features), dtype=np.float32)
        static_arr = static_arr.astype(np.float32)

        try:
            X = torch.from_numpy(X_np)
            y = torch.from_numpy(y_np)
            future = torch.from_numpy(future_np)
            static = torch.from_numpy(static_arr)
        except Exception as e:
            logger.error(
                f"Error converting NumPy arrays to tensors for index {idx}, basin {basin_id}: {e}"
            )
            raise RuntimeError(
                f"Failed to convert data to Tensors for index {idx}"
            ) from e

        input_end_ts_ms: Optional[int] = None  # Initialize as int or None
        try:
            if "date" in inp_df.columns and inp_df.height > 0:
                date_scalar = inp_df.get_column("date").item(
                    -1
                )  # This is often a Python datetime.datetime

                py_datetime_obj: Optional[datetime.datetime] = None
                if isinstance(date_scalar, datetime.datetime):
                    py_datetime_obj = date_scalar
                elif hasattr(
                    date_scalar, "to_pydatetime"
                ):  # For some Polars specific scalar date/datetime types
                    py_datetime_obj = date_scalar.to_pydatetime()
                elif date_scalar is None:
                    pass  # py_datetime_obj remains None
                else:
                    logger.warning(
                        f"Unexpected type for date scalar: {type(date_scalar)} for index {idx}, basin {basin_id}. "
                        f"Value: {date_scalar}. Cannot convert to Python datetime."
                    )

                if py_datetime_obj:
                    # Convert to UTC if timezone aware, then get timestamp
                    if (
                        py_datetime_obj.tzinfo is not None
                        and py_datetime_obj.tzinfo.utcoffset(py_datetime_obj)
                        is not None
                    ):
                        py_datetime_obj_utc = py_datetime_obj.astimezone(
                            datetime.timezone.utc
                        )
                    else:  # Assume naive is UTC or handle as per application logic
                        py_datetime_obj_utc = py_datetime_obj
                    input_end_ts_ms = int(py_datetime_obj_utc.timestamp() * 1000)
                # If py_datetime_obj is None, input_end_ts_ms remains None
            else:
                logger.warning(
                    f"Date column missing or inp_df is empty for index {idx}, basin {basin_id}."
                )

        except Exception as e:
            logger.error(
                f"Error getting input end date for index {idx}, basin {basin_id}. Error: {e}"
            )
            # input_end_ts_ms remains None or could be set to a default like 0 if preferred
            # input_end_ts_ms = 0 # Example default if None is not desired

        return {
            "X": X,
            "y": y,
            "static": static,
            "future": future,
            self.group_identifier: basin_id,
            "input_end_date": input_end_ts_ms,  # Integer millisecond timestamp or None
        }
