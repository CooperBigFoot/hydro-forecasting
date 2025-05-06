# filename: src/hydro_forecasting/data/in_memory_dataset.py
import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple
from returns.result import Result, Success, Failure, safe
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Sequence Finding Logic (Adapted from index_entry_creator.py) ---
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
    values = basin_data.select(value_columns).to_numpy()
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
                    basin/group, including a 'date' column.
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
        .bind(lambda df: Success((df, [col for col in df.columns if col != "date"])))
        .bind(
            lambda df_cols: Success(df_cols)
            if df_cols[1]
            else Failure("No value columns found for null checking (besides 'date')")
        )
        .bind(
            lambda df_cols: _calculate_valid_sequences(
                basin_data=df_cols[0],
                input_length=input_length,
                output_length=output_length,
                total_length=total_length,
                value_columns=df_cols[1],
            ).alt(lambda exc: f"NumPy calculation error: {exc}")
        )
    )


# --- Dataset Class ---
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
        ] = []  # (basin_id, start_row, end_row)
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
        required_cols = ["date", self.target] + self.forcing_features

        for basin_id in self.basin_ids:
            file_path = stage_dir / f"{basin_id}.parquet"
            if file_path.exists():
                try:
                    # Use scan_parquet for potentially better memory efficiency during loading
                    # Select only necessary columns early
                    lf = (
                        pl.scan_parquet(str(file_path))
                        .select(required_cols)
                        .sort("date")
                    )
                    df = lf.collect()  # Collect into memory
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
            self.chunk_data = pl.DataFrame()
            self.index_entries = []
            return

        # Concatenate all basin data for the chunk
        try:
            self.chunk_data = pl.concat(dfs_to_concat, how="vertical")
            logger.info(
                f"Chunk data loaded. Shape: {self.chunk_data.shape}. Memory usage: {self.chunk_data.estimated_size('mb'):.2f} MB"
            )
        except Exception as e:
            logger.error(f"Failed to concatenate dataframes for chunk: {e}")
            # Handle potential memory errors during concatenation
            # Fallback or error logging
            self.chunk_data = pl.DataFrame()  # Ensure chunk_data is an empty DF
            self.index_entries = []
            return

        # Free up memory
        del dfs_to_concat
        import gc

        gc.collect()

        # Precompute index
        logger.info("Precomputing index for in-memory chunk...")
        current_row = 0
        for basin_id in self.basin_ids:
            # Filter the chunk_data for the current basin
            # Ensure we handle cases where a basin might have been skipped during loading
            basin_df = self.chunk_data.filter(pl.col(self.group_identifier) == basin_id)

            if basin_df.height == 0:
                logger.warning(
                    f"No data found for basin {basin_id} in concatenated chunk, skipping indexing."
                )
                continue

            # Map basin_id to its row range in the concatenated DataFrame
            self.basin_row_map[basin_id] = (current_row, basin_df.height)

            find_result = find_valid_sequences(
                basin_df, self.input_length, self.output_length
            )

            if isinstance(find_result, Success):
                positions, _ = find_result.unwrap()
                for start_idx in positions:
                    # Store index entry relative to the start of this basin's data within the chunk
                    # We'll use basin_row_map later in __getitem__ to get absolute rows
                    self.index_entries.append(
                        (
                            basin_id,
                            int(start_idx),
                            int(start_idx + self.input_length + self.output_length),
                        )
                    )
            else:
                logger.warning(
                    f"Could not find valid sequences for {basin_id}: {find_result.failure()}"
                )

            current_row += basin_df.height  # Update the starting row for the next basin

        # Shuffle index entries for training stage
        if self.stage == "train":
            np.random.shuffle(self.index_entries)

        end_time = time.time()
        logger.info(
            f"Chunk loading and indexing complete for stage '{self.stage}'. "
            f"Processed {processed_count}, Skipped {skipped_count}. "
            f"Found {len(self.index_entries)} valid samples. Time: {end_time - start_time:.2f}s"
        )

    def __len__(self) -> int:
        """Return the total number of valid samples in this chunk."""
        return len(self.index_entries)

    def __getitem__(self, idx: int) -> dict[str, Union[torch.Tensor, str, int]]:
        """Retrieve a single sample from the in-memory chunk."""
        if not (0 <= idx < len(self.index_entries)):
            raise IndexError(
                f"Index {idx} out of range for chunk size {len(self.index_entries)}"
            )

        # 1. Get precomputed index entry
        basin_id, basin_start_row_relative, basin_end_row_relative = self.index_entries[
            idx
        ]

        # 2. Find the absolute row range for this basin in the concatenated chunk_data
        chunk_basin_start_abs, basin_len = self.basin_row_map.get(
            basin_id, (None, None)
        )
        if chunk_basin_start_abs is None:
            raise RuntimeError(
                f"Basin {basin_id} not found in basin_row_map. This should not happen."
            )

        # Calculate absolute start and end rows for the required slice within the chunk_data
        start_abs = chunk_basin_start_abs + basin_start_row_relative
        end_abs = chunk_basin_start_abs + basin_end_row_relative  # end_row is exclusive

        # 3. Slice the sequence directly from the in-memory DataFrame
        # Use slice instead of iloc for polars; end index is exclusive
        try:
            # Ensure slicing doesn't go out of bounds for the basin's data within the chunk
            if end_abs > chunk_basin_start_abs + basin_len:
                raise IndexError(
                    f"Calculated end index {end_abs} exceeds basin data length ({chunk_basin_start_abs + basin_len})"
                )

            sequence_df = self.chunk_data.slice(start_abs, end_abs - start_abs)

            if sequence_df.height != (self.input_length + self.output_length):
                raise ValueError(
                    f"Sliced sequence for index {idx} has incorrect length: {sequence_df.height}. Expected: {self.input_length + self.output_length}"
                )

        except Exception as e:
            logger.error(
                f"Error slicing data for index {idx}, basin {basin_id}, start_abs {start_abs}, end_abs {end_abs}: {e}"
            )
            # Return a dummy sample or re-raise the error depending on desired behavior
            raise RuntimeError(f"Failed to slice data for index {idx}") from e

        # 4. Extract X, y, and future features
        # Use .slice again for input and output splits
        inp_df = sequence_df.slice(0, self.input_length)
        out_df = sequence_df.slice(self.input_length, self.output_length)

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

        # 5. Retrieve static features
        static_arr = self.static_data_cache.get(
            basin_id,
            np.zeros(len(self.static_features), dtype=np.float32),  # Default if missing
        )
        if static_arr.shape[0] != len(self.static_features):
            logger.warning(
                f"Static feature array shape mismatch for basin {basin_id}. Expected {len(self.static_features)}, Got {static_arr.shape[0]}. Using zeros."
            )
            static_arr = np.zeros(len(self.static_features), dtype=np.float32)

        # 6. Convert to PyTorch tensors
        try:
            X = torch.tensor(X_np, dtype=torch.float32)
            y = torch.tensor(y_np, dtype=torch.float32)
            future = torch.tensor(future_np, dtype=torch.float32)
            static = torch.tensor(static_arr, dtype=torch.float32)

            # Optional: Add NaN checks here if needed during debugging
            # if torch.isnan(X).any() or torch.isnan(y).any() or torch.isnan(future).any() or torch.isnan(static).any():
            #     logger.warning(f"NaN found in tensors for sample index {idx}, basin {basin_id}")

        except Exception as e:
            logger.error(
                f"Error converting NumPy arrays to tensors for index {idx}, basin {basin_id}: {e}"
            )
            raise RuntimeError(
                f"Failed to convert data to Tensors for index {idx}"
            ) from e

        # 7. Get metadata (input end date)
        try:
            # Get the date corresponding to the *end* of the input sequence
            input_end_date_val = inp_df.select(pl.col("date")).item(
                -1, 0
            )  # Get last date from input slice

            # Convert to timestamp (e.g., milliseconds since epoch)
            # Polars datetime is microseconds since epoch, convert to milliseconds or seconds as needed
            input_end_ts_ms = int(input_end_date_val.timestamp(time_unit="ms"))

        except Exception as e:
            logger.error(
                f"Error getting input end date for index {idx}, basin {basin_id}: {e}"
            )
            input_end_ts_ms = 0  # Assign a default value or handle appropriately

        return {
            "X": X,
            "y": y,
            "static": static,
            "future": future,
            self.group_identifier: basin_id,
            "input_end_date": input_end_ts_ms,
        }
