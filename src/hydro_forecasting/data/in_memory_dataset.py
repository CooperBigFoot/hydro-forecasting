# in_memory_dataset.py
import logging

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from ..exceptions import DataProcessingError

# Configure logging
logger = logging.getLogger(__name__)


def _calculate_valid_sequences(
    basin_data: pl.DataFrame,
    input_length: int,
    output_length: int,
    total_length: int,
    value_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function containing the core numpy logic.

    Args:
        basin_data: Polars DataFrame for a single basin.
        input_length: Length of the input sequence.
        output_length: Length of the output sequence.
        total_length: Total sequence length (input + output).
        value_columns: List of columns to check for NaNs.

    Returns:
        Tuple containing an array of valid start indices and an array of dates.
    """
    existing_value_columns = [col for col in value_columns if col in basin_data.columns]
    if not existing_value_columns:
        logger.warning(
            f"No value columns found in basin_data for NaN checking. Columns expected: {value_columns}, Columns present: {basin_data.columns}"
        )
        return np.array([], dtype=int), basin_data.select(
            "date"
        ).to_series().to_numpy() if "date" in basin_data.columns else np.array([])

    values = basin_data.select(existing_value_columns).to_numpy()
    dates_series = basin_data.get_column("date")  # Get Series
    dates = dates_series.to_numpy()

    valid_row_mask = ~np.isnan(values).any(axis=1)
    input_valid_sums = np.convolve(valid_row_mask, np.ones(input_length, dtype=int), mode="valid")
    is_input_window_valid = input_valid_sums == input_length

    output_valid_sums = np.convolve(valid_row_mask, np.ones(output_length, dtype=int), mode="valid")
    is_output_window_valid = output_valid_sums == output_length

    num_possible_starts = len(valid_row_mask) - total_length + 1
    if num_possible_starts <= 0:
        return np.array([], dtype=int), dates

    input_ok = is_input_window_valid[:num_possible_starts]
    output_ok = is_output_window_valid[input_length : input_length + num_possible_starts]

    valid_sequence_starts_mask = input_ok & output_ok
    positions = np.where(valid_sequence_starts_mask)[0]

    return positions, dates


def find_valid_sequences(
    basin_data: pl.DataFrame,
    input_length: int,
    output_length: int,
    target_col_name: str,
    forcing_cols_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identify start indices of valid sequences within a time series DataFrame.

    A sequence is valid if both the input window (length `input_length`) and
    the subsequent output window (length `output_length`) contain no null (NaN)
    values in any column identified by `target_col_name` and `forcing_cols_names`.

    Args:
        basin_data: A Polars DataFrame containing time series data for a single
                    basin/group. It *should not* contain the group_identifier column
                    at this point, only 'date' and value columns.
        input_length: The required length of the input sequence.
        output_length: The required length of the output sequence.
        target_col_name: Name of the target column to check for NaNs.
        forcing_cols_names: List of forcing column names to check for NaNs.

    Returns:
        A tuple containing:
            - np.ndarray: An array of integer start indices for valid sequences.
            - np.ndarray: An array of dates corresponding to each row in basin_data.

    Raises:
        DataProcessingError: If validation fails or calculation errors occur.
    """
    total_length = input_length + output_length
    value_columns_to_check = [target_col_name] + forcing_cols_names
    value_columns_to_check = sorted(set(value_columns_to_check))  # Unique and sorted

    # Validation checks
    if basin_data.height < total_length:
        raise DataProcessingError(
            f"Data height ({basin_data.height}) is less than total sequence length ({total_length})"
        )

    if "date" not in basin_data.columns:
        raise DataProcessingError("DataFrame missing 'date' column")

    if not all(col in basin_data.columns for col in value_columns_to_check):
        raise DataProcessingError(
            f"DataFrame missing one or more value columns for null checking. Expected: {value_columns_to_check}, Present: {basin_data.columns}"
        )

    try:
        return _calculate_valid_sequences(
            basin_data=basin_data,
            input_length=input_length,
            output_length=output_length,
            total_length=total_length,
            value_columns=value_columns_to_check,
        )
    except Exception as exc:
        raise DataProcessingError(f"NumPy calculation error: {exc}") from exc


class InMemoryChunkDataset(Dataset):
    """
    Dataset that operates on a pre-loaded and tensorized chunk of basin data.

    Args:
        chunk_column_tensors: Dictionary mapping column names to PyTorch Tensors for the current chunk.
                              Each tensor represents a column's data for all basins in the chunk, concatenated.
        static_data_cache: Preloaded dictionary mapping basin_id to static feature PyTorch Tensors.
        index_entries: List of tuples defining valid sequences: (basin_id, start_row_relative_to_basin, end_row_relative_to_basin).
                       Indices are relative to the original per-basin data before concatenation into chunk_column_tensors.
        basin_row_map: Dictionary mapping basin_id to a tuple (chunk_basin_start_abs, num_rows_in_chunk_for_basin).
                       `chunk_basin_start_abs` is the absolute starting row index for this basin_id's data
                       within the `chunk_column_tensors`.
        input_length: Number of timesteps for model input.
        output_length: Number of timesteps for model output.
        target_name: Name of the target variable column (key in chunk_column_tensors).
        forcing_features_names: List of forcing feature column names (keys in chunk_column_tensors).
        static_features_names: List of static feature names (used for consistency, actual features are in static_data_cache).
        group_identifier_name: Name of the column identifying basins (e.g., 'gauge_id'), primarily for returning with sample.
        is_autoregressive: If True, include past target in input features X.
        input_features_ordered: Ordered list of feature names for constructing X.
        future_features_ordered: Ordered list of feature names for constructing future inputs.
        include_input_end_date: Whether to calculate and include 'input_end_date' in the output.
                                If True, 'date' tensor must be present in chunk_column_tensors.
    """

    def __init__(
        self,
        chunk_column_tensors: dict[str, torch.Tensor],
        static_data_cache: dict[str, torch.Tensor],
        index_entries: list[tuple[str, int, int]],
        basin_row_map: dict[str, tuple[int, int]],
        input_length: int,
        output_length: int,
        target_name: str,
        forcing_features_names: list[str],
        static_features_names: list[str],
        group_identifier_name: str,
        is_autoregressive: bool,
        input_features_ordered: list[str],
        future_features_ordered: list[str],
        include_input_end_date: bool = True,
    ):
        super().__init__()
        self.chunk_column_tensors = chunk_column_tensors
        self.static_data_cache = static_data_cache
        self.index_entries = index_entries
        self.basin_row_map = basin_row_map
        self.input_length = input_length
        self.output_length = output_length
        self.target_name = target_name
        self.forcing_features_names = sorted(set(forcing_features_names))
        self.group_identifier_name = group_identifier_name
        self.is_autoregressive = is_autoregressive
        self.input_features_ordered_for_X = input_features_ordered
        self.future_features_ordered = future_features_ordered
        self.include_input_end_date = include_input_end_date
        self.static_features_names = sorted(set(static_features_names))

        if self.include_input_end_date and "date" not in self.chunk_column_tensors:
            logger.warning(
                "'date' tensor not found in chunk_column_tensors, but include_input_end_date is True. input_end_date will be None."
            )
            self.include_input_end_date = False

    def __len__(self) -> int:
        return len(self.index_entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int | None]:
        if not (0 <= idx < len(self.index_entries)):
            raise IndexError(f"Index {idx} out of range for {len(self.index_entries)} samples.")

        basin_id, basin_start_row_relative, basin_end_row_relative = self.index_entries[idx]

        chunk_basin_start_abs, num_rows_in_chunk_for_basin = self.basin_row_map.get(basin_id, (None, None))
        if chunk_basin_start_abs is None or num_rows_in_chunk_for_basin is None:
            raise RuntimeError(
                f"Basin {basin_id} missing from basin_row_map or invalid length. "
                f"Basin_row_map keys: {list(self.basin_row_map.keys())}"
            )

        sequence_start_abs = chunk_basin_start_abs + basin_start_row_relative
        sequence_duration = self.input_length + self.output_length
        sequence_end_abs = sequence_start_abs + sequence_duration

        if sequence_end_abs > chunk_basin_start_abs + num_rows_in_chunk_for_basin:
            logger.error(
                f"Slicing error for basin {basin_id} at sample index {idx}: "
                f"Attempting to slice from absolute start {sequence_start_abs} for duration {sequence_duration} (ends at {sequence_end_abs}). "
                f"Basin data in chunk starts at {chunk_basin_start_abs} and has length {num_rows_in_chunk_for_basin} "
                f"(ends at {chunk_basin_start_abs + num_rows_in_chunk_for_basin}). "
                f"Relative start: {basin_start_row_relative}, relative end: {basin_end_row_relative}"
            )
            raise IndexError(
                f"Calculated sequence end index {sequence_end_abs} exceeds concatenated data length for basin {basin_id}."
            )

        # Slice the window for all relevant tensors
        target_tensor_window = self.chunk_column_tensors[self.target_name][sequence_start_abs:sequence_end_abs]

        forcing_tensors_window: dict[str, torch.Tensor] = {}
        for name in self.forcing_features_names:
            if name in self.chunk_column_tensors:
                forcing_tensors_window[name] = self.chunk_column_tensors[name][sequence_start_abs:sequence_end_abs]
            else:
                # This should not happen if DataModule prepares tensors correctly. Better safe than sorry tho.
                raise KeyError(f"Forcing feature {name} not found in chunk_column_tensors.")

        X_model_components = []
        for feat_name in self.input_features_ordered_for_X:
            if feat_name == self.target_name and self.is_autoregressive:
                X_model_components.append(target_tensor_window[: self.input_length])
            elif feat_name in forcing_tensors_window:
                X_model_components.append(forcing_tensors_window[feat_name][: self.input_length])
            else:
                raise KeyError(f"Feature {feat_name} for X_model not found in target or forcing tensors window.")
        X_model = torch.stack(X_model_components, dim=-1)

        y_model = target_tensor_window[self.input_length : self.input_length + self.output_length]
        if y_model.ndim > 1 and y_model.shape[-1] == 1:
            y_model = y_model.squeeze(-1)

        future_model_components = []
        if self.future_features_ordered:
            for feat_name in self.future_features_ordered:
                if feat_name in forcing_tensors_window:
                    future_model_components.append(
                        forcing_tensors_window[feat_name][self.input_length : self.input_length + self.output_length]
                    )
                else:
                    raise KeyError(f"Feature {feat_name} for future_model not found in forcing_tensors_window.")
            future_model = torch.stack(future_model_components, dim=-1)
        else:
            future_model = torch.empty((self.output_length, 0), dtype=X_model.dtype)

        # Static features
        static_tensor = self.static_data_cache.get(basin_id)
        if static_tensor is None:
            logger.warning(f"Static data for basin {basin_id} not found. Using zeros.")
            static_tensor = torch.zeros(len(self.static_features_names), dtype=X_model.dtype)
        elif static_tensor.shape[0] != len(self.static_features_names):
            logger.warning(
                f"Static data for basin {basin_id} has unexpected shape {static_tensor.shape}. "
                f"Expected {len(self.static_features_names)} features. Using zeros."
            )
            static_tensor = torch.zeros(len(self.static_features_names), dtype=X_model.dtype)

        input_end_date_ms: int | None = None
        if self.include_input_end_date:
            try:
                date_tensor_window = self.chunk_column_tensors["date"][sequence_start_abs:sequence_end_abs]
                if date_tensor_window.numel() > 0 and self.input_length > 0:
                    input_end_date_ms = date_tensor_window[self.input_length - 1].item()
                    if not isinstance(input_end_date_ms, int):  # Should be long, convert to int
                        input_end_date_ms = int(input_end_date_ms)
            except Exception as e:
                logger.error(f"Error retrieving input_end_date for basin {basin_id}, sample {idx}: {e}")
                input_end_date_ms = None

        return {
            "X": X_model,
            "y": y_model,
            "static": static_tensor,
            "future": future_model,
            self.group_identifier_name: basin_id,
            "input_end_date": input_end_date_ms,
        }
