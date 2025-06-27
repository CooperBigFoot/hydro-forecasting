import traceback  # Import traceback
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from returns.result import Failure, Result, Success, safe
from tqdm import tqdm

BATCH_SIZE = 100


def create_index_entries(
    gauge_ids: list[str],
    time_series_dirs: dict[str, Path],
    static_file_path: Path | None,
    input_length: int,
    output_length: int,
    output_dir: Path,
    group_identifier: str = "gauge_id",
) -> Result[dict[str, tuple[Path, Path]], str]:
    """
    Build and write index and metadata Parquet files for train/val/test stages.

    Reads time series data from pre-split directories, identifies valid
    sequences based on input/output lengths and null values, and writes
    index and metadata files to the specified output directory.

    Args:
        gauge_ids: List of identifiers (e.g., gauge IDs) to process.
        time_series_dirs: Dictionary mapping stage names ('train', 'val', 'test')
                          to directories containing the corresponding time series
                          Parquet files (one file per gauge_id).
        static_file_path: Optional path to the Parquet file containing static
                          attributes for all gauges.
        input_length: The number of time steps required for model input.
        output_length: The number of time steps required for model output (prediction).
        output_dir: The directory where the index and metadata Parquet files
                    will be saved.
        group_identifier: The name of the column used to identify individual
                          time series groups (e.g., 'gauge_id'). Defaults to 'gauge_id'.

    Returns:
        Success containing a dictionary mapping stage names to tuples of
        (index_file_path, metadata_file_path), or Failure with an error message.

    Raises:
        FileNotFoundError: If a stage directory specified in `time_series_dirs`
                           does not exist (returned as Failure).
    """
    # Validate stage directories
    for stage, dir_path in time_series_dirs.items():
        if not dir_path.exists() or not dir_path.is_dir():
            return Failure(f"Directory for '{stage}' does not exist: {dir_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, tuple[Path, Path]] = {}

    try:
        for stage, dir_path in time_series_dirs.items():
            idx_path = output_dir / f"{stage}_index.parquet"
            meta_path = output_dir / f"{stage}_index_meta.parquet"

            idx_writer: pq.ParquetWriter | None = None
            meta_writer: pq.ParquetWriter | None = None
            row_counter = 0

            input_length + output_length

            for i in tqdm(range(0, len(gauge_ids), BATCH_SIZE), desc=f"Indexing {stage}"):
                batch = gauge_ids[i : i + BATCH_SIZE]
                stage_result = process_stage_directory(
                    gauge_ids=batch,
                    stage_dir=dir_path,
                    static_file_path=static_file_path,
                    input_length=input_length,
                    output_length=output_length,
                    stage=stage,
                    group_identifier=group_identifier,
                )
                if isinstance(stage_result, Failure):
                    print(f"ERROR: Failed processing batch for stage {stage}: {stage_result.failure()}")
                    continue

                batch_entries = stage_result.unwrap()
                if not batch_entries:
                    continue

                # Convert batch entries to Polars DataFrame
                df_batch = pl.DataFrame(batch_entries)

                # Write index entries in streaming fashion
                table = df_batch.to_arrow()
                if idx_writer is None:
                    idx_writer = pq.ParquetWriter(str(idx_path), table.schema)
                idx_writer.write_table(table)

                # Compute per-batch metadata with global row numbering
                df_meta = (
                    df_batch.with_row_count("row_nr")
                    .with_columns((pl.col("row_nr") + row_counter).alias("global_row_nr"))
                    .group_by("file_path")
                    .agg(
                        [
                            pl.count().alias("count"),
                            pl.min("global_row_nr").alias("start_row_index"),
                        ]
                    )
                )
                meta_table = df_meta.select(["file_path", "count", "start_row_index"]).to_arrow()
                if meta_writer is None:
                    meta_writer = pq.ParquetWriter(str(meta_path), meta_table.schema)
                meta_writer.write_table(meta_table)

                # Advance global counter
                row_counter += df_batch.height

            # Close writers if they were used
            if idx_writer:
                idx_writer.close()
            else:
                print(f"WARNING: No valid sequences found for stage '{stage}'.")
                continue
            if meta_writer:
                meta_writer.close()

            results[stage] = (idx_path, meta_path)

        if not results:
            return Failure("No index entries were generated for any stage.")

        return Success(results)
    except Exception as e:
        return Failure(f"Unexpected error during index creation: {e}\n{traceback.format_exc()}")


def process_stage_directory(
    gauge_ids: list[str],
    stage_dir: Path,
    static_file_path: Path | None,
    input_length: int,
    output_length: int,
    stage: str,
    group_identifier: str,
) -> Result[list[dict], str]:
    """
    Process time series files within a single stage directory to find valid sequences.

    Args:
        gauge_ids: List of identifiers (e.g., gauge IDs) to process within this stage.
        stage_dir: Path to the directory containing Parquet files for this stage.
        static_file_path: Optional path to the static attributes file.
        input_length: The number of time steps for model input.
        output_length: The number of time steps for model output.
        stage: The name of the current processing stage (e.g., 'train').
        group_identifier: The name of the column identifying the group (e.g., 'gauge_id').

    Returns:
        Success containing a list of dictionaries, where each dictionary represents
        a valid index entry, or Failure with an error message if processing fails
        for any file in the batch.
    """
    entries: list[dict] = []
    total_seq = input_length + output_length

    for gid in gauge_ids:
        ts_path = stage_dir / f"{gid}.parquet"
        if not ts_path.exists():
            continue
        try:
            df = pl.read_parquet(ts_path)
            if group_identifier in df.columns:
                df = df.drop(group_identifier)
            else:
                print(f"WARNING: Column '{group_identifier}' not found in {ts_path}. Skipping drop.")

            find_result = find_valid_sequences(df, input_length, output_length)

            if isinstance(find_result, Failure):
                # print(
                #     f"WARNING: Skipping {ts_path} due to error in find_valid_sequences: {find_result.failure()}"
                # )
                continue

            positions, dates = find_result.unwrap()

            for idx in positions:
                if idx + total_seq > df.height:
                    continue
                end_date_idx = idx + input_length - 1
                if end_date_idx >= len(dates):
                    print(
                        f"WARNING: Calculated end_date_idx {end_date_idx} out of bounds for dates array (len {len(dates)}) in {ts_path}. Skipping sequence at index {idx}."
                    )
                    continue
                end_date = dates[end_date_idx]

                entry = {
                    "file_path": str(ts_path),
                    group_identifier: gid,
                    "start_idx": int(idx),
                    "end_idx": int(idx + total_seq),
                    "input_end_date": end_date,
                    "valid_sequence": True,
                    "stage": stage,
                    "static_file_path": str(static_file_path) if static_file_path else None,
                }
                entries.append(entry)
        except Exception as e:
            print(f"ERROR: Failed to process file {ts_path} for gauge {gid}: {e}\n{traceback.format_exc()}")
            continue
    return Success(entries)


@safe
def _calculate_valid_sequences(
    basin_data: pl.DataFrame,
    input_length: int,
    output_length: int,
    total_length: int,
    value_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Helper function containing the core numpy logic, wrapped by @safe."""
    values = basin_data.select(value_columns).to_numpy()
    dates = basin_data.select("date").to_numpy().flatten()

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
    valid_sequence_starts = input_ok & output_ok
    positions = np.where(valid_sequence_starts)[0]

    return positions, dates


def find_valid_sequences(
    basin_data: pl.DataFrame,
    input_length: int,
    output_length: int,
) -> Result[tuple[np.ndarray, np.ndarray], str]:
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
            else Failure(f"Data height ({df.height}) is less than total sequence length ({total_length})")
        )
        .bind(lambda df: Success(df) if "date" in df.columns else Failure("DataFrame missing 'date' column"))
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
