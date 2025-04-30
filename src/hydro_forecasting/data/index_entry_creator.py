import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional
from returns.result import Result, Success, Failure
from tqdm import tqdm

BATCH_SIZE = 100


def create_index_entries(
    gauge_ids: list[str],
    time_series_dirs: dict[str, Path],  # {'train': Path, 'val': Path, 'test': Path}
    static_file_path: Optional[Path],
    input_length: int,
    output_length: int,
    output_dir: Path,
) -> Result[dict[str, tuple[Path, Path]], str]:
    """
    Build and write index and metadata Parquet files for train/val/test stages
    using pre-split data in directories.
    """
    # Validate stage directories
    for stage, dir_path in time_series_dirs.items():
        if not dir_path.exists() or not dir_path.is_dir():
            return Failure(f"Directory for '{stage}' does not exist: {dir_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, tuple[Path, Path]] = {}

    try:
        for stage, dir_path in time_series_dirs.items():
            entries: list[dict] = []
            # Process gauges in batches
            for i in tqdm(
                range(0, len(gauge_ids), BATCH_SIZE), desc=f"Indexing {stage}"
            ):
                batch = gauge_ids[i : i + BATCH_SIZE]
                entries.extend(
                    process_stage_directory(
                        gauge_ids=batch,
                        stage_dir=dir_path,
                        static_file_path=static_file_path,
                        input_length=input_length,
                        output_length=output_length,
                        stage=stage,
                    )
                )

            if not entries:
                continue

            # Create DataFrame and save
            df = pl.DataFrame(entries).sort("file_path")
            idx_path = output_dir / f"{stage}_index.parquet"
            df.write_parquet(idx_path)

            # Metadata
            meta = (
                df.with_row_count("row_nr")
                .group_by("file_path")
                .agg(
                    [
                        pl.count().alias("count"),
                        pl.min("row_nr").alias("start_row_index"),
                    ]
                )
                .sort("file_path")
            )
            meta_path = output_dir / f"{stage}_index_meta.parquet"
            meta.write_parquet(meta_path)

            results[stage] = (idx_path, meta_path)

        return Success(results)
    except Exception as e:
        return Failure(f"Unexpected error during index creation: {e}")


def process_stage_directory(
    gauge_ids: list[str],
    stage_dir: Path,
    static_file_path: Optional[Path],
    input_length: int,
    output_length: int,
    stage: str,
) -> list[dict]:
    """Process a single split directory and collect index entries."""
    entries: list[dict] = []
    total_seq = input_length + output_length

    for gid in gauge_ids:
        ts_path = stage_dir / f"{gid}.parquet"
        if not ts_path.exists():
            continue
        try:
            df = pl.read_parquet(ts_path)
            df = df.with_columns(pl.lit(gid).alias("gauge_id"))
            positions, dates = find_valid_sequences(df, input_length, output_length)
            for idx in positions:
                if idx + total_seq > df.height:
                    continue
                end_date = dates[idx + input_length - 1]
                entry = {
                    "file_path": str(ts_path),
                    "gauge_id": gid,
                    "start_idx": int(idx),
                    "end_idx": int(idx + total_seq),
                    "input_end_date": end_date,
                    "valid_sequence": True,
                    "stage": stage,
                    "static_file_path": str(static_file_path)
                    if static_file_path
                    else None,
                }
                entries.append(entry)
        except Exception:
            continue
    return entries


def find_valid_sequences(
    basin_data: pl.DataFrame,
    input_length: int,
    output_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify start indices where both input and output windows have no nulls."""
    total = input_length + output_length

    # Debug info
    has_date = "date" in basin_data.columns
    print(
        f"Processing basin with {basin_data.height} rows and columns: {basin_data.columns}"
    )

    if basin_data.height < total:
        print(
            f"WARNING: Basin data has {basin_data.height} rows, but needs at least {total} for a valid sequence"
        )
        return np.array([], dtype=int), np.array([], dtype="datetime64[ns]")

    # Check for missing values in the data
    null_counts = basin_data.null_count()
    if null_counts.sum().item() > 0:
        non_zero_nulls = {
            col: count
            for col, count in zip(null_counts.columns, null_counts.row(0))
            if count > 0
        }
        print(f"WARNING: Basin data contains nulls: {non_zero_nulls}")

    # Select all columns except 'date' for null checking
    values = basin_data.select(pl.exclude("date")).to_numpy()
    dates = basin_data.select("date").to_numpy().flatten() if has_date else np.array([])

    # Create mask of valid rows (no NaNs in any column)
    valid_mask = (~np.isnan(values).any(axis=1)).astype(int)
    valid_count = valid_mask.sum()
    print(
        f"INFO: Basin has {valid_count} valid rows out of {basin_data.height} total rows"
    )

    if valid_count < total:
        print(
            f"WARNING: Not enough valid rows ({valid_count}) for a sequence of length {total}"
        )
        return np.array([], dtype=int), np.array([], dtype="datetime64[ns]")

    # Find contiguous input windows of length input_length where all values are valid
    input_ok = (
        np.convolve(valid_mask, np.ones(input_length, int), mode="valid")
        == input_length
    )
    input_window_count = input_ok.sum()

    # Find contiguous output windows of length output_length where all values are valid
    output_ok = (
        np.convolve(valid_mask, np.ones(output_length, int), mode="valid")
        == output_length
    )
    output_window_count = output_ok.sum()

    print(
        f"INFO: Found {input_window_count} valid input windows and {output_window_count} valid output windows"
    )

    # Pad the output windows and shift to align with input windows
    padded = np.pad(output_ok, (0, input_length), constant_values=False)
    output_shift = padded[input_length : input_length + len(input_ok)]

    # Find positions where both input and output windows are valid
    seq_ok = input_ok & output_shift
    positions = np.where(seq_ok)[0]

    print(f"INFO: Found {len(positions)} valid sequences of length {total}")

    return positions, dates
