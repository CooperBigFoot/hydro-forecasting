import numpy as np
import polars as pl
from pathlib import Path
from returns.result import Result, Success, Failure
from tqdm import tqdm
from .preprocessing import split_data, ProcessingConfig

BATCH_SIZE = 100


def load_batch_parquet(
    batch_gauge_ids: list[str],
    time_series_base_dir: Path,
) -> Result[pl.DataFrame, str]:
    """
    Load a batch of time-series Parquet files into a single Polars DataFrame.

    Args:
        batch_gauge_ids: List of gauge ID strings identifying Parquet files (one per gauge).
        time_series_base_dir: Base directory containing the gauge Parquet files.

    Returns:
        Success(pl.DataFrame): A vertically concatenated Polars DataFrame containing all loaded files,
            with an added "gauge_id" column.
        Failure(str): An error message if no files could be loaded or if there were read errors.

    Raises:
        Does not raise exceptions; returns a Failure on critical I/O or parsing errors.
    """
    data: list[pl.DataFrame] = []
    missing: list[str] = []

    for gauge_id in batch_gauge_ids:
        file_path = time_series_base_dir / f"{gauge_id}.parquet"
        if not file_path.exists():
            missing.append(gauge_id)
            continue
        try:
            df = pl.read_parquet(file_path)
            df = df.with_columns(pl.lit(gauge_id).alias("gauge_id"))
            data.append(df)
        except Exception:
            missing.append(gauge_id)
            continue

    if not data:
        return Failure(
            f"No files loaded for batch: {batch_gauge_ids}. Missing or failed: {missing}"
        )

    batch_df = pl.concat(data, how="vertical")
    return Success(batch_df)


def get_split_boundaries(
    train: pl.DataFrame | pl.LazyFrame,
    val: pl.DataFrame | pl.LazyFrame,
    test: pl.DataFrame | pl.LazyFrame,
    gauge_ids: list[str],
) -> dict[str, dict[str, np.datetime64 | None]]:
    """
    Compute the start dates of the validation and test splits per gauge.

    Args:
        train: Training set as a Polars DataFrame or LazyFrame (unused for boundaries).
        val: Validation set as a LazyFrame or DataFrame.
        test: Test set as a LazyFrame or DataFrame.
        gauge_ids: List of gauge ID strings to extract boundaries for.

    Returns:
        A dict mapping each gauge_id to a sub-dict:
        {
            "val_start": np.datetime64 or None,  # earliest date in the val split
            "test_start": np.datetime64 or None, # earliest date in the test split
        }
    """
    # Ensure eager loading for date computations
    val_df = val.collect() if isinstance(val, pl.LazyFrame) else val
    test_df = test.collect() if isinstance(test, pl.LazyFrame) else test

    boundaries: dict[str, dict[str, np.datetime64 | None]] = {}

    for gid in gauge_ids:
        g_val = val_df.filter(pl.col("gauge_id") == gid)
        g_test = test_df.filter(pl.col("gauge_id") == gid)

        val_min = g_val["date"].min() if g_val.height > 0 else None
        test_min = g_test["date"].min() if g_test.height > 0 else None

        boundaries[gid] = {
            "val_start": np.datetime64(val_min) if val_min is not None else None,
            "test_start": np.datetime64(test_min) if test_min is not None else None,
        }

    return boundaries


def find_valid_sequences(
    basin_data: pl.DataFrame,
    input_length: int,
    output_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identify starting indices of valid input/output sequences in time-series data.

    Args:
        basin_data: Polars DataFrame containing time-series with at least "date" and gauge columns.
        input_length: Number of past timesteps to include in the input sequence.
        output_length: Number of future timesteps to include in the output sequence.

    Returns:
        valid_positions: numpy array of integer start indices where both input and output windows
            contain no nulls across cols_to_check.
        dates: numpy array of numpy.datetime64[ns] corresponding to each row in basin_data.
    """
    total_len = input_length + output_length
    if basin_data.height < total_len:
        return np.array([], dtype=int), np.array([], dtype="datetime64[ns]")

    values = basin_data.select(pl.exclude("date")).to_numpy()
    dates = basin_data.select("date").to_numpy().flatten()

    # 1 if no NaNs in row, else 0
    valid_mask = (~np.isnan(values).any(axis=1)).astype(int)

    # rolling sums to check full-window validity
    input_ok = (
        np.convolve(valid_mask, np.ones(input_length, int), mode="valid")
        == input_length
    )
    output_ok = (
        np.convolve(valid_mask, np.ones(output_length, int), mode="valid")
        == output_length
    )

    # align output window to the same start indices
    padded = np.pad(output_ok, (0, input_length), constant_values=False)
    output_shift = padded[input_length : input_length + len(input_ok)]

    seq_ok = input_ok & output_shift
    positions = np.where(seq_ok)[0]

    return positions, dates


def determine_stage(
    input_end_date: np.datetime64,
    boundaries: dict[str, np.datetime64 | None],
) -> str:
    """
    Assign a sequence to train, val, or test based on its end-date and precomputed splits.

    Args:
        input_end_date: numpy.datetime64 indicating the last date of the input window.
        boundaries: Dict with keys "val_start" and "test_start" mapping to numpy.datetime64 or None.

    Returns:
        One of the strings: "train", "val", or "test".
    """
    val_start = boundaries.get("val_start")
    test_start = boundaries.get("test_start")

    if test_start is not None and input_end_date >= test_start:
        return "test"
    if val_start is not None and input_end_date >= val_start:
        return "val"
    return "train"


def create_index_entries(
    gauge_ids: list[str],
    time_series_base_dir: Path,
    static_file_path: Path,
    input_length: int,
    output_length: int,
    processing_config: ProcessingConfig,
    output_dir: Path | None = None,
) -> Result[dict[str, tuple[Path, Path]], str]:
    """
    Build and write index and metadata Parquet files for train/val/test stages using ROP.

    Returns:
        Success(dict): mapping stage names to (index_path, meta_path)
        Failure(str): error message on failure
    """
    # Early validations
    if output_dir is None:
        return Failure("output_dir must be specified for disk-based index creation.")

    proportions = [
        processing_config.train_prop,
        processing_config.val_prop,
        processing_config.test_prop,
    ]
    if not np.isclose(sum(proportions), 1.0):
        return Failure(
            f"Train/val/test proportions must sum to 1.0; got {proportions} (sum={sum(proportions)})"
        )

    try:
        stage_entries: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        total_seq = input_length + output_length
        num_batches = (len(gauge_ids) + BATCH_SIZE - 1) // BATCH_SIZE

        for start in tqdm(
            range(0, len(gauge_ids), BATCH_SIZE),
            total=num_batches,
            desc="Creating index entries",
        ):
            batch_ids = gauge_ids[start : start + BATCH_SIZE]
            load_res = load_batch_parquet(batch_ids, time_series_base_dir)
            if isinstance(load_res, Failure):
                continue
            batch_df = load_res.unwrap()

            train_df, val_df, test_df = split_data(
                df=batch_df, config=processing_config
            )
            boundaries = get_split_boundaries(train_df, val_df, test_df, batch_ids)

            for gid in batch_ids:
                ts_path = time_series_base_dir / f"{gid}.parquet"
                basin_df = batch_df.filter(pl.col("gauge_id") == gid)
                if basin_df.height == 0:
                    continue
                bounds = boundaries.get(gid, {"val_start": None, "test_start": None})
                positions, dates = find_valid_sequences(
                    basin_df, input_length, output_length
                )
                if positions.size == 0:
                    continue
                for idx in positions:
                    if idx + total_seq > basin_df.height:
                        continue
                    end_date = dates[idx + input_length - 1]
                    stage = determine_stage(end_date, bounds)
                    entry = {
                        "file_path": str(ts_path),
                        "gauge_id": gid,
                        "start_idx": int(idx),
                        "end_idx": int(idx + total_seq),
                        "input_end_date": end_date,
                        "valid_sequence": True,
                        "stage": stage,
                        "static_file_path": str(static_file_path),
                    }
                    stage_entries[stage].append(entry)

        output: dict[str, tuple[Path, Path]] = {}
        for stage in ("train", "val", "test"):
            entries = stage_entries.get(stage, [])
            if not entries:
                continue
            df = pl.DataFrame(entries).sort("file_path")
            idx_path = output_dir / f"{stage}_index.parquet"
            df.write_parquet(idx_path)

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
            output[stage] = (idx_path, meta_path)

        return Success(output)
    except Exception as e:
        return Failure(f"Unexpected error during index creation: {e}")
