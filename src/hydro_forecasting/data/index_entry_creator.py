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
    if basin_data.height < total:
        return np.array([], dtype=int), np.array([], dtype="datetime64[ns]")
    values = basin_data.select(pl.exclude("date")).to_numpy()
    dates = basin_data.select("date").to_numpy().flatten()
    valid_mask = (~np.isnan(values).any(axis=1)).astype(int)
    input_ok = (
        np.convolve(valid_mask, np.ones(input_length, int), mode="valid")
        == input_length
    )
    output_ok = (
        np.convolve(valid_mask, np.ones(output_length, int), mode="valid")
        == output_length
    )
    padded = np.pad(output_ok, (0, input_length), constant_values=False)
    output_shift = padded[input_length : input_length + len(input_ok)]
    seq_ok = input_ok & output_shift
    positions = np.where(seq_ok)[0]
    return positions, dates
