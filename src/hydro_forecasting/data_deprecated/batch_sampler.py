import math
import random
from pathlib import Path

import polars as pl
from torch.utils.data import Sampler


class FileGroupedBatchSampler(Sampler[list[int]]):
    """
    Yield batches of row indices grouped by file_path using metadata.

    Args:
        index_meta_path: Path to metadata Parquet file (file_path, count, start_row_index)
        batch_size: Number of samples per batch
        files_per_batch: Number of different files to sample from in each batch
        shuffle: Whether to shuffle the data
    """

    def __init__(
        self,
        index_meta_path: Path,
        batch_size: int,
        files_per_batch: int = 1,
        shuffle: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.files_per_batch = max(1, files_per_batch)

        # Load metadata: file_path, count, start_row_index
        meta_df = pl.read_parquet(index_meta_path)
        self._file_meta = [
            {
                "file_path": row["file_path"],
                "count": int(row["count"]),
                "start_row_index": int(row["start_row_index"]),
            }
            for row in meta_df.iter_rows(named=True)
        ]
        self._file_to_meta = {m["file_path"]: (m["count"], m["start_row_index"]) for m in self._file_meta}
        self._files = [m["file_path"] for m in self._file_meta]
        total_samples = sum(m["count"] for m in self._file_meta)
        self._num_batches = math.ceil(total_samples / batch_size)

    def __iter__(self):
        # Track remaining samples per file
        file_remaining = {f: self._file_to_meta[f][0] for f in self._files}
        file_next = {f: self._file_to_meta[f][1] for f in self._files}

        files = self._files.copy()
        if self.shuffle:
            random.shuffle(files)

        while sum(file_remaining.values()) > 0:
            available_files = [f for f in files if file_remaining[f] > 0]
            if not available_files:
                break
            if self.shuffle:
                random.shuffle(available_files)
            batch_files = available_files[: self.files_per_batch]
            batch = []
            samples_per_file = math.ceil(self.batch_size / len(batch_files))
            for f in batch_files:
                n_avail = file_remaining[f]
                n_take = min(samples_per_file, n_avail, self.batch_size - len(batch))
                start = file_next[f]
                inds = list(range(start, start + n_take))
                if self.shuffle:
                    random.shuffle(inds)
                batch.extend(inds)
                file_next[f] += n_take
                file_remaining[f] -= n_take
                if len(batch) >= self.batch_size:
                    batch = batch[: self.batch_size]
                    break
            if batch:
                yield batch

    def __len__(self) -> int:
        return self._num_batches
