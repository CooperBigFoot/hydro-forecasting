import math
import random
from torch.utils.data import Sampler
from typing import Dict, List, Iterator


class FileGroupedBatchSampler(Sampler[List[int]]):
    """
    Yield batches of indices grouped by multiple file paths.

    Args:
        batch_index_entries: List of dictionaries containing index entries
        batch_size: Number of samples per batch
        files_per_batch: Number of different files to sample from in each batch
        shuffle: Whether to shuffle the data
    """

    def __init__(
        self,
        batch_index_entries: List[Dict],
        batch_size: int,
        files_per_batch: int = 1,
        shuffle: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.files_per_batch = max(1, files_per_batch)

        # Group indices by file path
        self._file_to_inds: Dict[str, List[int]] = {}
        for i, e in enumerate(batch_index_entries):
            self._file_to_inds.setdefault(e["file_path"], []).append(i)

        # Calculate total number of batches
        total_samples = sum(len(v) for v in self._file_to_inds.values())
        self._num_batches = math.ceil(total_samples / batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        # Get list of files and initialize position trackers
        files = list(self._file_to_inds.keys())
        file_indices = {f: 0 for f in files}
        remaining_indices = {f: len(self._file_to_inds[f]) for f in files}

        # Shuffle the contents of each file if needed
        if self.shuffle:
            for f in files:
                random.shuffle(self._file_to_inds[f])

        # Continue until all indices are exhausted
        while sum(remaining_indices.values()) > 0:
            # Select files with remaining samples
            available_files = [f for f in files if remaining_indices[f] > 0]
            if not available_files:
                break

            # Shuffle the file order if needed
            if self.shuffle:
                random.shuffle(available_files)

            # Select a subset of files for this batch
            batch_files = available_files[: self.files_per_batch]

            # Create a batch by taking samples from each selected file
            batch = []
            samples_per_file = math.ceil(self.batch_size / len(batch_files))

            for f in batch_files:
                # Get indices from this file
                start_idx = file_indices[f]
                end_idx = min(start_idx + samples_per_file, len(self._file_to_inds[f]))
                file_batch = self._file_to_inds[f][start_idx:end_idx]
                batch.extend(file_batch)

                # Update tracking variables
                file_indices[f] = end_idx
                remaining_indices[f] = len(self._file_to_inds[f]) - end_idx

                # If we have enough samples, stop adding more
                if len(batch) >= self.batch_size:
                    batch = batch[: self.batch_size]  # Trim to batch_size
                    break

            # Only yield non-empty batches
            if batch:
                yield batch

    def __len__(self) -> int:
        return self._num_batches
