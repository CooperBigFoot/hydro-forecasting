# Performance Optimization Review: Hydrological Forecasting System

Thank you for providing the additional context about your 10,000 basin training scenario. This significantly changes my analysis, as you have a challenging data access pattern that requires balancing basin diversity with I/O efficiency.

## Current Architecture Analysis

Your system currently:

- Stores data for 10,000 individual basins in separate Parquet files
- Uses `FileCache` to avoid re-reading the same file repeatedly
- Uses `files_per_batch=50` to create diverse batches
- Employs 10 workers for parallel loading

This approach maintains good basin diversity but creates severe I/O bottlenecks, which explains the 60% CPU but only 2% GPU utilization you're observing.

## Critical Performance Issues

### 1. File Access Pattern Bottleneck

The most critical bottleneck appears to be in `FileGroupedBatchSampler.__iter__` and `FileCache.get_file`:

```python
# Each batch requires loading from 50 different files
for f in batch_files:
    n_avail = file_remaining[f]
    n_take = min(samples_per_file, n_avail, self.batch_size - len(batch))
    # ...
```

And in the cache:

```python
def get_file(self, file_path: str, columns: Optional[list[str]] = None):
    with self._lock:  # Global lock creates contention
        # ...
```

**Issues:**

- Accessing 50 different files per batch is extremely inefficient
- Global lock in `FileCache` serializes all file access
- LRU eviction doesn't consider access patterns

### 2. Data Processing Overhead

The conversion pipeline in `HydroLazyDataset.__getitem__` is inefficient:

```python
X_np = inp.select(self.input_features).to_numpy().astype(np.float32)
X = torch.tensor(X_np, dtype=torch.float32)
```

This creates multiple copies of the data and unnecessary conversions.

## Recommended Optimizations (Highest Priority First)

### 1. Implement a Two-Level Batching Strategy

Instead of creating each batch from 50 different files:

```
# Pseudo-approach
1. Load data from N files (e.g., 100-500) into a "super-batch" in memory
2. Create multiple training batches (e.g., 10-20) from this super-batch
3. Move to next set of files
```

This amortizes file I/O costs across multiple training iterations.

### 2. Redesign the File Caching System

Replace the global lock with more granular concurrency control:

```
# Approach
1. Implement file-level locks instead of a global lock
2. Use a frequency-based eviction policy alongside LRU
3. Increase cache size based on available system memory
4. Pre-warm the cache with most frequently accessed files
```

### 3. Data Format Reorganization

Consider a more I/O-friendly data organization:

```
# Options
1. Combine multiple basins into region-based shards
2. Create preprocessed "mini-batch" files that align with training batches
3. Use memory-mapped formats for faster random access
```

### 4. Implement Asynchronous Prefetching

Hide I/O latency with prefetching:

```
# Implementation idea
1. While GPU processes batch N, prefetch files for batch N+2
2. Use a dedicated I/O thread pool separate from worker threads
3. Prioritize prefetching based on batch sampling schedule
```

### 5. Optimize Data Transformations

Reduce conversion overhead:

```
# Approaches
1. Use zero-copy methods between Polars and PyTorch when possible
2. Pre-allocate tensors and fill them in-place
3. Process multiple samples at once in vectorized operations
```

## Implementation Strategy

1. **Profile First**: Use tools like `py-spy` or PyTorch Profiler to identify exact bottlenecks
2. **Start With Caching**: Implement file-level locks in `FileCache` for immediate gains
3. **Modify Batch Sampler**: Create "super-batches" to reduce file switching
4. **Long-Term**: Consider reorganizing your dataset format entirely

## Hardware Considerations

While optimizing code, consider these hardware options:

1. **Storage**: NVMe SSDs offer substantially better random I/O than SATA SSDs
2. **Memory**: Increasing RAM allows for larger cache sizes
3. **CPU**: More cores help with parallel data loading

## Final Thoughts

Your current architecture is optimized for basin diversity but not for I/O efficiency. The key insight is to maintain this diversity while dramatically reducing the number of separate file operations per GPU computation cycle.

The two-level batching strategy combined with an improved caching system should provide substantial gains without requiring a complete redesign of your data pipeline.

---

# Details on how to implement two stage training

Here’s a sketch of how you can recompute your 1 000‐basin chunks every 10 reloads, by integrating that logic into your LightningDataModule. You’ll:

- keep a master list of all 10 000 basins  
- on `setup()` build the initial shuffled chunks  
- in `train_dataloader()` recompute (reshuffle & re-chunk) whenever `current_epoch % recompute_every == 0`  

```python
from __future__ import annotations
import random
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning import LightningDataModule, Trainer
from .lazy_dataset import HydroLazyDataset
from dataclasses import dataclass, field

@dataclass
class HydroLazyDataModule(LightningDataModule):
    region_time_series_base_dirs: dict[str, Path]
    region_static_attributes_base_dirs: dict[str, Path]
    list_of_gauge_ids_to_process: list[str]
    batch_size: int
    input_length: int
    output_length: int
    files_per_batch: int
    num_workers: int
    chunk_size: int = 1000
    recompute_every: int = 10

    _all_basin_ids: list[str] = field(init=False)
    _chunks: list[list[str]] = field(init=False)

    def setup(self, stage: Optional[str] = None) -> None:
        # called once per fit; store and initial-shuffle all basin IDs
        self._all_basin_ids = self.list_of_gauge_ids_to_process.copy()
        self._recompute_chunks()

    def _recompute_chunks(self) -> None:
        """Shuffle full list and partition into chunk_size pieces."""
        random.shuffle(self._all_basin_ids)
        self._chunks = [
            self._all_basin_ids[i : i + self.chunk_size]
            for i in range(0, len(self._all_basin_ids), self.chunk_size)
        ]

    def train_dataloader(self) -> DataLoader:
        """Called every epoch when reload_dataloaders_every_n_epochs=1."""
        epoch = cast(int, self.trainer.current_epoch)
        # every `recompute_every` epochs, reshuffle & rebuild chunks
        if epoch % self.recompute_every == 0:
            self._recompute_chunks()

        # pick which chunk to load this epoch
        chunk_idx = epoch % len(self._chunks)
        basin_ids = self._chunks[chunk_idx]

        # load data & build a HydroLazyDataset for these basin_ids
        ds = HydroLazyDataset(
            region_time_series_dirs=self.region_time_series_base_dirs,
            region_static_dirs=self.region_static_attributes_base_dirs,
            gauge_ids=basin_ids,
            input_length=self.input_length,
            output_length=self.output_length,
            files_per_batch=self.files_per_batch,
            # … any other args …
        )
        sampler = RandomSampler(ds)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
```

Then in your training script:

```python
from pytorch_lightning import Trainer
trainer = Trainer(
    max_epochs=10,                      # 10 Lightning epochs → full 10 000-basin sweep
    reload_dataloaders_every_n_epochs=1 # reload every epoch/chunk
)
trainer.fit(model, datamodule=datamodule)
```

This way:

- each Lightning epoch loads 1 000 new basins  
- after 10 epochs you’ve covered all 10 000  
- every 10 epochs you reshuffle/repartition for a fresh random sweep
