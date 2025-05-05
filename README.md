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
