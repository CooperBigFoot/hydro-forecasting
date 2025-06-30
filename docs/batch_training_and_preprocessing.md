# Batch Training and Data Preprocessing Technical Guide

This document provides an in-depth technical description of the data processing and batch training pipeline used in the hydro-forecasting project. The system is designed to efficiently process large datasets by handling multiple basins (gauges) in batches, applying data quality checks, cleaning, and preprocessing, and finally feeding the data into a PyTorch Lightning model using an in-memory chunking mechanism.

## 1. Overview

The pipeline can be broken down into two main phases:

1. **Offline Preprocessing (`run_hydro_processor`)**: This phase is executed once per unique data configuration. It takes raw time series and attribute data, cleans it, validates it, applies transformations (like scaling), and saves the processed data and fitted transformers to disk. This phase is idempotent; given the same configuration, it will produce the same output, and it leverages caching to avoid re-computation.
2. **Online Data Loading (`HydroInMemoryDataModule`)**: This phase is executed during model training. The `DataModule` reads the pre-processed data from disk in a memory-efficient way. It loads the validation set entirely into memory for speed and cycles through the training data in "chunks" to keep memory usage manageable.

The entire process is orchestrated by the `HydroInMemoryDataModule`, which acts as the main entry point.

---

## 2. Configuration and Reproducibility

At the heart of the system is a configuration-driven approach that ensures reproducibility.

### 2.1. Run UUID Generation

Before any processing begins, a unique, deterministic **`run_uuid`** is generated from the `DataModule`'s configuration using `generate_run_uuid` from `data/config_utils.py`.

- **Hashing**: The function takes the `DataModule`'s configuration dictionary, converts it into a hashable format (handling lists, dicts, and `Path` objects), and serializes it into a deterministic JSON string.
- **UUID v5**: This string is then used to generate a UUIDv5. This ensures that the **exact same configuration will always produce the same `run_uuid`**.

This `run_uuid` serves as the key for caching. The `DataModule`'s `prepare_data` method uses this UUID to create a dedicated output directory (`<output_dir>/<run_uuid>`). If this directory already exists and contains a `_SUCCESS` marker, the entire preprocessing step is skipped, and the cached artifacts are used instead.

### 2.2. Configuration Validation

Before processing, the `DataModule`'s configuration is rigorously validated by functions in `data/datamodule_validators.py` and `data/preprocessing_validation.py`. This includes:

- Validating parameter types and values (e.g., positive integers, valid proportions).
- Ensuring paths are correctly specified.
- Validating the structure and compatibility of the `preprocessing_configs`. This is a crucial step that checks:
  - Each pipeline has a `strategy` (`per_group` or `unified`) and a `pipeline` object.
  - The pipeline object is compatible with the chosen strategy (e.g., a `GroupedPipeline` for `per_group`).
  - All required columns in the data are covered by a transformation pipeline.
  - The target variable is not accidentally included in the feature transformations, which could cause data leakage.

---

## 3. Phase 1: Offline Preprocessing with `run_hydro_processor`

If no cached data for the current `run_uuid` is found, `HydroInMemoryDataModule.prepare_data()` calls `run_hydro_processor` from `data/preprocessing.py`. This is the main workhorse of the offline processing phase.

### 3.1. Pipeline Creation

The processor first instantiates the necessary transformation pipelines based on the `preprocessing_config`. The `create_pipeline` factory function determines whether to create a `GroupedPipeline` (which fits a separate transformer for each basin) or a `UnifiedPipeline` (which fits one transformer on data from multiple basins).

### 3.2. Pre-validation Stage

A key optimization is the **pre-validation stage**. Instead of attempting to process all basins and having some fail midway, the system first performs a dry run to check data quality for all specified basins.

- `load_basins_timeseries_lazy` loads all specified basins in batches.
- `validate_basin_quality` is called for each batch. This function checks if a basin has enough non-null data points in the target column to meet the `min_train_years` requirement *after* splitting.
- Basins that fail this check are excluded from further processing. A detailed `BasinQualityReport` is generated for every basin and saved as a JSON file.
- Only the list of `valid_gauge_ids` is passed to the next stages.

### 3.3. Unified Pipeline Fitting

If any `UnifiedPipeline`s are configured, they are fitted *before* the main batch processing loop.

- A representative subset of the `valid_gauge_ids` is selected (the size is configurable via `fit_on_n_basins`).
- Data for this subset is loaded, cleaned, and the training portion is extracted.
- The `fit` method of the `UnifiedPipeline` is called on this data. The single fitted pipeline is then stored in memory to be applied to all basins later.

### 3.4. Batch Processing Loop

The `run_hydro_processor` then iterates through the `valid_gauge_ids` in batches (`basin_batch_size`). For each batch:

1. **Load**: `load_basins_timeseries_lazy` loads the raw time series data for the basins in the current batch into a Polars `LazyFrame`.
2. **Clean**: `apply_cleaning_steps` is called. This function:
    - Trims leading and trailing nulls for all required columns on a per-basin basis.
    - Forward-fills short gaps (up to `max_imputation_gap_size`). Backward filling is avoided to prevent data leakage from the future.
3. **Split**: `split_data` divides the cleaned data for each basin into training, validation, and test sets. **Crucially, the split is based on the proportions of non-null values in the target column**, ensuring that the training set contains a sufficient amount of valid target data.
4. **Fit (Grouped Pipelines)**: If `GroupedPipeline`s are used, their `fit` method is called *only on the training data (`train_df`)* from the current batch. This creates and fits a separate transformer for each basin in the batch.
5. **Transform**: The fitted pipelines (either the newly fitted `GroupedPipeline`s or the already-fitted `UnifiedPipeline`s) are used to transform the train, validation, and test dataframes for the current batch.
6. **Save Splits**: `write_train_val_test_splits_to_disk` saves the transformed data. For each basin, three separate Parquet files are created (e.g., `train/ky_123.parquet`, `val/ky_123.parquet`, `test/ky_123.parquet`). This granular structure is key to the online chunking mechanism.
7. **Merge Pipelines**: The fitted `GroupedPipeline`s from the batch are merged into the main pipeline object that persists across batches.

### 3.5. Finalization

After all batches are processed:

- The complete, fitted pipeline objects (containing all grouped transformers or the single unified transformer) are saved to disk (`.joblib` files).
- A `quality_summary.json` is created, aggregating the results from all the individual basin reports.
- A `_SUCCESS` file is created in the `run_uuid` directory, signaling that this run can be safely reused in the future.

---

## 4. Phase 2: Online Data Loading with `HydroInMemoryDataModule`

During `trainer.fit()`, the `DataModule`'s `setup()` and `*_dataloader()` methods are called.

### 4.1. `setup()` Method

The `setup` method prepares the dataloaders for training and validation.

1. **Identify Basins**: It scans the processed data directories (`<run_uuid>/processed_time_series/*`) to get the final lists of basin IDs available for the train, validation, and test splits.
2. **Load Static Data**: All processed static features are loaded into a dictionary (`static_data_cache`) mapping `basin_id` to its static feature tensor. This is kept in memory throughout training.
3. **Create Validation Pool**: A fixed subset of the training basins is selected to form the validation set (`_validation_gauge_id_pool`). The size is determined by `validation_chunk_size`.
4. **Cache Validation Set**: The *entire* validation set is loaded into memory by calling `_load_and_tensorize_chunk` on the `_validation_gauge_id_pool`. The resulting tensors, sequence indices, and metadata are cached in `_cached_validation_*` attributes. This ensures that validation is fast and does not involve disk I/O in every epoch.
5. **Partition Training Chunks**: The remaining training basins are partitioned into a list of chunks (`_shared_chunks`), where each chunk is a list of basin IDs. The `chunk_size` parameter controls how many basins are in each chunk.

### 4.2. `train_dataloader()` and Chunking

The training dataloader implements a memory-efficient chunking strategy.

1. **Advance Chunk**: At the beginning of each epoch (or more precisely, each time `train_dataloader` is called), the datamodule advances to the next chunk of basin IDs in `_shared_chunks`. If it reaches the end of the list, it reshuffles the basin IDs and creates a new set of chunks, ensuring data variability across full passes.
2. **Load and Tensorize Chunk**: It calls `_load_and_tensorize_chunk` with the basin IDs for the *current chunk*. This function:
    - Reads the corresponding processed `train/*.parquet` files from disk.
    - Concatenates them into a single, large Polars DataFrame.
    - Converts the DataFrame columns into a dictionary of PyTorch tensors (`chunk_column_tensors`).
    - Calls `find_valid_sequences` to identify all possible valid start/end indices for sequences within the chunk's data. This avoids having to check for NaNs on the fly during `__getitem__`.
    - Creates a `basin_row_map` to track the absolute start and end rows for each basin's data within the concatenated tensors.
3. **Instantiate Dataset**: An `InMemoryChunkDataset` is created with the in-memory tensors and indices for the current chunk.
4. **Return DataLoader**: A standard PyTorch `DataLoader` is returned, wrapping the chunk-specific dataset.

This process repeats, loading a new chunk of data from disk into memory whenever the previous one is exhausted.

### 4.3. `val_dataloader()`

The validation dataloader is much simpler. It uses the `_cached_validation_column_tensors` and `_cached_validation_index_entries` that were created once during the `setup` phase. It creates an `InMemoryChunkDataset` that points to these cached tensors, avoiding any disk reads during validation epochs.

### 4.4. `InMemoryChunkDataset.__getitem__`

This is where a single sample is constructed.

1. It receives an index `idx`.
2. It looks up the `(basin_id, start_row, end_row)` tuple from `self.index_entries`.
3. Using the `basin_row_map`, it calculates the absolute start and end indices within the large chunk tensors.
4. It slices the required `input_length` and `output_length` windows from the `chunk_column_tensors` (for features and target).
5. It retrieves the corresponding static feature tensor from the `static_data_cache` using the `basin_id`.
6. It assembles and returns a dictionary containing the `X`, `y`, `static`, and `future` tensors for the model.
