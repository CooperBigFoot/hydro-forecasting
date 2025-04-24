# Hydrological Data Processing

This document describes the data processing pipeline and output structure used in the hydro-forecasting project.

## Overview

The data processing pipeline handles:

- Loading raw time series data from multiple basins/catchments
- Loading and merging static basin attributes
- Applying preprocessing transformations
- Quality checking and filtering
- Splitting data into training, validation, and test sets
- Saving processed outputs in a structured format

## Processed Data Output Structure

Each data processing run produces a set of artifacts organized in a consistent directory structure. These artifacts are essential for reproducibility and model training.

### Directory Structure

The base directory is specified by `path_to_preprocessing_output_directory` in the `HydroLazyDataModule`. For each unique processing run, a subdirectory is created using a deterministic UUID based on the configuration:

```
<path_to_preprocessing_output_directory>/
└── <run_uuid>/
    ├── config.json
    ├── pipelines.joblib
    ├── quality_report.json
    ├── processed_timeseries/
    │   ├── basin_1.parquet
    │   ├── basin_2.parquet
    │   └── ... (one file per basin)
    ├── processed_static_attributes.parquet
    └── _SUCCESS
```

### File Descriptions

#### `config.json`

Contains the subset of the `HydroLazyDataModule` configuration used for this run. This file is essential for reproducibility as it records all parameters that affect data processing, including:

- Feature lists
- Target variable
- Input/output sequence lengths
- Preprocessing settings
- Train/val/test split proportions
- Domain information

The configuration is saved using the `save_config` utility function from `config_utils.py`.

#### `pipelines.joblib`

Contains the fitted scikit-learn preprocessing pipelines (or `GroupedPipeline` instances) used to transform the data. These pipelines can be loaded to:

- Apply the same transformations to new data
- Inverse-transform model predictions back to original scales

#### `quality_report.json`

A detailed report on the quality of the processed data, including:

- Total number of original and retained basins
- List of excluded basins with reasons for exclusion
- Basin-specific information:
  - Valid data periods
  - Processing steps applied
  - Imputation statistics

#### `processed_timeseries/`

Directory containing processed time series data, with one file per basin:

- Format: Parquet files (`.parquet`)
- Each file contains preprocessed time series with:
  - Date index
  - Transformed features
  - Transformed target
  - Basin identifier
  - Any additional metadata required for modeling

#### `processed_static_attributes.parquet`

A single Parquet file containing preprocessed static attributes for all basins:

- Each row corresponds to a unique basin
- Contains transformed static features used by models that incorporate catchment characteristics
- This file is optional and only present if static attributes were provided and processed

#### `_SUCCESS`

An empty marker file indicating that the processing run completed successfully and all artifacts were saved. The presence of this file can be used to verify that a processing run finished without errors.

### Run UUID Generation

The `<run_uuid>` is generated deterministically using the `generate_run_uuid` function in `config_utils.py`. The UUID is based on a hash of key configuration parameters, ensuring that:

1. Identical configurations produce the same UUID
2. Changes to the configuration produce different UUIDs
3. The output directory is uniquely identified by its configuration

This approach enables:

- Easy identification of processed data associated with specific configurations
- Detection of configuration changes
- Reuse of previously processed data when configurations match

## Using Processed Data

The processed data can be loaded using `HydroLazyDataModule`, which handles:

1. Reading the configuration from a specific run
2. Loading the appropriate data files
3. Setting up PyTorch datasets and dataloaders
4. Providing methods to inverse-transform model outputs

Example:

```python
datamodule = HydroLazyDataModule(
    path_to_preprocessing_output_directory="path/to/processed_data",
    # Other parameters...
)
datamodule.prepare_data()  # This will load from or create the run_uuid directory
datamodule.setup()
```

## Railway Oriented Programming Implementation

The data processing pipeline uses Railway Oriented Programming (ROP) through the `returns` library to handle errors in a functional way. The main implementation is in the `run_hydro_processor` function:

### Error Handling with Result Objects

The `run_hydro_processor` function returns a `Result` object that contains either:

- `Success` with the processing results
- `Failure` with an error message

This approach allows error handling to be more explicit and composable:

```python
processing_result = run_hydro_processor(...)

if isinstance(processing_result, Success):
    # Work with the successful result
    result_value = processing_result.unwrap()
    processed_dir = result_value["processed_timeseries_dir"]
    # Continue processing...
else:
    # Handle the error case
    error_message = processing_result.failure()
    print(f"Processing failed: {error_message}")
```

### Pipeline Operations

The function uses the `returns.pipeline` module to chain operations together, especially when saving artifacts:

```python
save_result = (
    save_config(datamodule_config, config_path)
    .bind(lambda _: save_pipelines(fitted_pipelines, pipelines_path))
    .bind(lambda _: save_config(quality_report, quality_report_path))
    .map(lambda _: success_marker_path.touch())
)
```

This ensures that if any step fails, the error is propagated through the pipeline without executing subsequent steps.
