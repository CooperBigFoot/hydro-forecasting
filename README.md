TODO

1. **Configuration Hashing/UUID Generation:** Write a function that takes the relevant configuration parameters from `HydroLazyDataModule` (like paths, split ratios, imputation settings, required columns, pipeline structure details, list of gauge IDs) and deterministically generates a unique identifier (e.g., a UUID derived from a stable hash of these parameters).
2. **Pipeline Persistence:** Implement helper functions to save a dictionary of scikit-learn `Pipeline` or `GroupedPipeline` objects to a file (e.g., using `joblib.dump`) and load them back.
3. **Configuration Persistence:** Implement helper functions to save the relevant subset of the `DataModule`'s configuration (the parameters used in step 1) to a human-readable file (like JSON) and load it back.
4. **Quality Report Persistence:** Ensure the `quality_report` dictionary can be easily saved to and loaded from a JSON file. This might already be straightforward.
5. **Define Output Structure:** Decide on a consistent directory structure within the main `path_to_preprocessing_output_directory`. For example, `.../output_dir/<run_uuid>/`, where `<run_uuid>` is the identifier from step 1. Inside this directory, define standard names for the saved files (e.g., `pipelines.joblib`, `config.json`, `quality_report.json`, `processed_data/`, `static_attributes.parquet`).
6. **Modify `run_hydro_processor` (or create a wrapper):** Adapt the `run_hydro_processor` function (or wrap it) so that *after* successful execution, it uses the functions from steps 2, 3, and 4 to save the fitted pipelines, the run configuration, and the quality report into the structured directory defined in step 5. It should probably also create a simple "success" marker file upon completion.
7. **Implement the Check-and-Load Logic:** Create a function (which will eventually be used within `prepare_data`) that:
    * Takes the `DataModule` configuration.
    * Generates the expected run UUID (using step 1).
    * Constructs the expected path to the run-specific output directory (using step 5).
    * Checks if this directory and the key files (pipelines, config, quality report, success marker) exist.
    * If they exist, it attempts to load the pipelines (step 2), config (step 3), and quality report (step 4).
    * Returns the loaded objects (or indicates failure if loading fails or files are missing).
8. **Integrate into `prepare_data`:** Replace the current unconditional call to `run_hydro_processor` in `prepare_data` with the logic from step 7. If the check-and-load function succeeds, populate the `DataModule` attributes (`self.fitted_pipelines`, `self.quality_report`, paths) from the loaded data. If it fails, call the modified `run_hydro_processor` (step 6) to perform the processing and save the results.

This breakdown allows you to develop and test each component (hashing, saving/loading pipelines, saving/loading config, the directory structure, the check-and-load logic) more independently before integrating them into the `DataModule`.

Okay, here are the prompts for an expert LLM coder to implement each building block, following your specified instructions and our plan.

---

**Prompt 1: Configuration Hashing/UUID Generation**

**Goal:** Implement a function that generates a deterministic UUID v5 based on a dictionary of configuration parameters relevant to data processing in the `HydroLazyDataModule`. This UUID will uniquely identify a specific processing run.

**File:** I created a new file: `src/hydro_forecasting/data/config_utils.py`

**Implementation Plan:**

1. Create the file `src/hydro_forecasting/data/config_utils.py`.
2. Add necessary imports: `json`, `hashlib`, `uuid`, `typing`, `pathlib`. Use Python 3.12 type hints (`dict`, `list`, etc.).
3. Define a constant for the UUID namespace, e.g., `PROCESSING_NAMESPACE = uuid.UUID("...")` (generate a fixed UUID once, e.g., using `uuid.uuid4()` and pasting it).
4. Define the function `generate_run_uuid(config: dict[str, typing.Any]) -> str`.
5. Inside the function:
    * Create a helper function `_make_hashable(obj: typing.Any) -> typing.Any` to recursively convert the config dictionary into a structure suitable for consistent hashing.
        * Convert dictionaries to sorted lists of key-value tuples: `sorted(obj.items())`.
        * Convert sets to sorted lists.
        * Convert `pathlib.Path` objects to strings.
        * Leave other basic types (str, int, float, bool, None, list, tuple) as is.
        * Recursively apply this to elements within lists/tuples and dictionary values.
    * Apply `_make_hashable` to the input `config` dictionary.
    * Serialize the hashable config representation into a JSON string using `json.dumps(hashable_config, sort_keys=True, ensure_ascii=False)`. Sorting keys is crucial for determinism.
    * Encode the JSON string to bytes: `json_string.encode('utf-8')`.
    * Calculate the SHA256 hash of the byte string: `hashlib.sha256(config_bytes).hexdigest()`.
    * Generate a UUID v5 using the predefined namespace and the SHA256 hash: `uuid.uuid5(PROCESSING_NAMESPACE, sha256_hash)`.
    * Return the UUID as a string: `str(run_uuid)`.
6. Add comprehensive Python 3.12 style type hints to the function signature and internal variables where appropriate.
7. Write a Google-style docstring explaining what the function does, its arguments, and what it returns. Mention the deterministic nature.

---

**Prompt 2: Pipeline Persistence**

**Goal:** Implement utility functions using `joblib` to save and load dictionaries containing scikit-learn `Pipeline` or `GroupedPipeline` objects. Use the `returns` library for error handling.

**File:** Add to `src/hydro_forecasting/data/config_utils.py`

**Implementation Plan:**

1. Open `src/hydro_forecasting/data/config_utils.py`.
2. Add necessary imports: `joblib`, `pathlib`, `typing`, `sklearn.pipeline`, `returns.result`. Import `Pipeline` from `sklearn.pipeline` and `GroupedPipeline` from `hydro_forecasting.preprocessing.grouped`. Use `Result`, `Success`, `Failure`.
3. Define the function `save_pipelines(pipelines: dict[str, typing.Union[Pipeline, GroupedPipeline]], filepath: pathlib.Path) -> Result[None, str]`.
4. Inside `save_pipelines`:
    * Use a `try...except` block to catch potential `IOError` or other exceptions during saving.
    * Call `filepath.parent.mkdir(parents=True, exist_ok=True)` to ensure the directory exists.
    * Call `joblib.dump(pipelines, filepath)`.
    * If successful, return `Success(None)`.
    * If an exception occurs, return `Failure(f"Failed to save pipelines to {filepath}: {e}")`.
5. Add type hints and a Google-style docstring.
6. Define the function `load_pipelines(filepath: pathlib.Path) -> Result[dict[str, typing.Union[Pipeline, GroupedPipeline]], str]`.
7. Inside `load_pipelines`:
    * Use a `try...except` block to catch `FileNotFoundError`, `IOError`, `joblib.externals.loky.process_executor.TerminatedWorkerError` (or general unpickling errors).
    * Check if the file exists: `if not filepath.is_file(): return Failure(f"Pipeline file not found: {filepath}")`.
    * Call `loaded_pipelines = joblib.load(filepath)`.
    * Perform a basic type check if possible (e.g., check if it's a dict and maybe check the type of the first value if the dict is not empty). Return `Failure` if the loaded object is not of the expected type.
    * If successful, return `Success(loaded_pipelines)`.
    * If an exception occurs, return `Failure(f"Failed to load pipelines from {filepath}: {e}")`.
8. Add type hints and a Google-style docstring.

---

**Prompt 3: Configuration Persistence**

**Goal:** Implement utility functions to save and load Python dictionaries (representing relevant `DataModule` configuration) to/from JSON files. Use the `returns` library for error handling. Also, define a helper to extract the relevant configuration subset.

**File:** Add to `src/hydro_forecasting/data/config_utils.py` and potentially modify lazy_datamodule.py for the extraction helper.

**Implementation Plan:**

1. Open `src/hydro_forecasting/data/config_utils.py`.
2. Add necessary imports: `json`, `pathlib`, `typing`, `returns.result`.
3. Define a helper function `_default_serializer(obj: typing.Any) -> str` to handle non-standard JSON types like `pathlib.Path`. If `isinstance(obj, pathlib.Path)`, return `str(obj)`, otherwise raise `TypeError`.
4. Define the function `save_config(config: dict[str, typing.Any], filepath: pathlib.Path) -> Result[None, str]`.
5. Inside `save_config`:
    * Use a `try...except` block (catching `IOError`, `TypeError`).
    * Ensure the directory exists: `filepath.parent.mkdir(parents=True, exist_ok=True)`.
    * Open the file for writing: `with open(filepath, 'w', encoding='utf-8') as f:`.
    * Dump the config: `json.dump(config, f, indent=4, sort_keys=True, default=_default_serializer)`.
    * Return `Success(None)` on success.
    * Return `Failure(f"Failed to save config to {filepath}: {e}")` on error.
6. Add type hints and a Google-style docstring.
7. Define the function `load_config(filepath: pathlib.Path) -> Result[dict[str, typing.Any], str]`.
8. Inside `load_config`:
    * Use a `try...except` block (catching `FileNotFoundError`, `IOError`, `json.JSONDecodeError`).
    * Check if file exists: `if not filepath.is_file(): return Failure(...)`.
    * Open the file for reading: `with open(filepath, 'r', encoding='utf-8') as f:`.
    * Load the config: `loaded_config = json.load(f)`.
    * Check if `loaded_config` is a dictionary. Return `Failure` if not.
    * Return `Success(loaded_config)` on success.
    * Return `Failure(f"Failed to load config from {filepath}: {e}")` on error.
9. Add type hints and a Google-style docstring.
10. **In lazy_datamodule.py (or `config_utils.py`)**:
    * Define a function `extract_relevant_config(datamodule: 'HydroLazyDataModule') -> dict[str, typing.Any]`. (Use forward reference for `HydroLazyDataModule` if defined in `config_utils.py`).
    * This function should access the attributes of the `datamodule` instance that were identified as key for the UUID generation (Step 1: paths, splits, features, target, preprocessing keys, train years, imputation gap, gauge list etc.).
    * Return a dictionary containing these specific attributes and their values. Ensure consistency with the parameters considered in `generate_run_uuid`.

---

**Prompt 4: Quality Report Persistence**

**Goal:** Verify that the `QualityReport` TypedDict structure is JSON-serializable and ensure any `pathlib.Path` objects within it are handled correctly by the `save_config` function developed in Step 3.

**File:** Review preprocessing.py (where `QualityReport` is defined) and potentially update `save_config` in `src/hydro_forecasting/data/config_utils.py` if needed.

**Implementation Plan:**

1. Examine the `QualityReport` and `BasinQualityReport` TypedDict definitions in preprocessing.py.
2. Identify all data types used within these structures (e.g., `dict`, `list`, `str`, `int`, `float`, `bool`, `Optional`, `Path`).
3. Confirm that the `_default_serializer` helper function within `save_config` (from Step 3) correctly handles `pathlib.Path` by converting it to a string.
4. If any other non-standard JSON-serializable types are found (unlikely, but possible), update the `_default_serializer` to handle them appropriately or ensure they are converted before saving.
5. No new functions are needed; the goal is verification and potential adjustment of the existing `save_config` serializer. Add comments if necessary to clarify handling.

---

**Prompt 5: Define Output Structure**

**Goal:** Document the standard directory structure for storing processed data artifacts associated with a specific run configuration.

**File:** Create or update a documentation file, e.g., `docs/data_processing.md` or add a section to the main README.md.

**Implementation Plan:**

1. Create/Open the chosen documentation file.
2. Add a section titled "Processed Data Output Structure" or similar.
3. Describe the structure:
    * The base directory is specified by `path_to_preprocessing_output_directory` in the `HydroLazyDataModule`.
    * Each unique processing run, identified by its configuration UUID, will have its own subdirectory: `<path_to_preprocessing_output_directory>/<run_uuid>/`.
    * Inside the `<run_uuid>` directory, the following standard files and directories will be stored:
        * `config.json`: The subset of the `DataModule` configuration used for this run (saved using `save_config`).
        * `pipelines.joblib`: The fitted scikit-learn pipelines (saved using `save_pipelines`).
        * `quality_report.json`: The processing quality report (saved using `save_config`).
        * `processed_timeseries/`: A directory containing the processed time series data (e.g., Parquet files per basin).
        * `processed_static_attributes.parquet`: (Optional) The processed static attributes file, if used.
        * `_SUCCESS`: An empty file acting as a marker to indicate that the processing run completed successfully and all artifacts were saved.
4. Explain that `<run_uuid>` is generated deterministically using the function from Step 1.

---

**Prompt 6: Modify `run_hydro_processor` (or Wrapper)**

**Goal:** Modify the `run_hydro_processor` function (or create a wrapper around it) to save its outputs (fitted pipelines, quality report, relevant config) and the processed data files into the standardized run-specific directory structure upon successful completion. It should also create the `_SUCCESS` marker file. The function should return a `Result` object.

**File:** Modify preprocessing.py or create a wrapper function. Let's proceed with modifying `run_hydro_processor`.

**Implementation Plan:**

1. Open preprocessing.py.
2. Import necessary functions from `config_utils`: `save_config`, `save_pipelines`. Import `Result`, `Success`, `Failure` from `returns.result`. Import `pathlib`.
3. Modify the `run_hydro_processor` function signature:
    * Add parameters: `run_uuid: str`, `datamodule_config: dict[str, typing.Any]`.
    * Change the return type annotation to `-> Result[ProcessingResult, str]`. The `ProcessingResult` TypedDict might need adjustment if paths change.
4. Inside `run_hydro_processor`:
    * Determine the main output directory based on the input `path_to_preprocessing_output_directory`.
    * Define the run-specific output directory: `run_output_dir = Path(path_to_preprocessing_output_directory) / run_uuid`.
    * **Crucially**: Adjust the internal logic that generates output paths (e.g., for processed Parquet files, reports) to place them *inside* `run_output_dir`. For example, the `processed_dir` passed to `process_basins_parallel` should likely be `run_output_dir / "processed_timeseries"`. The `reports_dir` might also go inside `run_output_dir`. Update the `ProcessingResult` TypedDict fields (`processed_dir`, `processed_path_to_static_attributes_directory`) to reflect these new locations relative to the base output directory or as absolute paths within the run directory.
    * Wrap the main processing logic (fitting pipelines, processing basins in parallel) in error handling that can return a `Failure`.
    * **After** the core processing successfully completes and `fitted_pipelines`, `quality_report`, and the final `ProcessingResult` dictionary are available:
        * Create the `run_output_dir`: `run_output_dir.mkdir(parents=True, exist_ok=True)`.
        * Define paths for saving artifacts: `config_path = run_output_dir / "config.json"`, `pipelines_path = run_output_dir / "pipelines.joblib"`, `report_path = run_output_dir / "quality_report.json"`, `success_marker_path = run_output_dir / "_SUCCESS"`.
        * Use a `returns.pipeline.pipeline` or `.bind` chain for saving:
            * `save_config(datamodule_config, config_path)`
            * `.bind(lambda _: save_pipelines(fitted_pipelines, pipelines_path))`
            * `.bind(lambda _: save_config(quality_report, report_path))` # Ensure quality_report is serializable (paths as strings)
            * `.map(lambda _: success_marker_path.touch())`
            * `.map(lambda _: final_processing_result)` # Pass the result dict through on success
        * If the saving pipeline succeeds, return `Success(final_processing_result)`.
        * If any saving step fails, return the `Failure` object generated by the save functions.
    * Ensure any intermediate error conditions during processing also return appropriate `Failure` objects.

---

**Prompt 7: Implement Check-and-Load Logic**

**Goal:** Implement a private method within `HydroLazyDataModule` that checks if processed data exists for the current configuration (using the run UUID) and, if valid, loads the quality report, fitted pipelines, and relevant paths.

**File:** Add a method to lazy_datamodule.py.

**Implementation Plan:**

1. Open lazy_datamodule.py.
2. Import necessary functions from `config_utils`: `generate_run_uuid`, `load_config`, `load_pipelines`, `extract_relevant_config`. Import `Result`, `Success`, `Failure`, `pipeline` from `returns.result` and `returns.pipeline`. Import `pathlib`, `typing`. Define `LoadedData` TypedDict: `class LoadedData(typing.TypedDict): quality_report: QualityReport; fitted_pipelines: dict[str, typing.Union[Pipeline, GroupedPipeline]]; processed_ts_dir: pathlib.Path; processed_static_path: typing.Optional[pathlib.Path]`.
3. Define a private method `_check_and_load_processed_data(self) -> Result[LoadedData, str]`.
4. Inside the method:
    * Get the relevant config subset: `relevant_config = extract_relevant_config(self)`.
    * Generate the run UUID: `run_uuid = generate_run_uuid(relevant_config)`.
    * Construct the run output directory path: `run_output_dir = self.path_to_preprocessing_output_directory / run_uuid`.
    * Define expected artifact paths: `config_path = run_output_dir / "config.json"`, `pipelines_path = run_output_dir / "pipelines.joblib"`, `report_path = run_output_dir / "quality_report.json"`, `success_marker_path = run_output_dir / "_SUCCESS"`, `processed_ts_dir = run_output_dir / "processed_timeseries"`, `processed_static_path_candidate = run_output_dir / "processed_static_attributes.parquet"`.
    * Check for existence of the run directory and the success marker: `if not run_output_dir.is_dir() or not success_marker_path.is_file(): return Failure(f"Valid processed data not found for run {run_uuid}")`.
    * Use `@pipeline(Result[LoadedData, str])` decorator for cleaner chaining:
        * `loaded_config = yield load_config(config_path)`
        * `loaded_report = yield load_config(report_path)` # This loads the dict, needs type casting later
        * `loaded_pipelines = yield load_pipelines(pipelines_path)`
        * Check if `processed_ts_dir` exists: `if not processed_ts_dir.is_dir(): return Failure(...)`.
        * Check if static file exists, set to Path or None: `processed_static_path = processed_static_path_candidate if processed_static_path_candidate.is_file() else None`.
        * **(Optional but recommended):** Compare `loaded_config` with `relevant_config` to ensure they match exactly. Return `Failure` if mismatch.
        * Construct the `LoadedData` dictionary, potentially casting `loaded_report` to `QualityReport` if needed (though runtime casting isn't standard for TypedDicts, ensure structure matches). Convert any path strings loaded from JSON back to `pathlib.Path` objects within the report if necessary.
        * Return `Success(LoadedData(...))` containing the loaded artifacts and paths.
5. Add type hints and a Google-style docstring.

---

**Prompt 8: Integrate into `prepare_data`**

**Goal:** Modify the `prepare_data` method in `HydroLazyDataModule` to use the `_check_and_load_processed_data` logic. If loading succeeds, skip running the processor and use the loaded artifacts. Otherwise, call the modified `run_hydro_processor` to generate and save the data.

**File:** Modify lazy_datamodule.py.

**Implementation Plan:**

1. Open lazy_datamodule.py.
2. Import `run_hydro_processor` from `preprocessing`, `generate_run_uuid`, `extract_relevant_config` from `config_utils`. Import `is_successful` from `returns.pipeline`.
3. In the `prepare_data` method:
    * Add the check at the beginning: `if self._prepare_data_has_run: return`.
    * Call the check-and-load method: `load_result = self._check_and_load_processed_data()`.
    * Check the result: `if is_successful(load_result):`.
        * Get the loaded data: `loaded_data = load_result.unwrap()`.
        * Assign loaded artifacts to instance attributes:
            * `self.quality_report = loaded_data['quality_report']`
            * `self.fitted_pipelines = loaded_data['fitted_pipelines']`
            * `self.processed_time_series_dir = loaded_data['processed_ts_dir']`
            * `self.processed_static_attributes_path = loaded_data['processed_static_path']`
        * Log an informative message (e.g., "Using existing processed data found in ...").
    * Else (if `load_result` is a `Failure`):
        * Log the failure reason: `log.warning(f"Failed to load existing data: {load_result.failure()}. Running preprocessing.")`.
        * Get relevant config: `relevant_config = extract_relevant_config(self)`.
        * Generate run UUID: `run_uuid = generate_run_uuid(relevant_config)`.
        * Prepare arguments for `run_hydro_processor` (paths, required columns, preprocessing config dict, splits, etc., taken from `self`).
        * Call the modified processor: `process_result = run_hydro_processor(..., run_uuid=run_uuid, datamodule_config=relevant_config)`.
        * Check the processor result: `if not is_successful(process_result):`.
            * Raise an exception or log a fatal error: `raise RuntimeError(f"Data processing failed: {process_result.failure()}")`.
        * If processing succeeded:
            * Get the results dict: `processing_outputs = process_result.unwrap()`.
            * Assign results to instance attributes:
                * `self.quality_report = processing_outputs['quality_report']`
                * `self.fitted_pipelines = processing_outputs['fitted_pipelines']`
                * `self.processed_time_series_dir = processing_outputs['processed_dir']` # Ensure this path is correct after Step 6 changes
                * `self.processed_static_attributes_path = processing_outputs['processed_path_to_static_attributes_directory']` # Ensure this path is correct
            * Log a message indicating processing was completed.
    * **After** either loading or successful processing:
        * Proceed with the rest of the original `prepare_data` logic (e.g., `create_index_entries`, `split_index_entries_by_stage`). Ensure this logic now correctly uses `self.processed_time_series_dir`, `self.quality_report`, etc.
        * Set the flag at the very end: `self._prepare_data_has_run = True`.

---
