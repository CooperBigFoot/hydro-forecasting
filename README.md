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
