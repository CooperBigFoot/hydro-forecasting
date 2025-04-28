## 1. Problem description

Current workflow:  

1. Phase 1: Batch‐read raw Parquet → extract raw training splits via split_data → fit all pipelines (on untrimmed, uninterpolated data) → discard.  
2. Phase 2: Batch‐read raw again → per‐basin quality checks (trim window, drop if too short, impute small gaps) → transform with pre‐fitted pipelines → write outputs.

Why it’s wrong:  
Pipelines see uncleaned data (with NaN edges and long gaps).
Quality checks only happen after fitting → transformer parameters don’t match the cleaned distributions.
Double I/O: two passes over the same source files.

---

## 2. Desired behavior

A single‐pass, per‐batch loop that for each batch of basins:

Read raw Parquet for that batch.
Clean & split each basin (trim, drop short, impute, then split_data into train/val/test).
Fit your GroupedPipeline instances directly on this batch’s cleaned train data (the GroupedPipeline.fit API will internally clone and remember per‐basin sub-pipelines).
Transform each basin’s cleaned train/val/test via apply_transformations.
Write processed Parquet and JSON report to disk.
Free memory and move to next batch.
At the end, process static attributes (if any), and save out the final pipelines.joblib, global quality_report.json, and_SUCCESS marker.

---

def run_hydro_processor(
    region_time_series_base_dirs: dict[str, Path],
    region_static_attributes_base_dirs: dict[str, Path],
    path_to_preprocessing_output_directory: Union[str, Path],
    required_columns: list[str],
    run_uuid: str,
    datamodule_config: dict[str, Any],
    preprocessing_config: Optional[dict[str, dict[str, Any]]] = None,
    min_train_years: float = 5.0,
    max_imputation_gap_size: int = 5,
    group_identifier: str = "gauge_id",
    train_prop: float = 0.5,
    val_prop: float = 0.25,
    test_prop: float = 0.25,
    processes: int = 6,
    list_of_gauge_ids_to_process: Optional[list[str]] = None,
    pipeline_fitting_batch_size: int = 50,
) -> tuple[bool, Optional[ProcessingResult], Optional[str]]:
    """
    Main function to run the hydrological data processor with pipeline fitting, supporting multi-region.

    Args:
        region_time_series_base_dirs: Mapping from region prefix to time series directory
        region_static_attributes_base_dirs: Mapping from region prefix to static attribute directory
        path_to_preprocessing_output_directory: Base directory for processed data
        required_columns: List of required data columns for quality checking
        run_uuid: Unique identifier for this processing run
        datamodule_config: Configuration dictionary for the data module
        preprocessing_config: Configuration for data preprocessing pipelines
        min_train_years: Minimum required years for training
        max_imputation_gap_size: Maximum gap length to impute with interpolation
        group_identifier: Column name identifying the basin
        train_prop: Proportion of data for training
        val_prop: Proportion of data for validation
        test_prop: Proportion of data for testing
        processes: Number of parallel processes to use
        list_of_gauge_ids_to_process: List of basin (gauge) IDs to process
        pipeline_fitting_batch_size: Maximum number of basins to process in a single batch

    Returns:
        Tuple containing (success flag, ProcessingResult on success, error message on failure)
    """
    # Validate input parameters
    if not list_of_gauge_ids_to_process:
        return False, None, "No gauge IDs provided for processing"

    # Setup paths and directories
    try:
        print("\n================ STARTING PREPROCESSING PIPELINE ================")

        # Create the run-specific output directory
        base_output_dir = Path(path_to_preprocessing_output_directory)
        run_output_dir = base_output_dir / run_uuid

        # Define subdirectories for organization
        processed_timeseries_dir = run_output_dir / "processed_timeseries"
        quality_reports_dir = run_output_dir / "quality_reports"

        # Define paths for artifacts
        config_path = run_output_dir / "config.json"
        pipelines_path = run_output_dir / "pipelines.joblib"
        quality_report_path = run_output_dir / "quality_report.json"
        success_marker_path = run_output_dir / "_SUCCESS"

        # Define path for processed static attributes
        processed_static_attributes_path = (
            run_output_dir / "processed_static_attributes.parquet"
        )

        # Create the config object
        config = Config(
            required_columns=required_columns,
            preprocessing_config=preprocessing_config,
            min_train_years=min_train_years,
            max_imputation_gap_size=max_imputation_gap_size,
            group_identifier=group_identifier,
            train_prop=train_prop,
            val_prop=val_prop,
            test_prop=test_prop,
        )

        # Create output directories
        processed_timeseries_dir.mkdir(parents=True, exist_ok=True)
        quality_reports_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, None, f"Failed to set up preprocessing directories: {str(e)}"

    # Define a function to process static attributes
    def process_static_data(
        static_df: pd.DataFrame, quality_report: QualityReport, fitted_pipelines: dict
    ) -> Optional[Path]:
        """Process static attributes and return the path if successful."""
        if static_df is None or "static" not in fitted_pipelines:
            return None

        print("\n================ PROCESSING STATIC ATTRIBUTES ================")
        print("INFO: Processing static attributes...")

        # Only keep static attributes for retained basins
        retained_basins = [
            basin_id
            for basin_id, report in quality_report["basins"].items()
            if basin_id not in quality_report["excluded_basins"] and report is not None
        ]

        if not retained_basins:
            print(
                "INFO: No basins were retained, skipping static attribute processing."
            )
            return None

        if "gauge_id" not in static_df.columns:
            print(
                "ERROR: 'gauge_id' column not found in loaded static_df, cannot filter or process."
            )
            return None

        filtered_static_df = static_df[
            static_df["gauge_id"].isin(retained_basins)
        ].copy()
        if filtered_static_df.empty:
            print(
                "INFO: No static attributes remaining after filtering for retained basins."
            )
            return None

        try:
            transformed_static = apply_transformations(
                filtered_static_df,
                config,
                fitted_pipelines,
                static_data=True,
            )
            processed_static_attributes_path.parent.mkdir(parents=True, exist_ok=True)
            transformed_static.to_parquet(processed_static_attributes_path)
            print(
                f"SUCCESS: Saved transformed static attributes for {len(transformed_static)} basins"
            )
            return processed_static_attributes_path
        except Exception as e:
            print(f"ERROR: Failed to transform or save static attributes: {str(e)}")
            return None

    # Main processing flow with explicit error handling
    try:
        # 1. Load static attributes
        static_df = load_static_attributes(
            region_static_attributes_base_dirs, list_of_gauge_ids_to_process
        )
        fitted_pipelines = {}
        
        # 2. Fit pipelines if preprocessing config is provided
        if preprocessing_config:
            try:
                fitted_pipelines = fit_pipelines(
                    static_df,
                    config,
                    list_of_gauge_ids_to_process,
                    region_time_series_base_dirs,
                    pipeline_fitting_batch_size=pipeline_fitting_batch_size,
                )
            except Exception as e:
                return False, None, f"Error during pipeline fitting: {str(e)}"
        
        # 3. Process basin time series
        try:
            quality_report = process_basins_parallel(
                list_of_gauge_ids_to_process,
                config,
                region_time_series_base_dirs,
                processed_timeseries_dir,
                quality_reports_dir,
                fitted_pipelines,
                processes,
            )
        except Exception as e:
            return False, None, f"Failed to process basin time series: {str(e)}"
        
        # 4. Process static attributes if available
        processed_static_path = process_static_data(
            static_df, quality_report, fitted_pipelines
        )
        
        # 5. Save artifacts
        success, _, error = save_config(datamodule_config, config_path)
        if not success:
            return False, None, f"Failed to save datamodule config: {error}"
            
        success, _, error = save_pipelines(fitted_pipelines, pipelines_path)
        if not success:
            return False, None, f"Failed to save pipelines: {error}"
            
        success, _, error = save_config(quality_report, quality_report_path)
        if not success:
            return False, None, f"Failed to save quality report: {error}"
        
        # Create success marker file
        success_marker_path.touch()
        
        # 6. Create and return final result
        result = {
            "quality_report": quality_report,
            "fitted_pipelines": fitted_pipelines,
            "run_output_dir": run_output_dir,
            "processed_timeseries_dir": processed_timeseries_dir,
            "processed_static_attributes_path": processed_static_path,
        }
        
        # Print summary
        print(
            "\n================ PROCESSING SUMMARY ================\n"
            f"SUCCESS: Completed processing {result['quality_report']['retained_basins']} "
            f"of {result['quality_report']['original_basins']} basins"
        )
        
        if result["quality_report"]["excluded_basins"]:
            print(
                f"WARNING: {len(result['quality_report']['excluded_basins'])} basins excluded due to quality issues"
            )
        
        return True, cast(ProcessingResult, result), None
        
    except Exception as e:
        return False, None, f"Unexpected error during hydro processing: {str(e)}"
