import pytorch_lightning as pl
from pathlib import Path
from typing import Union, Optional, Any
from torch.utils.data import DataLoader
import math
import torch
import numpy as np
import pandas as pd
from returns.result import Result, Success, Failure
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from .lazy_dataset import HydroLazyDataset
from .preprocessing import run_hydro_processor, ProcessingOutput, ProcessingConfig
from .index_entry_creator import create_index_entries
from ..preprocessing.grouped import GroupedPipeline
from .batch_sampler import FileGroupedBatchSampler
from .config_utils import (
    extract_relevant_config,
    save_config,
    generate_run_uuid,
    load_config,
)
from .clean_data import SummaryQualityReport
from ..preprocessing.static_preprocessing import load_static_pipeline
from ..preprocessing.time_series_preprocessing import load_time_series_pipelines


@dataclass
class LoadedData:
    """Data structure for loaded preprocessing artifacts."""

    summary_quality_report: SummaryQualityReport
    fitted_pipelines: dict[str, Union[Pipeline, GroupedPipeline]]
    processed_ts_dir: Path
    processed_static_path: Optional[Path]


def worker_init_fn(worker_id: int) -> None:
    """
    Adjust cache settings for each worker to avoid thrashing.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    ds.file_cache.max_memory_mb = 800


class HydroLazyDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule that lazily loads datasets.
    """

    def __init__(
        self,
        region_time_series_base_dirs: dict[str, Union[str, Path]],
        region_static_attributes_base_dirs: dict[str, Union[str, Path]],
        path_to_preprocessing_output_directory: Union[str, Union[str, Path]],
        group_identifier: str,
        batch_size: int,
        input_length: int,
        output_length: int,
        forcing_features: list[str],
        static_features: list[str],
        target: str,
        preprocessing_configs: dict[str, Union[Pipeline, GroupedPipeline]],
        num_workers: int,
        min_train_years: int,
        train_prop: float,
        val_prop: float,
        test_prop: float,
        max_imputation_gap_size: int,
        list_of_gauge_ids_to_process: Optional[list[str]] = None,
        domain_id: str = "source",
        domain_type: str = "source",
        is_autoregressive: bool = False,
        files_per_batch: int = 20,
    ):
        super().__init__()

        self.region_time_series_base_dirs = region_time_series_base_dirs
        self.region_static_attributes_base_dirs = region_static_attributes_base_dirs
        self.path_to_preprocessing_output_directory = Path(
            path_to_preprocessing_output_directory
        )
        self.group_identifier = group_identifier
        self.batch_size = batch_size
        self.input_length = input_length
        self.output_length = output_length
        self.forcing_features = forcing_features
        self.static_features = static_features
        self.target = target
        self.preprocessing_configs = preprocessing_configs
        self.num_workers = num_workers
        self.min_train_years = min_train_years
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.max_imputation_gap_size = max_imputation_gap_size
        self.list_of_gauge_ids_to_process = list_of_gauge_ids_to_process
        self.domain_id = domain_id
        self.domain_type = domain_type
        self.is_autoregressive = is_autoregressive
        self.files_per_batch = files_per_batch
        self._prepare_data_has_run: bool = False

        # Post initialization
        self.quality_report = {}
        self.fitted_pipelines = {}
        self.processed_time_series_dir = Path("")
        self.processed_static_attributes_path = None  # Now a file, not a dir
        self.index_entries = []
        self.index_entries_by_stage = {}
        self.train_index_entries = []
        self.val_index_entries = []
        self.test_index_entries = []

        # Dataset instances created in setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Validation using Railway Oriented Programming
        validation_result = self._validate_all()
        if isinstance(validation_result, Failure):
            raise ValueError(
                f"Invalid configuration for HydroLazyDataModule: {validation_result.failure()}"
            )

    def _validate_all(self) -> Result[None, str]:
        """Run all validations using railway pattern, returning Success or Failure."""
        return (
            # Start with a Success value to begin the railway
            Success(None)
            # Chain all validation steps with bind
            .bind(
                lambda _: self._validate_positive_integer("batch_size", self.batch_size)
            )
            .bind(
                lambda _: self._validate_positive_integer(
                    "input_length", self.input_length
                )
            )
            .bind(
                lambda _: self._validate_positive_integer(
                    "output_length", self.output_length
                )
            )
            .bind(
                lambda _: self._validate_positive_integer(
                    "num_workers", self.num_workers
                )
            )
            .bind(
                lambda _: self._validate_positive_integer(
                    "files_per_batch", self.files_per_batch
                )
            )
            .bind(
                lambda _: self._validate_non_negative_integer(
                    "max_imputation_gap_size", self.max_imputation_gap_size
                )
            )
            .bind(
                lambda _: self._validate_positive_float(
                    "min_train_years", self.min_train_years
                )
            )
            .bind(
                lambda _: self._validate_gauge_id_list(
                    "list_of_gauge_ids_to_process", self.list_of_gauge_ids_to_process
                )
            )
            .bind(
                lambda _: self._validate_string_list(
                    "forcing_features", self.forcing_features
                )
            )
            .bind(
                lambda _: self._validate_string_list(
                    "static_features", self.static_features
                )
            )
            .bind(lambda _: self._validate_non_empty_string("target", self.target))
            .bind(
                lambda _: self._validate_non_empty_string(
                    "group_identifier", self.group_identifier
                )
            )
            .bind(
                lambda _: self._validate_non_empty_string("domain_id", self.domain_id)
            )
            .bind(lambda _: self._validate_domain_type("domain_type", self.domain_type))
            .bind(
                lambda _: self._validate_preprocessing_config(
                    self.preprocessing_configs
                )
            )
            .bind(lambda _: self._validate_train_val_test_prop())
            .bind(lambda _: self._validate_target_not_in_forcing_features())
        )

    def _validate_positive_integer(
        self, param_name: str, value: int
    ) -> Result[int, str]:
        """
        Validate that a parameter is a positive integer.

        Args:
            param_name: Name of the parameter being validated
            value: Value to validate

        Returns:
            Success with the value if valid, or Failure with error message
        """
        if not isinstance(value, int):
            return Failure(
                f"Parameter '{param_name}' must be an integer, got {type(value).__name__}"
            )
        if value <= 0:
            return Failure(
                f"Parameter '{param_name}' must be greater than 0, got {value}"
            )
        return Success(value)

    def _validate_non_negative_integer(
        self, param_name: str, value: int
    ) -> Result[int, str]:
        """
        Validate that a parameter is a non-negative integer.

        Args:
            param_name: Name of the parameter being validated
            value: Value to validate

        Returns:
            Success with the value if valid, or Failure with error message
        """
        if not isinstance(value, int):
            return Failure(
                f"Parameter '{param_name}' must be an integer, got {type(value).__name__}"
            )
        if value < 0:
            return Failure(
                f"Parameter '{param_name}' must be greater than or equal to 0, got {value}"
            )
        return Success(value)

    def _validate_positive_float(
        self, param_name: str, value: float
    ) -> Result[float, str]:
        """
        Validate that a parameter is a positive float.

        Args:
            param_name: Name of the parameter being validated
            value: Value to validate

        Returns:
            Success with the value if valid, or Failure with error message
        """
        if not isinstance(value, (int, float)):
            return Failure(
                f"Parameter '{param_name}' must be a number, got {type(value).__name__}"
            )
        if value <= 0:
            return Failure(
                f"Parameter '{param_name}' must be greater than 0, got {value}"
            )
        return Success(value)

    def _validate_gauge_id_list(
        self, param_name: str, value: Optional[list[str]]
    ) -> Result[Optional[list[str]], str]:
        """
        Validate that a parameter is a list of gauge IDs.

        Args:
            param_name: Name of the parameter being validated
            value: Value to validate

        Returns:
            Success with the value if valid, or Failure with error message
        """
        if value is None:
            return Success(None)
        if not isinstance(value, list):
            return Failure(
                f"Parameter '{param_name}' must be a list, got {type(value).__name__}"
            )
        if len(value) == 0:
            return Failure(f"Parameter '{param_name}' must not be an empty list")
        if not all(isinstance(item, str) for item in value):
            return Failure(f"All items in '{param_name}' must be strings")
        return Success(value)

    def _validate_string_list(
        self, param_name: str, value: list[str]
    ) -> Result[list[str], str]:
        """
        Validate that a parameter is a non-empty list of strings.

        Args:
            param_name: Name of the parameter being validated
            value: Value to validate

        Returns:
            Success with the value if valid, or Failure with error message
        """
        if not isinstance(value, list):
            return Failure(
                f"Parameter '{param_name}' must be a list, got {type(value).__name__}"
            )
        if len(value) == 0:
            return Failure(f"Parameter '{param_name}' must not be an empty list")
        if not all(isinstance(item, str) for item in value):
            return Failure(f"All items in '{param_name}' must be strings")
        return Success(value)

    def _validate_non_empty_string(
        self, param_name: str, value: str
    ) -> Result[str, str]:
        """
        Validate that a parameter is a non-empty string.

        Args:
            param_name: Name of the parameter being validated
            value: Value to validate

        Returns:
            Success with the value if valid, or Failure with error message
        """
        if not isinstance(value, str):
            return Failure(
                f"Parameter '{param_name}' must be a string, got {type(value).__name__}"
            )
        if len(value) == 0:
            return Failure(f"Parameter '{param_name}' must not be an empty string")
        return Success(value)

    def _validate_domain_type(self, param_name: str, value: str) -> Result[str, str]:
        """
        Validate that a parameter is one of the allowed domain types.

        Args:
            param_name: Name of the parameter being validated
            value: Value to validate

        Returns:
            Success with the value if valid, or Failure with error message
        """
        if not isinstance(value, str):
            return Failure(
                f"Parameter '{param_name}' must be a string, got {type(value).__name__}"
            )
        if value not in ["source", "target"]:
            return Failure(
                f"Parameter '{param_name}' must be either 'source' or 'target', got '{value}'"
            )
        return Success(value)

    def _validate_preprocessing_config(
        self, config: dict[str, dict[str, object]]
    ) -> Result[dict[str, dict[str, object]], str]:
        """
        Validate the preprocessing configuration.

        Args:
            config: The preprocessing configuration to validate

        Returns:
            Success with the config if valid, or Failure with error message
        """
        for data_type, cfg in config.items():
            if "pipeline" not in cfg:
                return Failure(f"Missing 'pipeline' key in {data_type} config")

            pipeline = cfg["pipeline"]
            if not isinstance(pipeline, (Pipeline, GroupedPipeline)):
                return Failure(
                    f"Pipeline for {data_type} must be Pipeline or GroupedPipeline, got {type(pipeline)}"
                )

            if isinstance(pipeline, GroupedPipeline):
                if pipeline.group_identifier != self.group_identifier:
                    return Failure(
                        f"GroupedPipeline for {data_type} uses group_identifier "
                        f"'{pipeline.group_identifier}' but data module uses "
                        f"'{self.group_identifier}'"
                    )

            if data_type == "static_features" and "columns" not in cfg:
                return Failure("static_features config must include 'columns' key")

            pipeline_result = self._validate_pipeline_compatibility(pipeline)
            if isinstance(pipeline_result, Failure):
                return pipeline_result

        return Success(config)

    def _validate_pipeline_compatibility(
        self, pipeline: Union["Pipeline", "GroupedPipeline"]
    ) -> Result[Union["Pipeline", "GroupedPipeline"], str]:
        """
        Validate that a pipeline is compatible with our requirements.

        Args:
            pipeline: The pipeline to validate

        Returns:
            Success with the pipeline if valid, or Failure with error message
        """
        if isinstance(pipeline, GroupedPipeline):
            pipeline = pipeline.pipeline

        for _, transformer in pipeline.steps:
            required_methods = ["fit", "transform", "inverse_transform"]
            missing = [m for m in required_methods if not hasattr(transformer, m)]
            if missing:
                return Failure(
                    f"Transformer {transformer.__class__.__name__} missing required methods: {missing}"
                )

        return Success(pipeline)

    def _validate_train_val_test_prop(self) -> Result[None, str]:
        """
        Validate that training, validation, and test proportions sum to 1.

        Returns:
            Success if valid, or Failure with error message
        """
        total_prop = math.fsum([self.train_prop, self.val_prop, self.test_prop])
        if not math.isclose(total_prop, 1.0, abs_tol=1e-6):
            return Failure(
                f"Training, validation, and test proportions must sum to 1. Current sum: {total_prop}"
            )
        return Success(None)

    def _validate_target_not_in_forcing_features(self) -> Result[None, str]:
        """
        Validate that the target variable is not included in the forcing features
        unless the model is autoregressive.

        Returns:
            Success if valid, or Failure with error message
        """
        if not self.is_autoregressive and self.target in self.forcing_features:
            return Failure(
                f"Target variable '{self.target}' should not be included in forcing features. Set is_autoregressive=True to include it."
            )
        return Success(None)

    def _check_and_reuse_existing_processed_data(
        self,
    ) -> tuple[bool, Optional[LoadedData], Optional[str]]:
        """
        Check if processed data exists for the current configuration and load it if valid.

        This method:
        1. Validates the folder structure for a matching run UUID
        2. Checks for the _SUCCESS file indicating complete processing
        3. Validates the DataModule configuration matches the stored configuration

        Returns:
            Tuple (success, LoadedData or None, error message or None)
        """
        try:
            # Extract configuration and generate deterministic UUID
            config = extract_relevant_config(self)
            run_uuid = generate_run_uuid(config)
            run_dir = self.path_to_preprocessing_output_directory / run_uuid

            # Check if the directory exists
            if not run_dir.exists():
                return (
                    False,
                    None,
                    f"No processed data found for this configuration (UUID: {run_uuid})",
                )

            # Validate folder structure - required files
            required_files = [
                "config.json",
                "quality_summary.json",
                "_SUCCESS",
                "fitted_time_series_pipelines.joblib",
            ]

            missing_files = [f for f in required_files if not (run_dir / f).exists()]
            if missing_files:
                return (
                    False,
                    None,
                    f"Incomplete processed data: missing files {missing_files}",
                )

            # Validate folder structure - required directories
            required_dirs = [
                "processed_time_series",
                "processed_time_series/train",
                "processed_time_series/val",
                "processed_time_series/test",
                "quality_reports",
            ]

            missing_dirs = [d for d in required_dirs if not (run_dir / d).is_dir()]
            if missing_dirs:
                return (
                    False,
                    None,
                    f"Incomplete processed data: missing directories {missing_dirs}",
                )

            # Load and validate configuration
            config_path = run_dir / "config.json"
            config_result = load_config(config_path)
            if isinstance(config_result, Failure):
                return (
                    False,
                    None,
                    f"Failed to load stored configuration: {config_result.failure()}",
                )

            loaded_config = config_result.unwrap()
            config_mismatch = self._validate_loaded_config(loaded_config, config)
            if config_mismatch:
                return False, None, f"Configuration mismatch: {config_mismatch}"

            # Load quality report
            summary_path = run_dir / "quality_summary.json"
            with open(summary_path, "r") as f:
                import json

                quality_report = json.load(f)

            # Set up fitted pipelines dictionary
            fitted_pipelines = {}

            # Load time series pipelines
            ts_pipelines_path = run_dir / "fitted_time_series_pipelines.joblib"
            ts_pipelines_result = load_time_series_pipelines(ts_pipelines_path)
            if isinstance(ts_pipelines_result, Failure):
                return (
                    False,
                    None,
                    f"Failed to load time series pipelines: {ts_pipelines_result.failure()}",
                )

            # Add time series pipelines to combined dictionary
            fitted_pipelines.update(ts_pipelines_result.unwrap())

            # Check for static pipeline if it exists
            static_pipeline_path = run_dir / "fitted_static_pipeline.joblib"
            processed_static_path = run_dir / "processed_static_features.parquet"

            if static_pipeline_path.exists() and processed_static_path.exists():
                # Load static pipeline
                static_pipeline_result = load_static_pipeline(static_pipeline_path)
                if isinstance(static_pipeline_result, Failure):
                    return (
                        False,
                        None,
                        f"Failed to load static pipeline: {static_pipeline_result.failure()}",
                    )

                # Add static pipeline to combined dictionary
                fitted_pipelines["static_features"] = static_pipeline_result.unwrap()
            else:
                # No static data
                processed_static_path = None

            # Construct LoadedData dataclass
            loaded_data = LoadedData(
                quality_report=quality_report,
                fitted_pipelines=fitted_pipelines,
                processed_ts_dir=run_dir / "processed_time_series",
                processed_static_path=processed_static_path,
            )

            return True, loaded_data, None

        except Exception as e:
            import traceback

            return (
                False,
                None,
                f"Error checking for existing processed data: {e}\n{traceback.format_exc()}",
            )

    def _validate_loaded_config(
        self, loaded_config: dict, current_config: dict
    ) -> Optional[str]:
        critical_keys = [
            "input_length",
            "output_length",
            "forcing_features",
            "static_features",
            "target",
            "min_train_years",
            "max_imputation_gap_size",
            "train_prop",
            "val_prop",
            "test_prop",
        ]
        for key in critical_keys:
            if key in loaded_config and key in current_config:
                if loaded_config[key] != current_config[key]:
                    return (
                        f"Configuration mismatch for key '{key}': "
                        f"loaded={loaded_config[key]}, current={current_config[key]}"
                    )
        return None

    def prepare_data(self) -> None:
        """
        Preprocesses the data if not already done or loaded.

        This method orchestrates the data preparation pipeline:
        1. Checks if preparation has already run.
        2. Defines required columns based on features and target.
        3. Attempts to reuse existing processed data via `_check_and_reuse_existing_processed_data`.
        4. If reuse is successful, loads state from `LoadedData`.
        5. If reuse fails or is not applicable, runs the new `run_hydro_processor`.
        6. Handles the `Result` from `run_hydro_processor`.
        7. Loads fitted pipelines from disk using paths from `ProcessingOutput`.
        8. Determines the list of successfully processed gauge IDs.
        9. Creates index entries for the processed data splits.
        10. Sets the `_prepare_data_has_run` flag.

        Raises:
            RuntimeError: If `run_hydro_processor` fails or if loading pipelines fails,
                          or if no basins remain after processing.
            ValueError: If index creation fails.
        """
        if self._prepare_data_has_run:
            print("INFO: Data preparation has already run.")
            return

        print("INFO: Starting data preparation...")

        required_columns = self.forcing_features + [self.target]
        processing_config = ProcessingConfig(
            required_columns=required_columns,
            preprocessing_config=self.preprocessing_configs,
            min_train_years=self.min_train_years,
            max_imputation_gap_size=self.max_imputation_gap_size,
            group_identifier=self.group_identifier,
            train_prop=self.train_prop,
            val_prop=self.val_prop,
            test_prop=self.test_prop,
        )

        # 1. Attempt to reuse existing processed data
        reuse_success, loaded_data, reuse_message = (
            self._check_and_reuse_existing_processed_data()
        )

        processed_gauge_ids: list[str] = []

        if reuse_success and loaded_data:
            print("INFO: Reusing existing processed data.")

            self.summary_quality_report = loaded_data.summary_quality_report
            self.fitted_pipelines = loaded_data.fitted_pipelines
            self.processed_time_series_dir = loaded_data.processed_ts_dir
            self.processed_static_attributes_path = loaded_data.processed_static_path
            processed_gauge_ids = loaded_data.processed_gauge_ids

            if self.list_of_gauge_ids_to_process is None:
                self.list_of_gauge_ids_to_process = processed_gauge_ids
            else:
                self.list_of_gauge_ids_to_process = [
                    gid
                    for gid in self.list_of_gauge_ids_to_process
                    if gid in processed_gauge_ids
                ]

            print(
                f"INFO: Loaded {len(self.fitted_pipelines)} pipelines and data for {len(processed_gauge_ids)} basins."
            )

        else:
            print(
                f"INFO: No reusable data found or reuse failed. Reason: {reuse_message}. Running preprocessing..."
            )
            # 2. Run the new hydro processor
            relevant_config = extract_relevant_config(self)
            run_uuid = generate_run_uuid(relevant_config)
            print(f"INFO: Generated Run UUID: {run_uuid}")

            processing_result: Result[ProcessingOutput, str] = run_hydro_processor(
                region_time_series_base_dirs=self.region_time_series_base_dirs,
                region_static_attributes_base_dirs=self.region_static_attributes_base_dirs,
                path_to_preprocessing_output_directory=self.path_to_preprocessing_output_directory,
                required_columns=required_columns,
                run_uuid=run_uuid,
                datamodule_config=relevant_config,  # Pass relevant config for saving
                preprocessing_config=self.preprocessing_configs,  # Pass pipeline definitions
                min_train_years=self.min_train_years,
                max_imputation_gap_size=self.max_imputation_gap_size,
                group_identifier=self.group_identifier,
                train_prop=self.train_prop,
                val_prop=self.val_prop,
                test_prop=self.test_prop,
                list_of_gauge_ids_to_process=self.list_of_gauge_ids_to_process,
                basin_batch_size=50,  # TODO: Rethink this. The value should not be hardcoded
            )

            if isinstance(processing_result, Failure):
                raise RuntimeError(
                    f"Hydro processor failed: {processing_result.failure()}"
                )

            processing_output = processing_result.unwrap()
            print("INFO: Hydro processor completed successfully.")

            # 3. Update DataModule state from ProcessingOutput
            self.summary_quality_report = processing_output.summary_quality_report
            self.processed_time_series_dir = processing_output.processed_timeseries_dir
            self.processed_static_attributes_path = (
                processing_output.processed_static_attributes_path
            )

            # 4. Load fitted pipelines from disk
            print("INFO: Loading fitted pipelines...")
            ts_pipelines_result = load_time_series_pipelines(
                processing_output.fitted_time_series_pipelines_path
            )
            if isinstance(ts_pipelines_result, Failure):
                raise RuntimeError(
                    f"Failed to load time series pipelines: {ts_pipelines_result.failure()}"
                )
            self.fitted_pipelines = ts_pipelines_result.unwrap()
            print(f"INFO: Loaded {len(self.fitted_pipelines)} time series pipelines.")

            if processing_output.fitted_static_pipeline_path:
                static_pipeline_result = load_static_pipeline(
                    processing_output.fitted_static_pipeline_path
                )
                if isinstance(static_pipeline_result, Failure):
                    raise RuntimeError(
                        f"Failed to load static pipeline: {static_pipeline_result.failure()}"
                    )
                self.fitted_pipelines["static_features"] = (
                    static_pipeline_result.unwrap()
                )
                print("INFO: Loaded static features pipeline.")
            else:
                print("INFO: No static pipeline path found in processing output.")

            # 5. Determine the list of successfully processed gauge IDs
            if self.summary_quality_report:
                processed_gauge_ids = self.summary_quality_report.retained_basins
                # Update the datamodule's list to reflect only processed basins
                self.list_of_gauge_ids_to_process = processed_gauge_ids
                print(
                    f"INFO: {len(processed_gauge_ids)} basins retained after processing."
                )
            else:
                # This case should ideally not happen if processing succeeded
                print("WARNING: Summary quality report is missing after processing.")
                processed_gauge_ids = []
                self.list_of_gauge_ids_to_process = []

            if not processed_gauge_ids:
                raise RuntimeError("No basins remained after the preprocessing stage.")

        # 6. Create index entries for the processed data
        print("INFO: Creating index entries...")
        if not self.processed_time_series_dir:
            # This should not happen if processing or loading succeeded
            raise RuntimeError(
                "Processed time series directory is not set before creating index."
            )

        index_output_dir = (
            self.path_to_preprocessing_output_directory / run_uuid / "index"
        )
        index_output_dir.mkdir(parents=True, exist_ok=True)

        index_result = create_index_entries(
            gauge_ids=processed_gauge_ids,  # Use the final list
            time_series_base_dir=self.processed_time_series_dir,
            output_dir=index_output_dir,
            input_length=self.input_length,
            output_length=self.output_length,
            static_file_path=self.processed_static_attributes_path,
            processing_config=processing_config,
        )

        if isinstance(index_result, Failure):
            raise ValueError(
                f"Failed to create index entries: {index_result.failure()}"
            )

        self.index_paths = index_result.unwrap()
        self.train_index_path = self.index_paths.get("train")
        self.val_index_path = self.index_paths.get("val")
        self.test_index_path = self.index_paths.get("test")
        print(f"INFO: Index entries created successfully at {index_output_dir}.")

        # 7. Set the flag
        self._prepare_data_has_run = True
        print("INFO: Data preparation finished.")

    def setup(self, stage: Optional[str] = None):
        """
        Create datasets for training, validation, and testing.

        This method is called by PyTorch Lightning to set up datasets
        for each stage of training.

        Args:
            stage: Stage of training ('fit', 'validate', 'test', or None for all)
        """
        # Ensure prepare_data has been called
        if (
            not hasattr(self, "processed_time_series_dir")
            or not self.processed_time_series_dir
        ):
            raise RuntimeError("prepare_data() must be called before setup()")

        static_file_path = self.processed_static_attributes_path

        # Common dataset arguments
        common_args = {
            "target": self.target,
            "forcing_features": self.forcing_features,
            "static_features": self.static_features,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "group_identifier": self.group_identifier,
            "domain_id": self.domain_id,
            "domain_type": self.domain_type,
            "is_autoregressive": self.is_autoregressive,
        }

        if stage == "fit" or stage is None:
            self.train_dataset = HydroLazyDataset(
                index_file_path=self.train_index_path,
                index_meta_file_path=self.train_index_meta_path,
                **common_args,
            )
            self.val_dataset = HydroLazyDataset(
                index_file_path=self.val_index_path,
                index_meta_file_path=self.val_index_meta_path,
                **common_args,
            )
            print(
                f"INFO: Created training dataset with {len(self.train_dataset)} samples"
            )
            print(
                f"INFO: Created validation dataset with {len(self.val_dataset)} samples"
            )

        if stage == "test" or stage is None:
            self.test_dataset = HydroLazyDataset(
                index_file_path=self.test_index_path,
                index_meta_file_path=self.test_index_meta_path,
                **common_args,
            )
            print(f"INFO: Created test dataset with {len(self.test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        """
        Create the training data loader.

        Returns:
            DataLoader for the training dataset
        """
        if self.train_dataset is None:
            raise RuntimeError("setup() must be called before train_dataloader()")
        sampler = FileGroupedBatchSampler(
            index_meta_path=self.train_index_meta_path,
            batch_size=self.batch_size,
            shuffle=True,
            files_per_batch=self.files_per_batch,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create the validation data loader.

        Returns:
            DataLoader for the validation dataset
        """
        if self.val_dataset is None:
            raise RuntimeError("setup() must be called before val_dataloader()")
        sampler = FileGroupedBatchSampler(
            index_meta_path=self.val_index_meta_path,
            batch_size=self.batch_size,
            shuffle=False,
            files_per_batch=self.files_per_batch,
        )
        return DataLoader(
            self.val_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create the test data loader.

        Returns:
            DataLoader for the test dataset
        """
        if self.test_dataset is None:
            raise RuntimeError("setup() must be called before test_dataloader()")
        sampler = FileGroupedBatchSampler(
            index_meta_path=self.test_index_meta_path,
            batch_size=self.batch_size,
            shuffle=False,
            files_per_batch=self.files_per_batch,
        )
        return DataLoader(
            self.test_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        basin_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Inverse transform the predictions using the fitted pipeline for the target.

        Args:
            predictions: The predictions to inverse transform.
            basin_ids: The basin IDs corresponding to the predictions.

        Returns:
            np.ndarray: The inverse transformed predictions.
        """

        if "target" not in self.fitted_pipelines:
            raise RuntimeError("No 'target' pipeline (did you call prepare_data()?)")

        orig_shape = predictions.shape
        vals = np.asarray(predictions).ravel()
        gids = np.asarray(basin_ids).ravel()

        target_pipeline = self.fitted_pipelines["target"]
        missing = set(np.unique(gids)) - set(target_pipeline.fitted_pipelines)
        if missing:
            raise RuntimeError(f"No fitted pipeline for basins: {sorted(missing)}")

        df = pd.DataFrame({self.group_identifier: gids, self.target: vals})
        inv_df = target_pipeline.inverse_transform(df)

        inv_vals = inv_df[self.target].values
        return inv_vals.reshape(orig_shape)

    def save_configuration(self, filepath: Union[str, Path]) -> Optional[str]:
        """
        Save the DataModule configuration to a JSON file.

        Extracts relevant configuration parameters and saves them to the specified
        file path. This allows for reproducibility and tracking of data processing
        runs.

        Args:
            filepath: Path where the configuration will be saved

        Returns:
            None if saving was successful, or error message
        """
        config_path = Path(filepath)

        config = extract_relevant_config(self)

        config["run_uuid"] = generate_run_uuid(config)

        try:
            save_config(config, config_path)
            return None
        except Exception as e:
            return str(e)

    def get_configuration(self) -> dict[str, Any]:
        """
        Get the relevant configuration of this DataModule as a dictionary.

        Extracts key parameters that define the data processing configuration,
        including features, target, splits, and preprocessing settings.

        Returns:
            Dictionary containing the relevant configuration parameters
        """
        return extract_relevant_config(self)
