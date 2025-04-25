import pytorch_lightning as pl
from pathlib import Path
from typing import Union, Optional, Any, TypedDict
from torch.utils.data import DataLoader
import math
import torch
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from .lazy_dataset import HydroLazyDataset
from .preprocessing import run_hydro_processor, QualityReport
from .index_entry_creator import (
    create_index_entries,
    split_index_entries_by_stage,
)
from ..preprocessing.grouped import GroupedPipeline
from .batch_sampler import FileGroupedBatchSampler
from .config_utils import (
    extract_relevant_config,
    save_config,
    generate_run_uuid,
    load_config,
    load_pipelines,
)


class LoadedData(TypedDict):
    """Data structure for loaded preprocessing artifacts."""

    quality_report: QualityReport
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

        # Validation logic rewritten without returns
        validation_error = self._validate_all()
        if validation_error is not None:
            raise ValueError(
                f"Invalid configuration for HydroLazyDataModule: {validation_error}"
            )

    def _validate_all(self) -> Optional[str]:
        """Run all validations, return error string if any, else None."""
        checks = [
            self._validate_positive_integer("batch_size", self.batch_size),
            self._validate_positive_integer("input_length", self.input_length),
            self._validate_positive_integer("output_length", self.output_length),
            self._validate_positive_integer("num_workers", self.num_workers),
            self._validate_positive_integer("files_per_batch", self.files_per_batch),
            self._validate_non_negative_integer(
                "max_imputation_gap_size", self.max_imputation_gap_size
            ),
            self._validate_positive_float("min_train_years", self.min_train_years),
            self._validate_gauge_id_list(
                "list_of_gauge_ids_to_process", self.list_of_gauge_ids_to_process
            ),
            self._validate_string_list("forcing_features", self.forcing_features),
            self._validate_string_list("static_features", self.static_features),
            self._validate_non_empty_string("target", self.target),
            self._validate_non_empty_string("group_identifier", self.group_identifier),
            self._validate_non_empty_string("domain_id", self.domain_id),
            self._validate_domain_type("domain_type", self.domain_type),
            self._validate_preprocessing_config(self.preprocessing_configs),
            self._validate_train_val_test_prop(),
            self._validate_target_not_in_forcing_features(),
        ]
        for result in checks:
            if result is not None:
                return result
        return None

    def _validate_positive_integer(self, param_name: str, value: int) -> Optional[str]:
        if not isinstance(value, int):
            return f"Parameter '{param_name}' must be an integer, got {type(value).__name__}"
        if value <= 0:
            return f"Parameter '{param_name}' must be greater than 0, got {value}"
        return None

    def _validate_non_negative_integer(
        self, param_name: str, value: int
    ) -> Optional[str]:
        if not isinstance(value, int):
            return f"Parameter '{param_name}' must be an integer, got {type(value).__name__}"
        if value < 0:
            return f"Parameter '{param_name}' must be greater than or equal to 0, got {value}"
        return None

    def _validate_positive_float(self, param_name: str, value: float) -> Optional[str]:
        if not isinstance(value, (int, float)):
            return (
                f"Parameter '{param_name}' must be a number, got {type(value).__name__}"
            )
        if value <= 0:
            return f"Parameter '{param_name}' must be greater than 0, got {value}"
        return None

    def _validate_gauge_id_list(
        self, param_name: str, value: Optional[list[str]]
    ) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, list):
            return (
                f"Parameter '{param_name}' must be a list, got {type(value).__name__}"
            )
        if len(value) == 0:
            return f"Parameter '{param_name}' must not be an empty list"
        if not all(isinstance(item, str) for item in value):
            return f"All items in '{param_name}' must be strings"
        return None

    def _validate_string_list(self, param_name: str, value: list[str]) -> Optional[str]:
        if not isinstance(value, list):
            return (
                f"Parameter '{param_name}' must be a list, got {type(value).__name__}"
            )
        if len(value) == 0:
            return f"Parameter '{param_name}' must not be an empty list"
        if not all(isinstance(item, str) for item in value):
            return f"All items in '{param_name}' must be strings"
        return None

    def _validate_non_empty_string(self, param_name: str, value: str) -> Optional[str]:
        if not isinstance(value, str):
            return (
                f"Parameter '{param_name}' must be a string, got {type(value).__name__}"
            )
        if len(value) == 0:
            return f"Parameter '{param_name}' must not be an empty string"
        return None

    def _validate_domain_type(self, param_name: str, value: str) -> Optional[str]:
        if not isinstance(value, str):
            return (
                f"Parameter '{param_name}' must be a string, got {type(value).__name__}"
            )
        if value not in ["source", "target"]:
            return f"Parameter '{param_name}' must be either 'source' or 'target', got '{value}'"
        return None

    def _validate_preprocessing_config(
        self, config: dict[str, dict[str, object]]
    ) -> Optional[str]:
        for data_type, cfg in config.items():
            if "pipeline" not in cfg:
                return f"Missing 'pipeline' key in {data_type} config"
            pipeline = cfg["pipeline"]
            if not isinstance(pipeline, (Pipeline, GroupedPipeline)):
                return f"Pipeline for {data_type} must be Pipeline or GroupedPipeline, got {type(pipeline)}"
            if isinstance(pipeline, GroupedPipeline):
                if pipeline.group_identifier != self.group_identifier:
                    return (
                        f"GroupedPipeline for {data_type} uses group_identifier "
                        f"'{pipeline.group_identifier}' but data module uses "
                        f"'{self.group_identifier}'"
                    )
            if data_type == "static_features" and "columns" not in cfg:
                return "static_features config must include 'columns' key"
            err = self._validate_pipeline_compatibility(pipeline)
            if err is not None:
                return err
        return None

    def _validate_pipeline_compatibility(
        self, pipeline: Union["Pipeline", "GroupedPipeline"]
    ) -> Optional[str]:
        if isinstance(pipeline, GroupedPipeline):
            pipeline = pipeline.pipeline
        for _, transformer in pipeline.steps:
            required_methods = ["fit", "transform", "inverse_transform"]
            missing = [m for m in required_methods if not hasattr(transformer, m)]
            if missing:
                return f"Transformer {transformer.__class__.__name__} missing required methods: {missing}"
        return None

    def _validate_train_val_test_prop(self) -> Optional[str]:
        total_prop = math.fsum([self.train_prop, self.val_prop, self.test_prop])
        if math.isclose(total_prop, 1.0, abs_tol=1e-6):
            return None
        return f"Training, validation, and test proportions must sum to 1. Current sum: {total_prop}"

    def _validate_target_not_in_forcing_features(self) -> Optional[str]:
        if self.target in self.forcing_features:
            return f"Target variable '{self.target}' should not be included in forcing features. Set is_autoregressive=True to include it."
        return None

    def _check_and_load_processed_data(
        self,
    ) -> tuple[bool, Optional[LoadedData], Optional[str]]:
        """
        Check if processed data exists for the current configuration and load it.

        Returns:
            Tuple (success, LoadedData or None, error message or None)
        """
        relevant_config = extract_relevant_config(self)
        run_uuid = generate_run_uuid(relevant_config)
        run_output_dir = Path(self.path_to_preprocessing_output_directory) / run_uuid
        config_path = run_output_dir / "config.json"
        pipelines_path = run_output_dir / "pipelines.joblib"
        report_path = run_output_dir / "quality_report.json"
        success_marker_path = run_output_dir / "_SUCCESS"
        processed_ts_dir = run_output_dir / "processed_timeseries"
        processed_static_path_candidate = (
            run_output_dir / "processed_static_attributes.parquet"
        )

        if not run_output_dir.is_dir():
            return (
                False,
                None,
                f"Processed data not found for run {run_uuid} at {run_output_dir}",
            )
        if not success_marker_path.is_file():
            return (
                False,
                None,
                f"Incomplete processing run {run_uuid} (no _SUCCESS marker found)",
            )
        if not processed_ts_dir.is_dir():
            return (
                False,
                None,
                f"Processed time series directory not found at {processed_ts_dir}",
            )
        processed_static_path = (
            processed_static_path_candidate
            if processed_static_path_candidate.is_file()
            else None
        )
        try:
            loaded_config = load_config(config_path)
            loaded_report_dict = load_config(report_path)
            loaded_pipelines = load_pipelines(pipelines_path)
            config_error = self._validate_loaded_config(loaded_config, relevant_config)
            if config_error is not None:
                return False, None, config_error
            loaded_data: LoadedData = {
                "quality_report": loaded_report_dict,
                "fitted_pipelines": loaded_pipelines,
                "processed_ts_dir": processed_ts_dir,
                "processed_static_path": processed_static_path,
            }
            return True, loaded_data, None
        except Exception as e:
            return False, None, f"Failed to load processed data: {str(e)}"

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

    def prepare_data(self):
        """
        Process the data, apply preprocessing, and create index entries.
        This method is called only once on a single GPU.
        """
        if self._prepare_data_has_run:
            print("INFO: prepare_data() has already been run; skipping.")
            return

        required_columns = list(dict.fromkeys(self.forcing_features + [self.target]))

        success, loaded_data, error = self._check_and_load_processed_data()
        if success:
            print("INFO: Found existing processed data with matching configuration")
            self.quality_report = loaded_data["quality_report"]
            self.fitted_pipelines = loaded_data["fitted_pipelines"]
            self.processed_time_series_dir = loaded_data["processed_ts_dir"]
            self.processed_static_attributes_path = loaded_data["processed_static_path"]
            excluded_ids = set(self.quality_report.get("excluded_basins", {}).keys())
            retained_ids = [
                bid
                for bid in self.quality_report.get("basins", {}).keys()
                if bid not in excluded_ids
            ]
            if not retained_ids:
                raise RuntimeError("No valid basins found in existing processed data")
            self.list_of_gauge_ids_to_process = retained_ids
        else:
            print(f"INFO: No existing processed data found: {error}")
            print("INFO: Running full preprocessing pipeline...")

            relevant_config = extract_relevant_config(self)
            run_uuid = generate_run_uuid(relevant_config)
            ok, result_data, err = run_hydro_processor(
                region_time_series_base_dirs=self.region_time_series_base_dirs,
                region_static_attributes_base_dirs=self.region_static_attributes_base_dirs,
                path_to_preprocessing_output_directory=self.path_to_preprocessing_output_directory,
                required_columns=required_columns,
                preprocessing_config=self.preprocessing_configs,
                min_train_years=self.min_train_years,
                max_imputation_gap_size=self.max_imputation_gap_size,
                group_identifier=self.group_identifier,
                train_prop=self.train_prop,
                val_prop=self.val_prop,
                test_prop=self.test_prop,
                processes=self.num_workers,
                list_of_gauge_ids_to_process=self.list_of_gauge_ids_to_process,
                run_uuid=run_uuid,
                datamodule_config=relevant_config,
            )
            if not ok:
                raise RuntimeError(f"Data preprocessing failed: {err}")
            self.quality_report = result_data["quality_report"]
            excluded_ids = set(self.quality_report.get("excluded_basins", {}).keys())
            retained_ids = [
                bid
                for bid in self.quality_report.get("basins", {}).keys()
                if bid not in excluded_ids
            ]
            if not retained_ids:
                raise RuntimeError(
                    "All basins excluded during preprocessing; no valid basins to process"
                )
            self.list_of_gauge_ids_to_process = retained_ids
            self.fitted_pipelines = result_data["fitted_pipelines"]
            self.processed_time_series_dir = result_data["processed_timeseries_dir"]
            self.processed_static_attributes_path = result_data[
                "processed_static_attributes_path"
            ]

        # Create index entries with explicit split proportions and static file path
        self.index_entries = create_index_entries(
            gauge_ids=self.list_of_gauge_ids_to_process,
            time_series_base_dir=self.processed_time_series_dir,
            static_file_path=self.processed_static_attributes_path,
            input_length=self.input_length,
            output_length=self.output_length,
            train_prop=self.train_prop,
            val_prop=self.val_prop,
            test_prop=self.test_prop,
            cols_to_check=required_columns,
        )

        self.index_entries_by_stage = split_index_entries_by_stage(
            index_entries=self.index_entries,
        )

        self.train_index_entries = self.index_entries_by_stage["train"]
        self.val_index_entries = self.index_entries_by_stage["val"]
        self.test_index_entries = self.index_entries_by_stage["test"]

        self._prepare_data_has_run = True

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

        # Create datasets based on stage
        if stage == "fit" or stage is None:
            for entry in self.train_index_entries:
                entry["static_file_path"] = static_file_path

            self.train_dataset = HydroLazyDataset(
                batch_index_entries=self.train_index_entries,
                **common_args,
            )

            for entry in self.val_index_entries:
                entry["static_file_path"] = static_file_path

            self.val_dataset = HydroLazyDataset(
                batch_index_entries=self.val_index_entries,
                **common_args,
            )

            print(
                f"INFO: Created training dataset with {len(self.train_dataset)} samples"
            )
            print(
                f"INFO: Created validation dataset with {len(self.val_dataset)} samples"
            )

        if stage == "test" or stage is None:
            for entry in self.test_index_entries:
                entry["static_file_path"] = static_file_path

            self.test_dataset = HydroLazyDataset(
                batch_index_entries=self.test_index_entries,
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
            self.train_dataset.batch_index_entries,
            self.batch_size,
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

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
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

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle test data
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
