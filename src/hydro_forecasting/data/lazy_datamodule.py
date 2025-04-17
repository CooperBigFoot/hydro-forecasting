import pytorch_lightning as pl
from pathlib import Path
from typing import Union, Optional
from torch.utils.data import DataLoader
from returns.result import Result, Success, Failure
from returns.pipeline import is_successful
import math
import torch

from sklearn.pipeline import Pipeline
from .lazy_dataset import HydroLazyDataset
from .preprocessing import run_hydro_processor
from .index_entry_creator import (
    create_index_entries,
    split_index_entries_by_stage,
)
from ..preprocessing.grouped import GroupedPipeline
from .batch_sampler import FileGroupedBatchSampler


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
        path_to_time_series_directory: Union[str, Path],
        path_to_static_attributes_directory: Union[str, Path],
        path_to_preprocessing_output_directory: Union[str, Path],
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

        self.path_to_time_series_directory = Path(path_to_time_series_directory)
        self.path_to_static_attributes_directory = Path(
            path_to_static_attributes_directory
        )
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

        # Post initialization
        self.quality_report = {}
        self.fitted_pipelines = {}
        self.processed_time_series_dir = Path("")
        self.processed_static_attributes_dir = Path("")
        self.index_entries = []
        self.index_entries_by_stage = {}
        self.train_index_entries = []
        self.val_index_entries = []
        self.test_index_entries = []

        # Dataset instances created in setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        validation_result = (
            Success(None)
            .bind(
                lambda _: self._validate_preprocessing_config(
                    self.preprocessing_configs
                )
            )
            .bind(lambda _: self._validate_train_val_test_prop())
            .bind(lambda _: self._validate_target_not_in_forcing_features())
        )

        if not is_successful(validation_result):
            raise ValueError(f"Validation failed: {validation_result.failure()}")

    def _validate_preprocessing_config(
        self, config: dict[str, dict[str, object]]
    ) -> Result[None, str]:
        """
        Validates the preprocessing configuration structure and pipeline types.

        Args:
            config: Dictionary mapping data types to their pipeline configs.

        Returns:
            Success(None) if valid, Failure(str) with error message otherwise.
        """
        for data_type, cfg in config.items():
            # Check if 'pipeline' exists in the config dictionary
            if "pipeline" not in cfg:
                return Failure(f"Missing 'pipeline' key in {data_type} config")

            pipeline = cfg["pipeline"]

            if not isinstance(pipeline, (Pipeline, GroupedPipeline)):
                return Failure(
                    f"Pipeline for {data_type} must be Pipeline or GroupedPipeline, got {type(pipeline)}"
                )

            # Check GroupedPipeline compatibility
            if isinstance(pipeline, GroupedPipeline):
                if pipeline.group_identifier != self.group_identifier:
                    return Failure(
                        f"GroupedPipeline for {data_type} uses group_identifier "
                        f"'{pipeline.group_identifier}' but data module uses "
                        f"'{self.group_identifier}'"
                    )

            # Validate static_features has columns specified
            if data_type == "static_features" and "columns" not in cfg:
                return Failure("static_features config must include 'columns' key")

            result = self._validate_pipeline_compatibility(pipeline)
            if not is_successful(result):
                return result

        return Success(None)

    def _validate_pipeline_compatibility(
        self, pipeline: Union["Pipeline", "GroupedPipeline"]
    ) -> Result[None, str]:
        """
        Verifies that all transformers in the pipeline implement required methods.

        Args:
            pipeline: Pipeline or GroupedPipeline instance.

        Returns:
            Success(None) if valid, Failure(str) with error message otherwise.
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
        return Success(None)

    def _validate_train_val_test_prop(self) -> Result[None, str]:
        """
        Validates the training, validation, and test proportions.

        Returns:
            Success(None) if valid, Failure(str) with error message otherwise.
        """
        total_prop = math.fsum([self.train_prop, self.val_prop, self.test_prop])
        if math.isclose(total_prop, 1.0, abs_tol=1e-6):
            return Success(None)
        return Failure(
            f"Training, validation, and test proportions must sum to 1. Current sum: {total_prop}"
        )

    def _validate_target_not_in_forcing_features(self) -> Result[None, str]:
        """
        Validates that the target variable is not included in the forcing features.

        Returns:
            Success(None) if valid, Failure(str) with error message otherwise.
        """
        if self.target in self.forcing_features:
            return Failure(
                f"Target variable '{self.target}' should not be included in forcing features."
            )
        return Success(None)

    def prepare_data(self):
        """
        Process the data, apply preprocessing, and create index entries.
        This method is called only once on a single GPU.
        """

        required_columns = list(dict.fromkeys(self.forcing_features + [self.target]))

        results = run_hydro_processor(
            path_to_time_series_directory=self.path_to_time_series_directory,
            path_to_preprocessing_output_directory=self.path_to_preprocessing_output_directory,
            path_to_static_attributes_directory=self.path_to_static_attributes_directory,
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
        )

        self.quality_report = results["quality_report"]
        self.fitted_pipelines = results["fitted_pipelines"]
        self.processed_time_series_dir = results["processed_time_series_dir"]
        self.processed_static_attributes_dir = results[
            "processed_static_attributes_dir"
        ]

        # Create index entries with explicit split proportions
        self.index_entries = create_index_entries(
            gauge_ids=self.list_of_gauge_ids_to_process,
            time_series_base_dir=self.processed_time_series_dir,
            input_length=self.input_length,
            output_length=self.output_length,
            train_prop=self.train_prop,
            val_prop=self.val_prop,
            test_prop=self.test_prop,
        )

        self.index_entries_by_stage = split_index_entries_by_stage(
            index_entries=self.index_entries,
        )

        self.train_index_entries = self.index_entries_by_stage["train"]
        self.val_index_entries = self.index_entries_by_stage["val"]
        self.test_index_entries = self.index_entries_by_stage["test"]

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

        # Get the path to static attributes file
        static_file_path = (
            self.processed_static_attributes_dir / "static_attributes.parquet"
        )

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
            # Update static_file_path in each index entry
            for entry in self.train_index_entries:
                entry["static_file_path"] = static_file_path

            self.train_dataset = HydroLazyDataset(
                batch_index_entries=self.train_index_entries,
                **common_args,
            )

            # Update static_file_path in each index entry
            for entry in self.val_index_entries:
                entry["static_file_path"] = static_file_path

            self.val_dataset = HydroLazyDataset(
                batch_index_entries=self.val_index_entries,
                **common_args,
            )

            print(f"Created training dataset with {len(self.train_dataset)} samples")
            print(f"Created validation dataset with {len(self.val_dataset)} samples")

        if stage == "test" or stage is None:
            # Update static_file_path in each index entry
            for entry in self.test_index_entries:
                entry["static_file_path"] = static_file_path

            self.test_dataset = HydroLazyDataset(
                batch_index_entries=self.test_index_entries,
                **common_args,
            )

            print(f"Created test dataset with {len(self.test_dataset)} samples")

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
