import pytorch_lightning as pl
from pathlib import Path
from typing import Union, Optional
from returns.result import Result, Success, Failure

from sklearn.pipeline import Pipeline
from .preprocessing import run_hydro_processor
from .index_entry_creator import (
    SPLIT_CONFIG,
    create_index_entries,
    split_index_entries_by_stage,
)
from ..preprocessing.grouped import GroupedPipeline


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
        input_lenght: int,
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
        self.input_length = input_lenght
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

        # Post initialization
        self.quality_report = {}
        self.fitted_pipelines = {}
        self.processed_time_series_dir = Path("")
        self.processed_static_attributes_dir = Path("")
        self.index_entries = []
        self.index_entries_by_stage = {}

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
        required_keys = ["pipeline"]
        required_pipeline_keys = ["columns"]

        for data_type, cfg in config.items():
            missing = [k for k in required_keys if k not in cfg]
            if missing:
                return Failure(f"Missing required keys {missing} in {data_type} config")

            pipeline = cfg["pipeline"]

            if not isinstance(pipeline, (Pipeline, GroupedPipeline)):
                return Failure(
                    f"Pipeline for {data_type} must be Pipeline or GroupedPipeline, got {type(pipeline)}"
                )

            if isinstance(pipeline, Pipeline):
                missing_pipeline_keys = [
                    k for k in required_pipeline_keys if k not in pipeline.get_params()
                ]
                if missing_pipeline_keys:
                    return Failure(
                        f"Missing required keys {missing_pipeline_keys} in {data_type} pipeline"
                    )

            if isinstance(pipeline, GroupedPipeline):
                if pipeline.group_identifier != self.group_identifier:
                    return Failure(
                        f"GroupedPipeline for {data_type} uses group_identifier "
                        f"'{pipeline.group_identifier}' but data module uses "
                        f"'{self.group_identifier}'"
                    )

            result = self._validate_pipeline_compatibility(pipeline)
            if result.is_failure:
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

    def prepare_data(self):
        results = run_hydro_processor(
            path_to_time_series_directory=self.path_to_time_series_directory,
            path_to_static_attributes_directory=self.path_to_static_attributes_directory,
            path_to_preprocessing_output_directory=self.path_to_preprocessing_output_directory,
            required_columns=self.forcing_features + [self.target],
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

        # Update the global SPLIT_CONFIG with the current instance's properties
        SPLIT_CONFIG.train_prop = self.train_prop
        SPLIT_CONFIG.val_prop = self.val_prop
        SPLIT_CONFIG.test_prop = self.test_prop

        self.index_entries = create_index_entries(
            gauge_ids=self.list_of_gauge_ids_to_process,
            time_series_base_dir=self.processed_time_series_dir,
            input_length=self.input_length,
            output_length=self.output_length,
        )

        self.index_entries_by_stage = split_index_entries_by_stage(
            index_entries=self.index_entries,
        )
        pass

    def setup(self, stage=None):
        # Create the train, val, and test datasets
        if stage == "fit" or stage is None:
            self.train_dataset = ...
            self.val_dataset = ...

        if stage == "test" or stage is None:
            self.test_dataset = ...
