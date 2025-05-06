# filename: src/hydro_forecasting/data/in_memory_datamodule.py
from pytorch_lightning import LightningDataModule
from pathlib import Path
from typing import Union, Optional, Any
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import random
import math
import torch
import polars as pl
import numpy as np
from returns.result import Success, Failure
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from .in_memory_dataset import InMemoryChunkDataset  # Import the new dataset
from .preprocessing import run_hydro_processor, ProcessingOutput

from ..preprocessing.grouped import GroupedPipeline

from .config_utils import (
    extract_relevant_config,
    save_config,
    generate_run_uuid,
    load_config,
)
from .clean_data import SummaryQualityReport
from ..preprocessing.static_preprocessing import (
    load_static_pipeline,
    read_static_data_from_disk,
)
from ..preprocessing.time_series_preprocessing import load_time_series_pipelines
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoadedData:
    """Data structure for loaded preprocessing artifacts."""

    summary_quality_report: SummaryQualityReport
    fitted_pipelines: dict[str, Union[Pipeline, GroupedPipeline]]
    processed_ts_dir: Path
    processed_static_path: Optional[Path]


class HydroInMemoryDataModule(LightningDataModule):
    """
    A PyTorch Lightning DataModule that loads data in chunks into memory.
    """

    def __init__(
        self,
        region_time_series_base_dirs: dict[str, Union[str, Path]],
        region_static_attributes_base_dirs: dict[str, Union[str, Path]],
        path_to_preprocessing_output_directory: Union[str, Path],
        group_identifier: str,
        batch_size: int,
        input_length: int,
        output_length: int,
        forcing_features: list[str],
        static_features: list[str],
        target: str,
        preprocessing_configs: dict[
            str, dict[str, Any]
        ],  # Pipelines are now inside this dict
        num_workers: int,
        min_train_years: int,
        train_prop: float,
        val_prop: float,
        test_prop: float,
        max_imputation_gap_size: int,
        chunk_size: int,
        recompute_every: int = 10,  # Default to recomputing chunks every 10 epochs
        list_of_gauge_ids_to_process: Optional[list[str]] = None,
        domain_id: str = "source",
        domain_type: str = "source",
        is_autoregressive: bool = False,
        load_engine: str = "polars",  # Engine for InMemoryChunkDataset loading
    ):
        """
        Initializes the HydroInMemoryDataModule.

        Args:
            region_time_series_base_dirs: Mapping from region prefix to raw time series directory.
            region_static_attributes_base_dirs: Mapping from region prefix to raw static attribute directory.
            path_to_preprocessing_output_directory: Base directory for storing/finding processed data.
            group_identifier: Column name identifying basins (e.g., 'gauge_id').
            batch_size: Number of samples per batch.
            input_length: Length of input sequence.
            output_length: Length of output sequence (forecast horizon).
            forcing_features: List of forcing feature column names.
            static_features: List of static feature column names.
            target: Name of the target variable column.
            preprocessing_configs: Dictionary defining preprocessing pipelines for 'features', 'target', 'static_features'.
                                   Example: {"features": {"pipeline": GroupedPipeline(...)}, ...}
            num_workers: Number of workers for DataLoader.
            min_train_years: Minimum required years of data for a basin to be included.
            train_prop: Proportion of data for training split.
            val_prop: Proportion of data for validation split.
            test_prop: Proportion of data for test split.
            max_imputation_gap_size: Maximum gap size for imputation during preprocessing.
            chunk_size: Number of basins to load into memory per chunk.
            recompute_every: Number of epochs after which to reshuffle and recompute chunks.
            list_of_gauge_ids_to_process: Optional list to explicitly define which basins to process/use.
            domain_id: Identifier for the data domain (e.g., 'CH', 'US').
            domain_type: Type of domain ('source' or 'target').
            is_autoregressive: Whether the model uses past target values as input features.
            load_engine: Engine to use for loading Parquet files in InMemoryChunkDataset ('polars' or 'pyarrow').
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=["preprocessing_configs"]
        )  # Don't save pipelines directly

        # Store preprocessing configs separately as they contain complex objects
        self.preprocessing_configs = preprocessing_configs

        # Assign attributes from hparams for easy access
        self.region_time_series_base_dirs = self.hparams.region_time_series_base_dirs
        self.region_static_attributes_base_dirs = (
            self.hparams.region_static_attributes_base_dirs
        )
        self.path_to_preprocessing_output_directory = Path(
            self.hparams.path_to_preprocessing_output_directory
        )
        self.group_identifier = self.hparams.group_identifier
        self.batch_size = self.hparams.batch_size
        self.input_length = self.hparams.input_length
        self.output_length = self.hparams.output_length
        self.forcing_features = self.hparams.forcing_features
        self.static_features = self.hparams.static_features
        self.target = self.hparams.target
        self.num_workers = self.hparams.num_workers
        self.min_train_years = self.hparams.min_train_years
        self.train_prop = self.hparams.train_prop
        self.val_prop = self.hparams.val_prop
        self.test_prop = self.hparams.test_prop
        self.max_imputation_gap_size = self.hparams.max_imputation_gap_size
        self.chunk_size = self.hparams.chunk_size
        self.recompute_every = self.hparams.recompute_every
        self.list_of_gauge_ids_to_process = self.hparams.list_of_gauge_ids_to_process
        self.domain_id = self.hparams.domain_id
        self.domain_type = self.hparams.domain_type
        self.is_autoregressive = self.hparams.is_autoregressive
        self.load_engine = self.hparams.load_engine

        # Internal state initialization
        self._prepare_data_has_run: bool = False
        self.run_uuid: Optional[str] = None
        self.summary_quality_report: Optional[SummaryQualityReport] = None
        self.fitted_pipelines: dict[str, Union[Pipeline, GroupedPipeline]] = {}
        self.processed_time_series_dir: Optional[Path] = None
        self.processed_static_attributes_path: Optional[Path] = None

        # Chunk management state
        self._all_train_basin_ids: list[str] = []
        self._val_basin_ids: list[str] = []
        self._test_basin_ids: list[str] = []
        self._chunks: list[list[str]] = []
        self._current_chunk_index: int = -1
        self.static_data_cache: dict[str, np.ndarray] = {}  # Loaded in setup

        # Validation (optional, can be reused from HydroLazyDataModule if applicable)
        # validation_result = self._validate_all() # Assuming _validate_all exists
        # if isinstance(validation_result, Failure):
        #     raise ValueError(f"Invalid config: {validation_result.failure()}")

    # --- Core Methods ---

    def prepare_data(self) -> None:
        """
        Preprocesses the data if not already done or loaded.

        Checks for existing processed data matching the configuration. If found,
        reuses it. Otherwise, runs the `run_hydro_processor` to clean, transform,
        and save the data. Finally, identifies the basins available for each split.
        """
        if self._prepare_data_has_run:
            logger.info("Data preparation has already run.")
            return

        logger.info("Starting data preparation...")

        # 1. Attempt to reuse existing processed data
        # Use the *current* config to generate UUID and check for reuse
        current_config_dict = extract_relevant_config(self)  # Pass self
        # Add preprocessing_config details to current_config_dict for UUID generation
        current_config_dict["preprocessing_configs"] = {
            k: {kk: vv.__class__.__name__ for kk, vv in v.items()}  # Store class names
            for k, v in self.preprocessing_configs.items()
        }
        self.run_uuid = generate_run_uuid(current_config_dict)
        logger.info(f"Generated Run UUID for current config: {self.run_uuid}")

        reuse_success, loaded_data, reuse_message = (
            self._check_and_reuse_existing_processed_data(self.run_uuid)
        )

        processed_gauge_ids: list[str] = []

        if reuse_success and loaded_data:
            logger.info(
                f"Reusing existing processed data from run_uuid: {self.run_uuid}"
            )
            self.summary_quality_report = loaded_data.summary_quality_report
            self.fitted_pipelines = loaded_data.fitted_pipelines
            self.processed_time_series_dir = loaded_data.processed_ts_dir
            self.processed_static_attributes_path = loaded_data.processed_static_path
            processed_gauge_ids = (
                self.summary_quality_report.retained_basins
            )  # Use retained basins

            logger.info(
                f"Loaded {len(self.fitted_pipelines)} pipelines and data for {len(processed_gauge_ids)} basins from reused run."
            )

        else:
            logger.info(
                f"No reusable data found or reuse failed for UUID {self.run_uuid}. Reason: {reuse_message}. Running preprocessing..."
            )
            # 2. Run hydro processor if reuse failed
            processing_result = run_hydro_processor(
                region_time_series_base_dirs=self.region_time_series_base_dirs,
                region_static_attributes_base_dirs=self.region_static_attributes_base_dirs,
                path_to_preprocessing_output_directory=self.path_to_preprocessing_output_directory,
                required_columns=self.forcing_features + [self.target],
                run_uuid=self.run_uuid,
                datamodule_config=current_config_dict,  # Save the config used
                preprocessing_config=self.preprocessing_configs,
                min_train_years=self.min_train_years,
                max_imputation_gap_size=self.max_imputation_gap_size,
                group_identifier=self.group_identifier,
                train_prop=self.train_prop,
                val_prop=self.val_prop,
                test_prop=self.test_prop,
                list_of_gauge_ids_to_process=self.list_of_gauge_ids_to_process,
                basin_batch_size=50,  # Keep a reasonable batch size for processing
            )

            if isinstance(processing_result, Failure):
                raise RuntimeError(
                    f"Hydro processor failed: {processing_result.failure()}"
                )

            processing_output = processing_result.unwrap()
            logger.info("Hydro processor completed successfully.")

            # 3. Update DataModule state from processing output
            self.summary_quality_report = processing_output.summary_quality_report
            self.processed_time_series_dir = processing_output.processed_timeseries_dir
            self.processed_static_attributes_path = (
                processing_output.processed_static_attributes_path
            )

            # 4. Load fitted pipelines
            logger.info("Loading fitted pipelines...")
            self._load_pipelines_from_output(processing_output)

            # 5. Get processed gauge IDs
            processed_gauge_ids = self.summary_quality_report.retained_basins
            logger.info(f"{len(processed_gauge_ids)} basins retained after processing.")

            if not processed_gauge_ids:
                raise RuntimeError("No basins remained after the preprocessing stage.")

        # 6. Determine basin IDs for each split based on existing files
        self._determine_split_basin_ids(processed_gauge_ids)

        # 7. Set the flag
        self._prepare_data_has_run = True
        logger.info("Data preparation finished.")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Loads static data and prepares initial chunks for training.

        Args:
            stage: Current pipeline stage ('fit', 'validate', 'test', None).
        """
        if not self._prepare_data_has_run or not self.processed_time_series_dir:
            raise RuntimeError("prepare_data() must be called before setup()")

        # Load static data once
        if not self.static_data_cache:
            self._load_static_data()

        # Prepare initial chunks only during the 'fit' stage
        if stage == "fit" or stage is None:
            if not self._all_train_basin_ids:
                logger.warning(
                    "No training basin IDs available. Cannot compute chunks."
                )
                self._chunks = []
            elif not self._chunks:  # Only compute if not already computed
                self._recompute_chunks()
                logger.info(
                    f"Initial training chunks computed: {len(self._chunks)} chunks of approx size {self.chunk_size}"
                )

    def train_dataloader(self) -> DataLoader:
        """
        Creates the training DataLoader, handling chunk rotation and reloading.

        Returns:
            A DataLoader instance for the current training chunk.
        """
        if not self._all_train_basin_ids:
            logger.error(
                "No training basin IDs available. Cannot create train_dataloader."
            )
            # Return an empty dataloader to avoid crashing Trainer
            return DataLoader([])

        if not self._chunks:
            logger.warning("Training chunks not computed. Attempting to compute now.")
            self._recompute_chunks()
            if not self._chunks:
                logger.error(
                    "Failed to compute training chunks. Returning empty dataloader."
                )
                return DataLoader([])

        epoch = self.trainer.current_epoch if self.trainer else 0

        # Recompute chunks periodically
        if (
            epoch > 0
            and epoch % self.recompute_every == 0
            and self._current_chunk_index == len(self._chunks) - 1
        ):
            logger.info(f"Epoch {epoch}: Recomputing training chunks.")
            self._recompute_chunks()
            self._current_chunk_index = -1  # Reset index after recomputing

        # Cycle through chunks
        self._current_chunk_index = (self._current_chunk_index + 1) % len(self._chunks)
        current_basin_ids = self._chunks[self._current_chunk_index]
        logger.info(
            f"Epoch {epoch}: Loading training chunk {self._current_chunk_index + 1}/{len(self._chunks)} with {len(current_basin_ids)} basins."
        )

        # Instantiate dataset for the current chunk
        train_dataset = InMemoryChunkDataset(
            basin_ids=current_basin_ids,
            processed_data_dir=self.processed_time_series_dir,
            stage="train",
            static_data_cache=self.static_data_cache,
            input_length=self.input_length,
            output_length=self.output_length,
            target=self.target,
            forcing_features=self.forcing_features,
            static_features=self.static_features,
            group_identifier=self.group_identifier,
            is_autoregressive=self.is_autoregressive,
            load_engine=self.load_engine,
        )

        # Use a standard RandomSampler for shuffling within the chunk
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(train_dataset),  # Shuffle samples within the chunk
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            # worker_init_fn=worker_init_fn, # May need custom init for chunking
            # multiprocessing_context=mp.get_context("spawn"),
        )

    def val_dataloader(self) -> DataLoader:
        """
        Creates the validation DataLoader. Loads all validation basins at once.

        Returns:
            A DataLoader instance for the validation set.
        """
        if not self._val_basin_ids:
            logger.warning(
                "No validation basin IDs available. Returning empty dataloader."
            )
            return DataLoader([])

        logger.info(f"Loading validation data for {len(self._val_basin_ids)} basins...")
        val_dataset = InMemoryChunkDataset(
            basin_ids=self._val_basin_ids,
            processed_data_dir=self.processed_time_series_dir,
            stage="val",
            static_data_cache=self.static_data_cache,
            input_length=self.input_length,
            output_length=self.output_length,
            target=self.target,
            forcing_features=self.forcing_features,
            static_features=self.static_features,
            group_identifier=self.group_identifier,
            is_autoregressive=self.is_autoregressive,
            load_engine=self.load_engine,
        )

        if len(val_dataset) == 0:
            logger.warning(
                "Validation dataset is empty after loading. Returning empty dataloader."
            )
            return DataLoader([])

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(val_dataset),  # No shuffling for validation
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            # worker_init_fn=worker_init_fn,
            # multiprocessing_context=mp.get_context("spawn"),
        )

    def test_dataloader(self) -> DataLoader:
        """
        Creates the test DataLoader. Loads all test basins at once.

        Returns:
            A DataLoader instance for the test set.
        """
        if not self._test_basin_ids:
            logger.warning("No test basin IDs available. Returning empty dataloader.")
            return DataLoader([])

        logger.info(f"Loading test data for {len(self._test_basin_ids)} basins...")
        test_dataset = InMemoryChunkDataset(
            basin_ids=self._test_basin_ids,
            processed_data_dir=self.processed_time_series_dir,
            stage="test",
            static_data_cache=self.static_data_cache,
            input_length=self.input_length,
            output_length=self.output_length,
            target=self.target,
            forcing_features=self.forcing_features,
            static_features=self.static_features,
            group_identifier=self.group_identifier,
            is_autoregressive=self.is_autoregressive,
            load_engine=self.load_engine,
        )

        if len(test_dataset) == 0:
            logger.warning(
                "Test dataset is empty after loading. Returning empty dataloader."
            )
            return DataLoader([])

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(test_dataset),  # No shuffling for testing
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            # worker_init_fn=worker_init_fn,
            # multiprocessing_context=mp.get_context("spawn"),
        )

    # --- Helper Methods ---

    def _recompute_chunks(self) -> None:
        """Shuffles the full list of training basin IDs and partitions into chunks."""
        if not self._all_train_basin_ids:
            logger.error("Cannot recompute chunks: _all_train_basin_ids is empty.")
            self._chunks = []
            return

        logger.info(
            f"Recomputing chunks from {len(self._all_train_basin_ids)} training basins."
        )
        shuffled_basin_ids = random.sample(
            self._all_train_basin_ids, len(self._all_train_basin_ids)
        )
        self._chunks = [
            shuffled_basin_ids[i : i + self.chunk_size]
            for i in range(0, len(shuffled_basin_ids), self.chunk_size)
        ]
        # Filter out potential empty chunks if total basins is not multiple of chunk_size
        self._chunks = [chunk for chunk in self._chunks if chunk]
        logger.info(f"Created {len(self._chunks)} chunks.")

    def _load_static_data(self) -> None:
        """Loads static attributes from the processed static file into memory."""
        logger.info("Loading static data cache...")
        if (
            self.processed_static_attributes_path
            and self.processed_static_attributes_path.exists()
        ):
            try:
                # Use Polars for efficient loading
                static_df = pl.read_parquet(self.processed_static_attributes_path)

                # Ensure group_identifier and static_features are present
                required_static_cols = [self.group_identifier] + self.static_features
                missing_cols = [
                    col for col in required_static_cols if col not in static_df.columns
                ]
                if missing_cols:
                    logger.error(
                        f"Static data file missing required columns: {missing_cols}. Cannot load static cache."
                    )
                    self.static_data_cache = {}
                    return

                # Convert to dictionary {basin_id: np.ndarray}
                self.static_data_cache = {
                    row[self.group_identifier]: np.array(
                        [
                            row.get(f, 0.0) for f in self.static_features
                        ],  # Use get for safety
                        dtype=np.float32,
                    )
                    for row in static_df.select(required_static_cols).iter_rows(
                        named=True
                    )
                }
                logger.info(
                    f"Loaded static data for {len(self.static_data_cache)} basins."
                )
            except Exception as e:
                logger.error(
                    f"Failed to load processed static data from {self.processed_static_attributes_path}: {e}"
                )
                self.static_data_cache = {}
        else:
            logger.warning(
                "Processed static attributes file not found or not specified. Static cache will be empty."
            )
            self.static_data_cache = {}

    def _determine_split_basin_ids(self, processed_gauge_ids: list[str]) -> None:
        """
        Determines the list of basin IDs available for each split (train, val, test)
        by checking for the existence of their corresponding Parquet files in the
        processed_time_series subdirectories.

        Args:
            processed_gauge_ids: List of basin IDs that were successfully processed.
        """
        self._all_train_basin_ids = []
        self._val_basin_ids = []
        self._test_basin_ids = []

        if not self.processed_time_series_dir:
            logger.error(
                "Processed time series directory not set. Cannot determine split IDs."
            )
            return

        for stage in ["train", "val", "test"]:
            stage_dir = self.processed_time_series_dir / stage
            if not stage_dir.is_dir():
                logger.warning(f"Directory for stage '{stage}' not found: {stage_dir}")
                continue

            basin_list = []
            for basin_id in processed_gauge_ids:
                if (stage_dir / f"{basin_id}.parquet").exists():
                    basin_list.append(basin_id)

            if stage == "train":
                self._all_train_basin_ids = basin_list
            elif stage == "val":
                self._val_basin_ids = basin_list
            elif stage == "test":
                self._test_basin_ids = basin_list

        logger.info(f"Found {len(self._all_train_basin_ids)} basins for train split.")
        logger.info(f"Found {len(self._val_basin_ids)} basins for val split.")
        logger.info(f"Found {len(self._test_basin_ids)} basins for test split.")

    def _check_and_reuse_existing_processed_data(
        self, run_uuid: str
    ) -> tuple[bool, Optional[LoadedData], Optional[str]]:
        """
        Check if processed data exists for the given run_uuid and load it if valid.

        Args:
            run_uuid: The deterministic UUID for the current configuration.

        Returns:
            Tuple (success, LoadedData or None, error message or None)
        """
        try:
            run_dir = self.path_to_preprocessing_output_directory / run_uuid
            logger.info(f"Checking for existing processed data at: {run_dir}")

            if not run_dir.is_dir():
                return False, None, "Run directory not found."

            # Check for _SUCCESS marker file first
            success_marker_path = run_dir / "_SUCCESS"
            if not success_marker_path.exists():
                return False, None, f"_SUCCESS marker not found in {run_dir}."

            # Required files for reuse (excluding index files)
            required_files = [
                "config.json",
                "quality_summary.json",
                "fitted_time_series_pipelines.joblib",
            ]
            missing_files = [f for f in required_files if not (run_dir / f).exists()]
            if missing_files:
                return False, None, f"Missing required files for reuse: {missing_files}"

            # Required directories
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
                    f"Missing required directories for reuse: {missing_dirs}",
                )

            # --- Configuration Validation ---
            config_path = run_dir / "config.json"
            config_result = load_config(config_path)
            if isinstance(config_result, Failure):
                return (
                    False,
                    None,
                    f"Failed to load stored configuration: {config_result.failure()}",
                )

            loaded_config_dict = config_result.unwrap()
            current_config_dict = extract_relevant_config(self)  # Pass self
            # Add preprocessing pipeline details for comparison
            current_config_dict["preprocessing_configs"] = {
                k: {kk: vv.__class__.__name__ for kk, vv in v.items()}
                for k, v in self.preprocessing_configs.items()
            }

            # Compare critical keys (excluding complex objects like pipelines)
            critical_keys = list(current_config_dict.keys())
            critical_keys.remove(
                "preprocessing_configs"
            )  # Compare pipelines separately if needed

            mismatched_keys = []
            for key in critical_keys:
                # Handle Path objects in loaded config
                loaded_val = loaded_config_dict.get(key)
                current_val = current_config_dict.get(key)

                # Normalize Path objects to strings for comparison if necessary
                if isinstance(current_val, Path):
                    current_val = str(current_val)
                if isinstance(loaded_val, Path):
                    loaded_val = str(loaded_val)
                    # Normalize dicts with Path values
                if isinstance(current_val, dict) and any(
                    isinstance(v, Path) for v in current_val.values()
                ):
                    current_val = {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in current_val.items()
                    }
                if isinstance(loaded_val, dict) and any(
                    isinstance(v, Path) for v in loaded_val.values()
                ):
                    loaded_val = {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in loaded_val.items()
                    }

                if loaded_val != current_val:
                    mismatched_keys.append(
                        f"{key} (loaded: {loaded_val}, current: {current_val})"
                    )

            if mismatched_keys:
                return (
                    False,
                    None,
                    f"Configuration mismatch: {'; '.join(mismatched_keys)}",
                )
            # --- End Configuration Validation ---

            # Load summary quality report
            summary_path = run_dir / "quality_summary.json"
            with open(summary_path, "r") as f:
                import json

                summary_quality_report_dict = json.load(f)
                summary_quality_report_obj = SummaryQualityReport(
                    **summary_quality_report_dict
                )

            # Load fitted pipelines
            fitted_pipelines_dict = {}
            ts_pipelines_path = run_dir / "fitted_time_series_pipelines.joblib"
            ts_pipelines_result = load_time_series_pipelines(ts_pipelines_path)
            if isinstance(ts_pipelines_result, Failure):
                return (
                    False,
                    None,
                    f"Failed to load time series pipelines: {ts_pipelines_result.failure()}",
                )
            fitted_pipelines_dict.update(ts_pipelines_result.unwrap())

            # Load static pipeline if exists
            static_pipeline_path = run_dir / "fitted_static_pipeline.joblib"
            processed_static_path = run_dir / "processed_static_features.parquet"
            if static_pipeline_path.exists() and processed_static_path.exists():
                static_pipeline_result = load_static_pipeline(static_pipeline_path)
                if isinstance(static_pipeline_result, Failure):
                    logger.warning(
                        f"Found static pipeline file but failed to load: {static_pipeline_result.failure()}"
                    )
                    processed_static_path = (
                        None  # Treat as if static data is not available
                    )
                else:
                    fitted_pipelines_dict["static"] = (
                        static_pipeline_result.unwrap()
                    )  # Use 'static' key
            else:
                processed_static_path = None  # Static data not present in this run

            # Construct LoadedData dataclass
            loaded_data = LoadedData(
                summary_quality_report=summary_quality_report_obj,
                fitted_pipelines=fitted_pipelines_dict,
                processed_ts_dir=run_dir / "processed_time_series",
                processed_static_path=processed_static_path,
            )

            logger.info(
                f"Successfully validated and prepared to reuse data from {run_dir}"
            )
            return True, loaded_data, None

        except Exception as e:
            import traceback

            logger.error(
                f"Error checking for existing processed data: {e}\n{traceback.format_exc()}"
            )
            return False, None, f"Unexpected error during reuse check: {e}"

    def _load_pipelines_from_output(self, processing_output: ProcessingOutput) -> None:
        """Loads fitted pipelines from the paths specified in ProcessingOutput."""
        self.fitted_pipelines = {}

        # Load time series pipelines
        ts_pipelines_path = processing_output.fitted_time_series_pipelines_path
        if ts_pipelines_path and ts_pipelines_path.exists():
            ts_pipelines_result = load_time_series_pipelines(ts_pipelines_path)
            if isinstance(ts_pipelines_result, Success):
                self.fitted_pipelines.update(ts_pipelines_result.unwrap())
            else:
                raise RuntimeError(
                    f"Failed to load time series pipelines: {ts_pipelines_result.failure()}"
                )
        else:
            logger.warning("Fitted time series pipelines file not found.")

        # Load static pipeline if path exists
        if (
            processing_output.fitted_static_pipeline_path
            and processing_output.fitted_static_pipeline_path.exists()
        ):
            static_pipeline_path = processing_output.fitted_static_pipeline_path
            static_pipeline_result = load_static_pipeline(static_pipeline_path)
            if isinstance(static_pipeline_result, Success):
                self.fitted_pipelines["static"] = (
                    static_pipeline_result.unwrap()
                )  # Use key 'static'
            else:
                logger.warning(
                    f"Failed to load static pipeline: {static_pipeline_result.failure()}"
                )
        else:
            logger.info(
                "No fitted static pipeline found or specified in processing output."
            )

        logger.info(
            f"Successfully loaded {len(self.fitted_pipelines)} categories of fitted pipelines."
        )

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        basin_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Inverse transform predictions using the fitted 'target' pipeline.

        Args:
            predictions: NumPy array of model predictions.
            basin_ids: NumPy array of basin IDs corresponding to predictions.

        Returns:
            NumPy array of inverse-transformed predictions.

        Raises:
            RuntimeError: If the 'target' pipeline is not fitted or available.
            ValueError: If basin IDs are missing from the fitted pipeline.
        """
        if "target" not in self.fitted_pipelines:
            raise RuntimeError(
                "No 'target' pipeline found. Was prepare_data() called and did it succeed?"
            )

        target_pipeline = self.fitted_pipelines["target"]
        if not isinstance(target_pipeline, GroupedPipeline):
            raise RuntimeError("Expected 'target' pipeline to be a GroupedPipeline.")

        if not target_pipeline.fitted_pipelines:
            raise RuntimeError(
                "The 'target' GroupedPipeline has not been fitted (no fitted_pipelines found)."
            )

        # --- Input Validation ---
        if predictions.shape[0] != basin_ids.shape[0]:
            raise ValueError(
                f"Shape mismatch: predictions ({predictions.shape[0]}) and basin_ids ({basin_ids.shape[0]})"
            )

        # --- Prepare DataFrame for inverse_transform ---
        # Flatten predictions if they are multi-dimensional per basin (e.g., [n_samples, n_horizons])
        # Assume the first dimension matches basin_ids
        orig_shape = predictions.shape
        if predictions.ndim > 1:
            predictions_flat = predictions.reshape(-1)
            # Repeat basin_ids to match the flattened predictions
            basin_ids_expanded = np.repeat(
                basin_ids, predictions.shape[1] if predictions.ndim > 1 else 1
            )
        else:
            predictions_flat = predictions
            basin_ids_expanded = basin_ids

        df = pl.DataFrame(
            {self.group_identifier: basin_ids_expanded, self.target: predictions_flat}
        )

        # --- Check for missing basins in pipeline ---
        unique_request_gids = set(basin_ids_expanded)
        missing_gids = unique_request_gids - set(
            target_pipeline.fitted_pipelines.keys()
        )
        if missing_gids:
            raise ValueError(
                f"Inverse transform failed: No fitted pipeline for basins: {sorted(list(missing_gids))}"
            )

        # --- Perform Inverse Transformation ---
        try:
            # GroupedPipeline expects pandas DataFrame
            inv_df = target_pipeline.inverse_transform(df.to_pandas())
            inv_vals = inv_df[self.target].values
        except Exception as e:
            logger.error(f"Error during inverse_transform: {e}")
            raise RuntimeError("Inverse transformation failed.") from e

        # --- Reshape and Return ---
        try:
            # Reshape back to the original prediction shape
            reshaped_inv_vals = inv_vals.reshape(orig_shape)
        except ValueError as e:
            logger.error(
                f"Error reshaping inverse transformed values. Original shape: {orig_shape}, Inv_vals shape: {inv_vals.shape}"
            )
            raise RuntimeError(
                "Failed to reshape inverse transformed predictions."
            ) from e

        return reshaped_inv_vals
