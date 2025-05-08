from pytorch_lightning import LightningDataModule
from pathlib import Path
from typing import Union, Optional, Any
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.multiprocessing as mp
import random

import torch
import polars as pl
import numpy as np
from returns.result import (
    Success,
    Failure,
)
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from .in_memory_dataset import InMemoryChunkDataset
from .preprocessing import (
    run_hydro_processor,
    ProcessingOutput,
)

from ..preprocessing.grouped import GroupedPipeline

from .config_utils import (
    extract_relevant_config,
    generate_run_uuid,
    load_config,
)
from .clean_data import SummaryQualityReport
from ..preprocessing.static_preprocessing import (
    load_static_pipeline,
)
from ..preprocessing.time_series_preprocessing import load_time_series_pipelines
import logging

from .datamodule_validators import validate_hydro_inmemory_datamodule_config


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
    Implements synchronized, memory-efficient chunking for train and val data.
    Chunks are recomputed (reshuffled and re-partitioned) every time the
    current set of chunks is exhausted.
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
        preprocessing_configs: dict[str, dict[str, Any]],
        num_workers: int,
        min_train_years: int,
        train_prop: float,
        val_prop: float,
        test_prop: float,
        max_imputation_gap_size: int,
        chunk_size: int,
        # recompute_every: int = 10, # Removed
        list_of_gauge_ids_to_process: Optional[list[str]] = None,
        domain_id: str = "source",
        domain_type: str = "source",
        is_autoregressive: bool = False,
        load_engine: str = "polars",
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
            preprocessing_configs: Dictionary defining preprocessing pipelines.
            num_workers: Number of workers for DataLoader.
            min_train_years: Minimum required years of data for a basin to be included.
            train_prop: Proportion of data for training split.
            val_prop: Proportion of data for validation split.
            test_prop: Proportion of data for test split.
            max_imputation_gap_size: Maximum gap size for imputation during preprocessing.
            chunk_size: Number of basins to load into memory per shared chunk.
            # recompute_every: REMOVED - Chunks are recomputed after each full pass.
            list_of_gauge_ids_to_process: Optional list to explicitly define which basins to process/use.
            domain_id: Identifier for the data domain.
            domain_type: Type of domain ('source' or 'target').
            is_autoregressive: Whether the model uses past target values as input features.
            load_engine: Engine to use for loading Parquet files in InMemoryChunkDataset.
        """
        super().__init__()
        self.preprocessing_configs = preprocessing_configs
        # Call save_hyperparameters() to make all __init__ args available via self.hparams
        # This must be done before validation since validators access self.hparams
        self.save_hyperparameters(ignore=["preprocessing_configs"])

        # The validator function will access most parameters from self.hparams,
        # but for preprocessing_configs, it directly uses the self.preprocessing_configs attribute.
        validation_result = validate_hydro_inmemory_datamodule_config(self)
        if isinstance(validation_result, Failure):
            raise ValueError(f"HydroInMemoryDataModule configuration error: {validation_result.failure()}")

        self.region_time_series_base_dirs = {k: Path(v) for k, v in self.hparams.region_time_series_base_dirs.items()}
        self.region_static_attributes_base_dirs = {
            k: Path(v) for k, v in self.hparams.region_static_attributes_base_dirs.items()
        }
        self.path_to_preprocessing_output_directory = Path(self.hparams.path_to_preprocessing_output_directory)
        self.group_identifier = self.hparams.group_identifier
        self.batch_size = self.hparams.batch_size
        self.input_length = self.hparams.input_length
        self.output_length = self.hparams.output_length
        self.forcing_features = self.hparams.forcing_features
        self.static_features = self.hparams.static_features
        self.target = self.hparams.target
        self.num_workers = self.hparams.num_workers
        self.min_train_years = float(self.hparams.min_train_years)
        self.train_prop = self.hparams.train_prop
        self.val_prop = self.hparams.val_prop
        self.test_prop = self.hparams.test_prop
        self.max_imputation_gap_size = self.hparams.max_imputation_gap_size
        self.chunk_size = self.hparams.chunk_size
        self.list_of_gauge_ids_to_process = self.hparams.list_of_gauge_ids_to_process
        self.domain_id = self.hparams.domain_id
        self.domain_type = self.hparams.domain_type
        self.is_autoregressive = self.hparams.is_autoregressive
        self.load_engine = self.hparams.load_engine

        self._prepare_data_has_run: bool = False
        self.run_uuid: Optional[str] = None
        self.summary_quality_report: Optional[SummaryQualityReport] = None
        self.fitted_pipelines: dict[str, Union[Pipeline, GroupedPipeline]] = {}
        self.processed_time_series_dir: Optional[Path] = None
        self.processed_static_attributes_path: Optional[Path] = None
        self.static_data_cache: dict[str, np.ndarray] = {}

        self.chunkable_basin_ids: list[str] = []
        self._shared_chunks: list[list[str]] = []
        self._current_shared_chunk_index: int = -1
        self._test_basin_ids: list[str] = []

    def prepare_data(self) -> None:
        """
        Preprocesses data if not already done.
        Identifies basins for training, validation (chunkable), and testing.
        """
        if self._prepare_data_has_run:
            logger.info("Data preparation has already run.")
            return

        logger.info("Starting data preparation...")
        current_config_dict = extract_relevant_config(self)

        if self.preprocessing_configs:
            current_config_dict["preprocessing_configs"] = {
                k: {kk: vv.__class__.__name__ if hasattr(vv, "__class__") else str(vv) for kk, vv in v.items()}
                for k, v in self.preprocessing_configs.items()
            }
        else:
            current_config_dict["preprocessing_configs"] = None

        self.run_uuid = generate_run_uuid(current_config_dict)
        logger.info(f"Generated Run UUID for current config: {self.run_uuid}")

        reuse_success, loaded_data, reuse_message = self._check_and_reuse_existing_processed_data(self.run_uuid)

        processed_gauge_ids_from_report: list[str] = []

        if reuse_success and loaded_data:
            logger.info(f"Reusing existing processed data from run_uuid: {self.run_uuid}")

            self.summary_quality_report = loaded_data.summary_quality_report
            self.fitted_pipelines = loaded_data.fitted_pipelines
            self.processed_time_series_dir = loaded_data.processed_ts_dir
            self.processed_static_attributes_path = loaded_data.processed_static_path
            processed_gauge_ids_from_report = self.summary_quality_report.retained_basins

            logger.info(
                f"Loaded {len(self.fitted_pipelines)} pipelines and data for {len(processed_gauge_ids_from_report)} basins from reused run."
            )
        else:
            logger.info(
                f"No reusable data found or reuse failed for UUID {self.run_uuid}. Reason: {reuse_message}. Running preprocessing..."
            )
            processing_result = run_hydro_processor(
                region_time_series_base_dirs=self.region_time_series_base_dirs,
                region_static_attributes_base_dirs=self.region_static_attributes_base_dirs,
                path_to_preprocessing_output_directory=self.path_to_preprocessing_output_directory,
                required_columns=self.forcing_features + [self.target],
                run_uuid=self.run_uuid,
                datamodule_config=current_config_dict, 
                preprocessing_config=self.preprocessing_configs,
                min_train_years=self.min_train_years,
                max_imputation_gap_size=self.max_imputation_gap_size,
                group_identifier=self.group_identifier,
                train_prop=self.train_prop,
                val_prop=self.val_prop,
                test_prop=self.test_prop,
                list_of_gauge_ids_to_process=self.list_of_gauge_ids_to_process,
                basin_batch_size=50, 
            )

            if isinstance(processing_result, Failure):
                raise RuntimeError(f"Hydro processor failed: {processing_result.failure()}")
            processing_output = processing_result.unwrap()
            logger.info("Hydro processor completed successfully.")

            self.summary_quality_report = processing_output.summary_quality_report
            self.processed_time_series_dir = processing_output.processed_timeseries_dir
            self.processed_static_attributes_path = processing_output.processed_static_attributes_path
            self._load_pipelines_from_output(processing_output)
            processed_gauge_ids_from_report = self.summary_quality_report.retained_basins
            logger.info(f"{len(processed_gauge_ids_from_report)} basins retained after processing.")

            if not processed_gauge_ids_from_report:
                raise RuntimeError("No basins remained after the preprocessing stage.")

        self._determine_basin_ids_for_splits(processed_gauge_ids_from_report)
        self._prepare_data_has_run = True
        logger.info("Data preparation finished.")

    def _determine_basin_ids_for_splits(self, processed_basins_from_report: list[str]) -> None:
        """
        Determines `self.chunkable_basin_ids` and `self._test_basin_ids`.
        Chunkable basins must exist in both train/ and val/ subdirectories.
        """
        if not self.processed_time_series_dir:
            logger.error("Processed time series directory not set. Cannot determine split IDs.")
            return

        train_dir = self.processed_time_series_dir / "train"
        val_dir = self.processed_time_series_dir / "val"
        test_dir = self.processed_time_series_dir / "test"

        self.chunkable_basin_ids = []
        self._test_basin_ids = []

        # Basins from summary_quality_report.retained_basins are the starting pool
        for basin_id in processed_basins_from_report:
            train_file = train_dir / f"{basin_id}.parquet"
            val_file = val_dir / f"{basin_id}.parquet"
            test_file = test_dir / f"{basin_id}.parquet"

            if train_file.exists() and val_file.exists():
                self.chunkable_basin_ids.append(basin_id)

            if test_file.exists():
                self._test_basin_ids.append(basin_id)

        logger.info(f"Found {len(self.chunkable_basin_ids)} basins for synchronized train/val chunking.")
        logger.info(f"Found {len(self._test_basin_ids)} basins for test split.")

        if not self.chunkable_basin_ids:
            logger.warning(
                "No basins available for chunkable train/val splits. Training and validation might be empty."
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Loads static data and prepares initial shared chunks for training/validation.

        Args:
            stage: Current pipeline stage ('fit', 'validate', 'test', None).
        """
        if not self._prepare_data_has_run or not self.processed_time_series_dir:
            raise RuntimeError("prepare_data() must be called before setup()")

        if not self.static_data_cache:
            self._load_static_data()

        if stage == "fit" or stage is None:  # Primarily for 'fit' stage
            if not self.chunkable_basin_ids:
                logger.warning("No chunkable_basin_ids available. Shared chunks will be empty.")
                self._shared_chunks = []
            elif not self._shared_chunks:  # Only compute if not already computed
                self._initialize_and_partition_shared_chunks()
                logger.info(
                    f"Initial shared chunks computed: {len(self._shared_chunks)} chunks of approx size {self.chunk_size} from {len(self.chunkable_basin_ids)} basins."
                )

    def _initialize_and_partition_shared_chunks(self) -> None:
        """
        Shuffles `self.chunkable_basin_ids` and partitions them into `self._shared_chunks`.
        Resets chunk index.
        """
        if not self.chunkable_basin_ids:
            logger.error("Cannot initialize shared chunks: chunkable_basin_ids is empty.")
            self._shared_chunks = []
            return

        logger.info(f"Initializing and partitioning shared chunks from {len(self.chunkable_basin_ids)} basins.")
        shuffled_basin_ids = random.sample(self.chunkable_basin_ids, len(self.chunkable_basin_ids))
        self._shared_chunks = [
            shuffled_basin_ids[i : i + self.chunk_size] for i in range(0, len(shuffled_basin_ids), self.chunk_size)
        ]
        self._shared_chunks = [chunk for chunk in self._shared_chunks if chunk]  # Filter out empty chunks

        self._current_shared_chunk_index = -1  # Reset for pre-increment logic
        # self._full_passes_completed = 0 # Removed
        logger.info(f"Created {len(self._shared_chunks)} shared chunks.")

    def _advance_and_recompute_shared_chunks_if_needed(self) -> None:
        """
        Manages the lifecycle of shared chunks. Advances the chunk index,
        and triggers re-shuffling/re-partitioning when all current chunks are exhausted.
        This method is called by PTL when reload_dataloaders_every_n_epochs=1 via train_dataloader.
        """
        if not self._shared_chunks:  # No chunks to advance
            return

        self._current_shared_chunk_index += 1

        if self._current_shared_chunk_index >= len(self._shared_chunks):
            # All current chunks have been processed
            logger.info("Completed a full pass through shared chunks. Recomputing shared chunks.")
            self._initialize_and_partition_shared_chunks()
            # The _initialize_and_partition_shared_chunks method resets
            # _current_shared_chunk_index to -1.
            # So, the next increment at the start of this method (if called again immediately)
            # will correctly point to the first new chunk (index 0).
            # We need to ensure the current call also yields the first new chunk.
            self._current_shared_chunk_index = 0  # Point to the first chunk of the new set.
            if not self._shared_chunks:  # If re-initialization resulted in no chunks
                logger.warning("Re-initialization of shared chunks resulted in an empty list.")
                return  # Nothing more to do

    def train_dataloader(self) -> DataLoader:
        """
        Creates the training DataLoader using a shared chunk of basin IDs.
        Manages shared chunk lifecycle if `reload_dataloaders_every_n_epochs=1`.

        Returns:
            A DataLoader instance for the current training chunk.
        """
        self._advance_and_recompute_shared_chunks_if_needed()

        if (
            not self._shared_chunks
            or self._current_shared_chunk_index < 0
            or self._current_shared_chunk_index >= len(self._shared_chunks)
        ):
            epoch_num_str = f"Epoch {self.trainer.current_epoch}" if self.trainer else "N/A"
            logger.error(
                f"{epoch_num_str}: No shared chunks available or invalid chunk index ({self._current_shared_chunk_index}). "
                f"Total chunks: {len(self._shared_chunks)}. Cannot create train_dataloader."
            )
            return DataLoader([], batch_size=self.batch_size)

        current_basins_for_epoch = self._shared_chunks[self._current_shared_chunk_index]

        epoch_num = self.trainer.current_epoch if self.trainer else 0
        logger.info(
            f"Epoch {epoch_num}: Train Dataloader using shared chunk {self._current_shared_chunk_index + 1}/{len(self._shared_chunks)} with {len(current_basins_for_epoch)} basins."
        )

        train_dataset = InMemoryChunkDataset(
            basin_ids=current_basins_for_epoch,
            processed_data_dir=self.processed_time_series_dir,  # type: ignore
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
        if len(train_dataset) == 0:
            logger.warning(f"Training dataset for chunk {self._current_shared_chunk_index + 1} is empty.")
            # Return empty dataloader but ensure PTL doesn't crash due to sampler issues
            return DataLoader(
                train_dataset,  # Pass empty dataset
                batch_size=self.batch_size,
                # No sampler needed for empty dataset, or handle gracefully
            )

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(train_dataset),
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
            multiprocessing_context=mp.get_context("spawn"),

        )

    def val_dataloader(self) -> DataLoader:
        """
        Creates the validation DataLoader using the *same* shared chunk of basin IDs
        as the current training epoch.

        Returns:
            A DataLoader instance for the current validation chunk.
        """
        if (
            not self._shared_chunks
            or self._current_shared_chunk_index < 0
            or self._current_shared_chunk_index >= len(self._shared_chunks)
        ):
            epoch_num_str = f"Epoch {self.trainer.current_epoch}" if self.trainer else "N/A"
            logger.error(
                f"{epoch_num_str}: No shared chunks or invalid index ({self._current_shared_chunk_index}) for validation. "
                f"Total chunks: {len(self._shared_chunks)}. Train Dataloader might not have been called or "
                "reload_dataloaders_every_n_epochs not set to 1."
            )
            return DataLoader([], batch_size=self.batch_size)

        current_basins_for_epoch = self._shared_chunks[self._current_shared_chunk_index]

        epoch_num = self.trainer.current_epoch if self.trainer else 0
        logger.info(
            f"Epoch {epoch_num}: Validation Dataloader using shared chunk {self._current_shared_chunk_index + 1}/{len(self._shared_chunks)} with {len(current_basins_for_epoch)} basins."
        )

        val_dataset = InMemoryChunkDataset(
            basin_ids=current_basins_for_epoch,
            processed_data_dir=self.processed_time_series_dir,  # type: ignore
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
            logger.warning(f"Validation dataset for chunk {self._current_shared_chunk_index + 1} is empty.")
            return DataLoader(
                val_dataset,  # Pass empty dataset
                batch_size=self.batch_size,
            )

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(val_dataset),
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
            multiprocessing_context=mp.get_context("spawn"),


        )

    def test_dataloader(self) -> DataLoader:
        """
        Creates the test DataLoader. Loads all test basins at once.

        Returns:
            A DataLoader instance for the test set.
        """
        if not self._test_basin_ids:
            logger.warning("No test basin IDs available. Returning empty dataloader.")
            return DataLoader([], batch_size=self.batch_size)

        logger.info(f"Loading test data for {len(self._test_basin_ids)} basins...")
        test_dataset = InMemoryChunkDataset(
            basin_ids=self._test_basin_ids,
            processed_data_dir=self.processed_time_series_dir,  # type: ignore
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
            logger.warning("Test dataset is empty after loading. Returning empty dataloader.")
            return DataLoader(
                test_dataset,  # Pass empty dataset
                batch_size=self.batch_size,
            )

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(test_dataset),
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
            multiprocessing_context=mp.get_context("spawn"),


        )

    def _load_static_data(self) -> None:
        """Loads static attributes from the processed static file into memory."""
        logger.info("Loading static data cache...")
        if self.processed_static_attributes_path and self.processed_static_attributes_path.exists():
            try:
                static_df = pl.read_parquet(self.processed_static_attributes_path)
                required_static_cols = [self.group_identifier] + self.static_features
                missing_cols = [col for col in required_static_cols if col not in static_df.columns]
                if missing_cols:
                    logger.error(
                        f"Static data file missing required columns: {missing_cols}. Static cache may be incomplete."
                    )

                self.static_data_cache = {
                    row[self.group_identifier]: np.array(
                        [row.get(f, 0.0) for f in self.static_features],
                        dtype=np.float32,
                    )
                    for row in static_df.select(
                        [self.group_identifier] + [sf for sf in self.static_features if sf in static_df.columns]
                    ).iter_rows(named=True)
                    if self.group_identifier in row
                }
                logger.info(f"Loaded static data for {len(self.static_data_cache)} basins.")
            except Exception as e:
                logger.error(f"Failed to load processed static data from {self.processed_static_attributes_path}: {e}")
                self.static_data_cache = {}
        else:
            logger.warning("Processed static attributes file not found or not specified. Static cache will be empty.")
            self.static_data_cache = {}

    def _check_and_reuse_existing_processed_data(
        self, run_uuid: str
    ) -> tuple[bool, Optional[LoadedData], Optional[str]]:
        """
        Check if processed data exists for the given run_uuid and load it if valid.
        """
        try:
            run_dir = self.path_to_preprocessing_output_directory / run_uuid
            logger.info(f"Checking for existing processed data at: {run_dir}")

            if not run_dir.is_dir():
                return False, None, "Run directory not found."

            success_marker_path = run_dir / "_SUCCESS"
            if not success_marker_path.exists():
                return False, None, f"_SUCCESS marker not found in {run_dir}."

            required_files = [
                "config.json",
                "quality_summary.json",
                "fitted_time_series_pipelines.joblib",
            ]
            missing_files = [f for f in required_files if not (run_dir / f).exists()]
            if missing_files:
                return False, None, f"Missing required files for reuse: {missing_files}"

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

            config_path = run_dir / "config.json"
            config_result = load_config(config_path)
            if isinstance(config_result, Failure):
                return (
                    False,
                    None,
                    f"Failed to load stored configuration: {config_result.failure()}",
                )

            loaded_config_dict = config_result.unwrap()
            current_config_dict = extract_relevant_config(self)
            # Critical: Ensure current_config_dict also serializes preprocessing_configs by name
            if self.preprocessing_configs:
                current_config_dict["preprocessing_configs"] = {
                    k: {kk: vv.__class__.__name__ if hasattr(vv, "__class__") else str(vv) for kk, vv in v.items()}
                    for k, v in self.preprocessing_configs.items()
                }
            else:
                current_config_dict["preprocessing_configs"] = None

            mismatched_keys = []
            # Normalize Path objects and dicts of Paths for comparison
            for key in current_config_dict.keys():  # Use current_config_dict keys as source of truth for comparison
                loaded_val = loaded_config_dict.get(key)
                current_val = current_config_dict.get(key)

                # Normalize Path objects to strings
                if isinstance(current_val, Path):
                    current_val = str(current_val)
                if isinstance(loaded_val, Path):
                    loaded_val = str(loaded_val)

                # Normalize dicts containing Path objects
                if isinstance(current_val, dict) and any(isinstance(v, Path) for v in current_val.values()):
                    current_val = {
                        k_path: (str(v_path) if isinstance(v_path, Path) else v_path)
                        for k_path, v_path in current_val.items()
                    }
                if isinstance(loaded_val, dict) and any(isinstance(v, Path) for v in loaded_val.values()):
                    loaded_val = {
                        k_path: (str(v_path) if isinstance(v_path, Path) else v_path)
                        for k_path, v_path in loaded_val.items()
                    }

                # Special handling for preprocessing_configs if it's a dict of dicts
                if key == "preprocessing_configs" and isinstance(current_val, dict) and isinstance(loaded_val, dict):
                    # This ensures that the order of keys within inner dicts doesn't matter
                    # And that class names are compared correctly
                    current_pc_norm = (
                        {k_pc: tuple(sorted(v_pc.items())) for k_pc, v_pc in current_val.items()}
                        if current_val
                        else None
                    )
                    loaded_pc_norm = (
                        {k_pc: tuple(sorted(v_pc.items())) for k_pc, v_pc in loaded_val.items()} if loaded_val else None
                    )
                    if current_pc_norm != loaded_pc_norm:
                        mismatched_keys.append(f"{key} (preprocessing_configs structure mismatch)")

                elif loaded_val != current_val:
                    mismatched_keys.append(f"{key} (loaded: {loaded_val}, current: {current_val})")

            if mismatched_keys:
                # Check if the only mismatch is 'preprocessing_configs' which might have been handled specifically
                if not (len(mismatched_keys) == 1 and mismatched_keys[0].startswith("preprocessing_configs")):
                    return (
                        False,
                        None,
                        f"Configuration mismatch: {'; '.join(mismatched_keys)}",
                    )

            summary_path = run_dir / "quality_summary.json"
            with open(summary_path, "r") as f:
                import json

                summary_quality_report_dict = json.load(f)
                summary_quality_report_obj = SummaryQualityReport(**summary_quality_report_dict)

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

            static_pipeline_path = run_dir / "fitted_static_pipeline.joblib"
            processed_static_path = run_dir / "processed_static_features.parquet"
            if static_pipeline_path.exists() and processed_static_path.exists():
                static_pipeline_result = load_static_pipeline(static_pipeline_path)
                if isinstance(static_pipeline_result, Failure):
                    logger.warning(f"Found static pipeline file but failed to load: {static_pipeline_result.failure()}")
                    processed_static_path = None
                else:
                    fitted_pipelines_dict["static"] = static_pipeline_result.unwrap()
            else:
                processed_static_path = None

            loaded_data = LoadedData(
                summary_quality_report=summary_quality_report_obj,
                fitted_pipelines=fitted_pipelines_dict,
                processed_ts_dir=run_dir / "processed_time_series",
                processed_static_path=processed_static_path,
            )

            logger.info(f"Successfully validated and prepared to reuse data from {run_dir}")
            return True, loaded_data, None

        except Exception as e:
            import traceback

            logger.error(f"Error checking for existing processed data: {e}\n{traceback.format_exc()}")
            return False, None, f"Unexpected error during reuse check: {e}"

    def _load_pipelines_from_output(self, processing_output: ProcessingOutput) -> None:
        """Loads fitted pipelines from the paths specified in ProcessingOutput."""
        self.fitted_pipelines = {}
        ts_pipelines_path = processing_output.fitted_time_series_pipelines_path
        if ts_pipelines_path and ts_pipelines_path.exists():
            ts_pipelines_result = load_time_series_pipelines(ts_pipelines_path)
            if isinstance(ts_pipelines_result, Success):
                self.fitted_pipelines.update(ts_pipelines_result.unwrap())
            else:
                raise RuntimeError(f"Failed to load time series pipelines: {ts_pipelines_result.failure()}")
        else:
            logger.warning("Fitted time series pipelines file not found.")

        if processing_output.fitted_static_pipeline_path and processing_output.fitted_static_pipeline_path.exists():
            static_pipeline_path = processing_output.fitted_static_pipeline_path
            static_pipeline_result = load_static_pipeline(static_pipeline_path)
            if isinstance(static_pipeline_result, Success):
                self.fitted_pipelines["static"] = static_pipeline_result.unwrap()
            else:
                logger.warning(f"Failed to load static pipeline: {static_pipeline_result.failure()}")
        else:
            logger.info("No fitted static pipeline found or specified in processing output.")
        logger.info(f"Successfully loaded {len(self.fitted_pipelines)} categories of fitted pipelines.")

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        basin_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Inverse transform predictions using the fitted 'target' pipeline.
        """
        if "target" not in self.fitted_pipelines:
            raise RuntimeError("No 'target' pipeline found. Was prepare_data() called and did it succeed?")
        target_pipeline = self.fitted_pipelines["target"]
        if not isinstance(target_pipeline, GroupedPipeline):
            raise RuntimeError("Expected 'target' pipeline to be a GroupedPipeline.")
        if not target_pipeline.fitted_pipelines:
            raise RuntimeError("The 'target' GroupedPipeline has not been fitted.")

        if predictions.shape[0] != basin_ids.shape[0]:
            raise ValueError(
                f"Shape mismatch: predictions ({predictions.shape[0]}) and basin_ids ({basin_ids.shape[0]})"
            )

        orig_shape = predictions.shape
        all_basin_ids_repeated = []
        all_predictions_flat = []

        for i in range(predictions.shape[0]):
            current_basin_id = str(basin_ids[i])
            pred_values = predictions[i].ravel()
            all_basin_ids_repeated.extend([current_basin_id] * len(pred_values))
            all_predictions_flat.extend(pred_values)

        df_to_transform = pl.DataFrame(
            {
                self.group_identifier: all_basin_ids_repeated,
                self.target: all_predictions_flat,
            }
        )

        unique_request_gids = set(map(str, basin_ids))
        missing_gids = unique_request_gids - set(
            map(str, target_pipeline.fitted_pipelines.keys())  # Ensure comparison with string keys
        )
        if missing_gids:
            raise ValueError(f"Inverse transform failed: No fitted pipeline for basins: {sorted(list(missing_gids))}")

        try:
            # GroupedPipeline expects pandas DataFrame
            inv_df_pd = target_pipeline.inverse_transform(df_to_transform.to_pandas())
            inv_vals = inv_df_pd[self.target].values
        except Exception as e:
            logger.error(f"Error during inverse_transform: {e}")
            raise RuntimeError("Inverse transformation failed.") from e

        try:
            reshaped_inv_vals = inv_vals.reshape(orig_shape)
        except ValueError as e:
            logger.error(
                f"Error reshaping inverse transformed values. Original shape: {orig_shape}, Inv_vals shape: {inv_vals.shape}"
            )
            raise RuntimeError("Failed to reshape inverse transformed predictions.") from e
        return reshaped_inv_vals

    def get_train_dataloader(self) -> DataLoader:
        return self.train_dataloader()

    def get_val_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def get_test_dataloader(self) -> DataLoader:
        return self.test_dataloader()
