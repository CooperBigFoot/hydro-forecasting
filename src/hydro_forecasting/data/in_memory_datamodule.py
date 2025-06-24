import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import torch
from pytorch_lightning import LightningDataModule
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from ..exceptions import (
    ConfigurationError,
    DataProcessingError,
    FileOperationError,
    PipelineCompatibilityError,
)
from ..experiment_utils.seed_manager import SeedManager
from ..preprocessing.grouped import GroupedPipeline
from ..preprocessing.static_preprocessing import (
    load_static_pipeline,
)
from ..preprocessing.time_series_preprocessing import load_time_series_pipelines
from ..preprocessing.unified import UnifiedPipeline
from .clean_data import SummaryQualityReport
from .config_utils import (
    extract_pipeline_metadata,
    extract_relevant_config,
    generate_run_uuid,
    load_config,
)
from .datamodule_validators import validate_hydro_inmemory_datamodule_config
from .in_memory_dataset import InMemoryChunkDataset, find_valid_sequences
from .preprocessing import (
    ProcessingOutput,
    run_hydro_processor,
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class LoadedData:
    """Data structure for loaded preprocessing artifacts."""

    summary_quality_report: SummaryQualityReport
    fitted_pipelines: dict[str, Pipeline | GroupedPipeline]
    processed_ts_dir: Path
    processed_static_path: Path | None


class HydroInMemoryDataModule(LightningDataModule):
    """
    A PyTorch Lightning DataModule that loads data in chunks into memory.
    Implements synchronized, memory-efficient chunking for train data.
    Validation data is loaded once into a fixed, cached pool.
    Training chunks are recomputed (reshuffled and re-partitioned) every time the
    current set of training chunks is exhausted.

    Raises:
        ValueError: If configuration validation fails.
        RuntimeError: If data processing operations fail.
        FileOperationError: If file operations fail.
        DataQualityError: If data quality checks fail.
    """

    def __init__(
        self,
        region_time_series_base_dirs: dict[str, str | Path],
        region_static_attributes_base_dirs: dict[str, str | Path],
        path_to_preprocessing_output_directory: str | Path,
        group_identifier: str,
        batch_size: int,
        input_length: int,
        output_length: int,
        forcing_features: list[str],
        static_features: list[str],
        target: str,
        preprocessing_configs: dict[str, dict[str, Any]],
        num_workers: int,
        min_train_years: float,
        train_prop: float,
        val_prop: float,
        test_prop: float,
        max_imputation_gap_size: int,
        chunk_size: int,
        validation_chunk_size: int | None = None,
        list_of_gauge_ids_to_process: list[str] | None = None,
        domain_id: str = "source",
        domain_type: str = "source",
        is_autoregressive: bool = False,
        include_input_end_date_in_batch: bool = True,
        random_seed: int | None = None,
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
            chunk_size: Number of basins to load into memory per shared training chunk.
            validation_chunk_size: Number of unique basins for the validation set.
                                   Defaults to 2 * chunk_size if None.
            list_of_gauge_ids_to_process: Optional list to explicitly define which basins to process/use.
            domain_id: Identifier for the data domain.
            domain_type: Type of domain ('source' or 'target').
            is_autoregressive: Whether the model uses past target values as input features.
            include_input_end_date_in_batch: If True, 'input_end_date' (ms timestamp or None)
                                              will be included in the batch dictionary.
            random_seed: Seed for reproducible random operations. If None, operations will be non-deterministic.

        Raises:
            ValueError: If configuration validation fails.
        """
        super().__init__()
        self.preprocessing_configs = self._adapt_preprocessing_configs(preprocessing_configs)

        effective_validation_chunk_size = validation_chunk_size if validation_chunk_size is not None else chunk_size * 2

        self.validation_chunk_size_for_validation_only = effective_validation_chunk_size

        self.save_hyperparameters(ignore=["preprocessing_configs"])

        if self.hparams.validation_chunk_size is None:
            self.hparams.validation_chunk_size = effective_validation_chunk_size

        try:
            validate_hydro_inmemory_datamodule_config(self)
        except ConfigurationError as e:
            raise ValueError(f"HydroInMemoryDataModule configuration error: {e}") from e

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
        self.include_input_end_date_in_batch = self.hparams.include_input_end_date_in_batch

        self._prepare_data_has_run: bool = False
        self.run_uuid: str | None = None
        self.summary_quality_report: SummaryQualityReport | None = None
        self.fitted_pipelines: dict[str, Pipeline | GroupedPipeline | UnifiedPipeline] = {}
        self.processed_time_series_dir: Path | None = None
        self.processed_static_attributes_path: Path | None = None
        self.static_data_cache: dict[str, torch.Tensor] = {}

        self.chunkable_basin_ids: list[str] = []
        self._shared_chunks: list[list[str]] = []  # For training
        self._current_shared_chunk_index: int = -1  # For training

        # Validation specific attributes
        self._validation_gauge_id_pool: list[str] | None = None
        self._cached_validation_column_tensors: dict[str, torch.Tensor] | None = None
        self._cached_validation_index_entries: list[tuple[str, int, int]] | None = None
        self._cached_validation_basin_row_map: dict[str, tuple[int, int]] | None = None

        self._test_basin_ids: list[str] = []

        if self.hparams.is_autoregressive:
            self.input_features_ordered_for_X = [self.hparams.target] + sorted(
                [f for f in self.hparams.forcing_features if f != self.hparams.target]
            )
        else:
            self.input_features_ordered_for_X = sorted(self.hparams.forcing_features)

        self.future_features_ordered = sorted(set(self.hparams.forcing_features))

        self.seed_manager = SeedManager(random_seed)
        if random_seed is not None:
            self.seed_manager.set_global_seeds()
            logger.info(f"Initialized SeedManager with seed: {random_seed}")
        else:
            logger.info("Initialized SeedManager without seed (non-deterministic mode)")

    def _adapt_preprocessing_configs(
        self, preprocessing_configs: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """
        Adapt preprocessing configs to ensure they follow the right schema with strategy support.

        Args:
            preprocessing_configs: Original preprocessing configuration

        Returns:
            Adapted preprocessing configuration with strategy keys
        """
        adapted_configs = {}

        for pipeline_name, config in preprocessing_configs.items():
            if isinstance(config, dict):
                # Check if config already has the new schema
                if "strategy" in config:
                    # Already in new format, validate it
                    if config["strategy"] not in ["per_group", "unified"]:
                        raise ConfigurationError(
                            f"Invalid strategy '{config['strategy']}' for {pipeline_name}. Must be 'per_group' or 'unified'"
                        )
                    adapted_configs[pipeline_name] = config
                else:
                    # Old format, adapt it based on pipeline type
                    pipeline = config.get("pipeline")
                    if pipeline is None:
                        raise ConfigurationError(
                            f"Pipeline configuration for '{pipeline_name}' must include 'pipeline' key"
                        )

                    # Infer strategy from pipeline type
                    if isinstance(pipeline, GroupedPipeline):
                        strategy = "per_group"
                    elif isinstance(pipeline, UnifiedPipeline):
                        strategy = "unified"
                    elif isinstance(pipeline, Pipeline):
                        # sklearn Pipeline could be either - default to unified for static features
                        strategy = "unified" if pipeline_name == "static_features" else "per_group"
                    else:
                        raise ConfigurationError(f"Unknown pipeline type for '{pipeline_name}': {type(pipeline)}")

                    # Create adapted config with strategy
                    adapted_config = config.copy()
                    adapted_config["strategy"] = strategy
                    adapted_configs[pipeline_name] = adapted_config
            else:
                raise ConfigurationError(f"Invalid configuration format for '{pipeline_name}'")

        return adapted_configs

        # remove temp attribute used for validation
        if hasattr(self, "validation_chunk_size_for_validation_only"):
            delattr(self, "validation_chunk_size_for_validation_only")

    def prepare_data(self) -> None:
        """
        Prepares data for training by running preprocessing or loading existing processed data.

        Raises:
            RuntimeError: If data processing fails or no basins remain after preprocessing.
        """
        if self._prepare_data_has_run:
            logger.info("Data preparation has already run.")
            return

        logger.info("Starting data preparation...")
        current_config_dict = extract_relevant_config(self)

        if self.preprocessing_configs:
            serialized_configs = {}
            for k, v in self.preprocessing_configs.items():
                serialized_config = {}
                for kk, vv in v.items():
                    if kk == "pipeline":
                        # Extract pipeline metadata instead of full serialization
                        serialized_config["pipeline_steps"] = extract_pipeline_metadata(vv)
                    elif kk == "columns" and isinstance(vv, list):
                        # Preserve column lists as-is
                        serialized_config[kk] = vv
                    else:
                        # For other values (strategy, etc.), convert to string
                        serialized_config[kk] = str(vv)
                serialized_configs[k] = serialized_config
            current_config_dict["preprocessing_configs"] = serialized_configs
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
            try:
                processing_output = run_hydro_processor(
                    region_time_series_base_dirs=self.region_time_series_base_dirs,
                    region_static_attributes_base_dirs=self.region_static_attributes_base_dirs,
                    path_to_preprocessing_output_directory=self.path_to_preprocessing_output_directory,
                    required_columns=list(set(self.forcing_features + [self.target])),
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
                    basin_batch_size=50,  # Consider making this configurable
                    random_seed=self.seed_manager.get_component_seed("preprocessing_unified"),
                )
            except DataProcessingError as e:
                raise RuntimeError(f"Hydro processor failed: {e}") from e
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
        if not self.processed_time_series_dir:
            logger.error("Processed time series directory not set. Cannot determine split IDs.")
            return

        train_dir = self.processed_time_series_dir / "train"
        val_dir = self.processed_time_series_dir / "val"
        test_dir = self.processed_time_series_dir / "test"

        self.chunkable_basin_ids = []
        self._test_basin_ids = []

        for basin_id in processed_basins_from_report:
            train_file = train_dir / f"{basin_id}.parquet"
            if train_file.exists():  # Check train file for chunkable_basin_ids
                val_file = val_dir / f"{basin_id}.parquet"  # Also ensure val file exists for it
                if val_file.exists():
                    self.chunkable_basin_ids.append(basin_id)

            test_file = test_dir / f"{basin_id}.parquet"
            if test_file.exists():
                self._test_basin_ids.append(basin_id)

        with self.seed_manager.temporary_seed("basin_id_shuffle", "datamodule_setup"):
            import random

            random.shuffle(self.chunkable_basin_ids)

        logger.info(
            f"Found {len(self.chunkable_basin_ids)} basins for synchronized train/val chunking and validation pool selection."
        )
        logger.info(f"Found {len(self._test_basin_ids)} basins for test split.")

        if not self.chunkable_basin_ids:
            logger.warning(
                "No basins available for training chunks or validation pool. Training/validation might be empty."
            )

    def setup(self, stage: str | None = None) -> None:
        if not self._prepare_data_has_run or not self.processed_time_series_dir:
            raise RuntimeError("prepare_data() must be called before setup()")

        if not self.static_data_cache:  # Load static data if not already loaded
            self._load_static_data()

        if stage == "fit" or stage is None:
            # 1. Setup Validation Pool and Cache (once)
            if self._validation_gauge_id_pool is None and self.chunkable_basin_ids:
                num_val_basins = min(cast(int, self.hparams.validation_chunk_size), len(self.chunkable_basin_ids))
                with self.seed_manager.temporary_seed("validation_pool_selection", "datamodule_setup"):
                    import random

                    self._validation_gauge_id_pool = random.sample(self.chunkable_basin_ids, num_val_basins)
                logger.info(
                    f"Created fixed validation pool with {len(self._validation_gauge_id_pool)} basins: {self._validation_gauge_id_pool[:5]}..."
                )  # Log first 5

            if self._cached_validation_column_tensors is None and self._validation_gauge_id_pool:
                logger.info(f"Loading and caching data for {len(self._validation_gauge_id_pool)} validation basins...")
                (
                    tensors,
                    indices,
                    row_map,
                ) = self._load_and_tensorize_chunk(self._validation_gauge_id_pool, "val")

                if tensors and indices and row_map:
                    self._cached_validation_column_tensors = tensors
                    self._cached_validation_index_entries = indices
                    self._cached_validation_basin_row_map = row_map
                    logger.info(
                        f"Successfully cached validation data. Index entries: {len(self._cached_validation_index_entries)}."
                    )
                else:
                    logger.error(f"Failed to load and cache validation data for pool: {self._validation_gauge_id_pool}")

            # 2. Setup Training Chunks (can be re-initialized per full pass)
            if not self.chunkable_basin_ids:
                logger.warning("No chunkable_basin_ids for training. Shared training chunks will be empty.")
                self._shared_chunks = []
            elif not self._shared_chunks:  # Only initialize if empty (first time or after recompute)
                self._initialize_and_partition_shared_chunks()
                logger.info(
                    f"Initial training shared chunks: {len(self._shared_chunks)} chunks of approx size {self.hparams.chunk_size} from {len(self.chunkable_basin_ids)} basins."
                )

    def _initialize_and_partition_shared_chunks(self) -> None:
        """Initializes or re-initializes training chunks."""
        if not self.chunkable_basin_ids:
            logger.error("Cannot initialize training chunks: chunkable_basin_ids is empty.")
            self._shared_chunks = []
            return

        logger.info(f"Initializing/Re-initializing training shared chunks from {len(self.chunkable_basin_ids)} basins.")
        # It's important to shuffle the basin IDs each time we re-partition
        # to ensure different chunk compositions over full passes.
        with self.seed_manager.temporary_seed("training_basin_shuffle", "training_chunks"):
            import random

            shuffled_training_basin_ids = random.sample(self.chunkable_basin_ids, len(self.chunkable_basin_ids))

        self._shared_chunks = [
            shuffled_training_basin_ids[i : i + self.hparams.chunk_size]
            for i in range(0, len(shuffled_training_basin_ids), self.hparams.chunk_size)
        ]
        self._shared_chunks = [chunk for chunk in self._shared_chunks if chunk]  # Remove empty chunks if any
        self._current_shared_chunk_index = -1  # Reset index for the new set of chunks
        logger.info(f"Created {len(self._shared_chunks)} training shared chunks.")

    def _advance_and_recompute_shared_chunks_if_needed(self) -> None:
        """Advances training chunk index and recomputes chunks if exhausted."""
        if not self._shared_chunks:  # No training chunks to advance
            return

        self._current_shared_chunk_index += 1
        if self._current_shared_chunk_index >= len(self._shared_chunks):
            logger.info("Completed full pass through training shared chunks. Recomputing.")
            self._initialize_and_partition_shared_chunks()  # This will shuffle and re-partition
            self._current_shared_chunk_index = 0  # Start from the first of the new chunks
            if not self._shared_chunks:
                logger.warning("Re-initialization of training shared chunks resulted in an empty list.")
                return

    def _load_and_tensorize_chunk(
        self, basin_ids_for_chunk: list[str], stage: str
    ) -> tuple[dict[str, torch.Tensor] | None, list[tuple[str, int, int]] | None, dict[str, tuple[int, int]] | None]:
        if not self.processed_time_series_dir or not basin_ids_for_chunk:
            logger.warning(
                f"Cannot load chunk for stage '{stage}': No processed time series dir or no basin IDs provided."
            )
            return None, None, None

        stage_dir = self.processed_time_series_dir / stage
        if not stage_dir.exists():
            logger.error(f"Stage directory not found: {stage_dir}")
            return None, None, None

        columns_to_select = [self.target] + self.forcing_features
        if self.include_input_end_date_in_batch:
            columns_to_select.append("date")
        columns_to_select = sorted(set(columns_to_select))

        dfs_to_concat: list[pl.DataFrame] = []
        for basin_id in basin_ids_for_chunk:
            file_path = stage_dir / f"{basin_id}.parquet"
            if file_path.exists():
                try:
                    df_scan = pl.scan_parquet(str(file_path))

                    # Ensure all selected columns exist, select them, and cast date
                    available_cols_in_scan = df_scan.collect_schema().names()
                    final_cols_to_select_for_file = [col for col in columns_to_select if col in available_cols_in_scan]

                    if not final_cols_to_select_for_file:
                        logger.warning(f"No columns from {columns_to_select} found in {file_path}. Skipping this file.")
                        continue

                    df_select = df_scan.select(final_cols_to_select_for_file)

                    if "date" in final_cols_to_select_for_file:  # Ensure date is datetime if present
                        df_select = df_select.with_columns(pl.col("date").cast(pl.Datetime("us")))

                    df = df_select.sort(
                        "date" if "date" in final_cols_to_select_for_file else final_cols_to_select_for_file[0]
                    ).collect()

                    dfs_to_concat.append(df.with_columns(pl.lit(basin_id).alias(self.group_identifier)))
                except Exception as e:
                    logger.warning(f"Could not load or process file {file_path} for stage {stage}: {e}")
            else:
                logger.warning(f"File not found for stage {stage}, skipping: {file_path}")

        if not dfs_to_concat:
            logger.error(f"No valid data loaded for chunk in stage '{stage}' with basin IDs: {basin_ids_for_chunk}.")
            return None, None, None

        try:
            chunk_df_with_ids = pl.concat(dfs_to_concat, how="vertical_relaxed")  # Use relaxed for safety
            logger.info(
                f"Stage '{stage}' chunk data loaded for {len(basin_ids_for_chunk)} basins. Shape: {chunk_df_with_ids.shape}. Est. Mem: {chunk_df_with_ids.estimated_size('mb'):.2f} MB"
            )
        except Exception as e:
            logger.error(f"Failed to concatenate dataframes for chunk in stage '{stage}': {e}")
            return None, None, None

        del dfs_to_concat
        gc.collect()

        index_entries: list[tuple[str, int, int]] = []
        basin_row_map: dict[str, tuple[int, int]] = {}
        current_absolute_start_row = 0

        for basin_id_in_order in sorted(chunk_df_with_ids[self.group_identifier].unique().to_list()):
            basin_specific_df_from_concat = chunk_df_with_ids.filter(pl.col(self.group_identifier) == basin_id_in_order)

            if basin_specific_df_from_concat.height == 0:
                continue

            basin_row_map[basin_id_in_order] = (current_absolute_start_row, basin_specific_df_from_concat.height)

            cols_for_valid_seq = [self.target] + self.forcing_features
            basin_df_for_indexing = basin_specific_df_from_concat.select(
                [col for col in ["date"] + cols_for_valid_seq if col in basin_specific_df_from_concat.columns]
            )

            try:
                positions, _ = find_valid_sequences(
                    basin_df_for_indexing,
                    self.input_length,
                    self.output_length,
                    self.target,
                    self.forcing_features,
                )
                for start_idx_relative_to_basin in positions:
                    index_entries.append(
                        (
                            basin_id_in_order,  # Use basin_id from the ordered unique list
                            int(start_idx_relative_to_basin),
                            int(start_idx_relative_to_basin + self.input_length + self.output_length),
                        )
                    )
            except DataProcessingError as e:
                logger.warning(f"Could not find valid sequences for basin {basin_id_in_order} in stage '{stage}': {e}")
            current_absolute_start_row += basin_specific_df_from_concat.height

        if not index_entries:
            logger.warning(f"No valid sequences found for any basin in the current chunk for stage '{stage}'.")
            return None, None, None

        # Tensorize only columns that were actually loaded and are needed.
        # `columns_to_select` is the superset of what might be needed.
        # `chunk_df_with_ids.columns` tells us what was actually loaded.
        columns_present_in_chunk = chunk_df_with_ids.columns
        columns_to_tensorize = [
            col for col in columns_to_select if col in columns_present_in_chunk and col != self.group_identifier
        ]

        df_for_tensor_conversion = chunk_df_with_ids.select(columns_to_tensorize)
        chunk_column_tensors: dict[str, torch.Tensor] = {}

        if "date" in df_for_tensor_conversion.columns and self.include_input_end_date_in_batch:
            date_col_for_tensor = df_for_tensor_conversion.get_column("date")
            if date_col_for_tensor.dtype == pl.Datetime:
                date_timestamps_ms = date_col_for_tensor.dt.timestamp(time_unit="ms").cast(pl.Int64)
                chunk_column_tensors["date"] = torch.from_numpy(date_timestamps_ms.to_numpy())
            else:
                logger.warning("'date' column is not of Datetime type, cannot convert to timestamp for tensor.")
            df_for_tensor_conversion = df_for_tensor_conversion.drop("date")

        # Tensorize remaining numeric columns
        if df_for_tensor_conversion.width > 0:
            try:
                # Ensure all columns for to_torch are numeric before conversion
                numeric_df_for_tensor = df_for_tensor_conversion.select(
                    [pl.col(c).cast(pl.Float32, strict=False) for c in df_for_tensor_conversion.columns]
                )
                other_tensors_dict = numeric_df_for_tensor.to_torch(return_type="dict")
                chunk_column_tensors.update(other_tensors_dict)
            except Exception as e:
                logger.error(
                    f"Error during Polars to_torch conversion for stage '{stage}': {e}. Columns: {df_for_tensor_conversion.columns}, Dtypes: {df_for_tensor_conversion.dtypes}"
                )
                return None, None, None

        return chunk_column_tensors, index_entries, basin_row_map

    def train_dataloader(self) -> DataLoader:
        self._advance_and_recompute_shared_chunks_if_needed()  # Handles training chunk advancement/recompute

        if (
            not self._shared_chunks
            or self._current_shared_chunk_index < 0
            or self._current_shared_chunk_index >= len(self._shared_chunks)
        ):
            current_epoch = getattr(self.trainer, "current_epoch", "N/A")
            logger.error(
                f"Epoch {current_epoch}: No valid training shared chunk available. Cannot create train_dataloader."
            )
            return DataLoader([], batch_size=self.batch_size)

        current_basins_for_train_chunk = self._shared_chunks[self._current_shared_chunk_index]
        current_epoch = getattr(self.trainer, "current_epoch", 0)
        logger.info(
            f"Epoch {current_epoch}: Train Dataloader using chunk {self._current_shared_chunk_index + 1}/{len(self._shared_chunks)} with {len(current_basins_for_train_chunk)} basins."
        )

        chunk_column_tensors, index_entries, basin_row_map = self._load_and_tensorize_chunk(
            current_basins_for_train_chunk, "train"
        )

        if not chunk_column_tensors or not index_entries or not basin_row_map:
            logger.warning(
                f"Train Dataloader: Failed to load or process training chunk {self._current_shared_chunk_index + 1}. Returning empty DataLoader."
            )
            return DataLoader([], batch_size=self.batch_size)

        with self.seed_manager.temporary_seed("training_sequence_shuffle", "sequence_operations"):
            import random

            random.shuffle(index_entries)  # Shuffle index_entries for training

        train_dataset = InMemoryChunkDataset(
            chunk_column_tensors=chunk_column_tensors,
            static_data_cache=self.static_data_cache,
            index_entries=index_entries,
            basin_row_map=basin_row_map,
            input_length=self.input_length,
            output_length=self.output_length,
            target_name=self.target,
            forcing_features_names=self.forcing_features,
            static_features_names=self.static_features,
            group_identifier_name=self.group_identifier,
            is_autoregressive=self.is_autoregressive,
            input_features_ordered=self.input_features_ordered_for_X,
            future_features_ordered=self.future_features_ordered,
            include_input_end_date=self.include_input_end_date_in_batch,
        )

        if len(train_dataset) == 0:
            logger.warning(f"Training dataset for chunk {self._current_shared_chunk_index + 1} is empty.")

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(train_dataset),  # Use RandomSampler as index_entries are already shuffled per chunk
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=self.seed_manager.worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        current_epoch = getattr(self.trainer, "current_epoch", "N/A")
        if (
            not self._cached_validation_column_tensors
            or not self._cached_validation_index_entries
            or not self._cached_validation_basin_row_map
        ):
            logger.error(
                f"Epoch {current_epoch}: Validation data not cached. Cannot create val_dataloader. "
                "Ensure setup() was called correctly and data was loaded."
            )
            return DataLoader([], batch_size=self.batch_size)

        logger.info(
            f"Epoch {current_epoch}: Val Dataloader using cached validation data with {len(self._cached_validation_index_entries)} samples from {len(self._validation_gauge_id_pool or [])} basins."
        )

        val_dataset = InMemoryChunkDataset(
            chunk_column_tensors=self._cached_validation_column_tensors,
            static_data_cache=self.static_data_cache,
            index_entries=self._cached_validation_index_entries,  # Not shuffled for validation
            basin_row_map=self._cached_validation_basin_row_map,
            input_length=self.input_length,
            output_length=self.output_length,
            target_name=self.target,
            forcing_features_names=self.forcing_features,
            static_features_names=self.static_features,
            group_identifier_name=self.group_identifier,
            is_autoregressive=self.is_autoregressive,
            input_features_ordered=self.input_features_ordered_for_X,
            future_features_ordered=self.future_features_ordered,
            include_input_end_date=self.include_input_end_date_in_batch,
        )

        if len(val_dataset) == 0:
            logger.warning(f"Validation dataset (from cache) is empty for epoch {current_epoch}.")

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(val_dataset),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=self.seed_manager.worker_init_fn,
        )

    def test_dataloader(self) -> DataLoader:
        if not self._test_basin_ids:
            logger.warning("No test basin IDs available. Returning empty test dataloader.")
            return DataLoader([], batch_size=self.batch_size)

        logger.info(f"Loading test data for {len(self._test_basin_ids)} basins...")
        chunk_column_tensors, index_entries, basin_row_map = self._load_and_tensorize_chunk(
            self._test_basin_ids, "test"
        )

        if not chunk_column_tensors or not index_entries or not basin_row_map:
            logger.warning("Test Dataloader: Failed to load or process test data. Returning empty DataLoader.")
            return DataLoader([], batch_size=self.batch_size)

        test_dataset = InMemoryChunkDataset(
            chunk_column_tensors=chunk_column_tensors,
            static_data_cache=self.static_data_cache,
            index_entries=index_entries,
            basin_row_map=basin_row_map,
            input_length=self.input_length,
            output_length=self.output_length,
            target_name=self.target,
            forcing_features_names=self.forcing_features,
            static_features_names=self.static_features,
            group_identifier_name=self.group_identifier,
            is_autoregressive=self.is_autoregressive,
            input_features_ordered=self.input_features_ordered_for_X,
            future_features_ordered=self.future_features_ordered,
            include_input_end_date=self.include_input_end_date_in_batch,
        )

        if len(test_dataset) == 0:
            logger.warning("Test dataset is empty after loading.")

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(test_dataset),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=self.seed_manager.worker_init_fn,
        )

    def _load_static_data(self) -> None:
        """Loads static attributes and converts them to PyTorch Tensors."""
        logger.info("Loading static data cache and converting to Tensors...")
        if self.processed_static_attributes_path and self.processed_static_attributes_path.exists():
            try:
                static_df = pl.read_parquet(self.processed_static_attributes_path)
                required_static_cols = [self.hparams.group_identifier] + self.hparams.static_features
                missing_cols = [col for col in required_static_cols if col not in static_df.columns]
                if missing_cols:
                    logger.error(f"Static data file missing required columns: {missing_cols}.")

                # Ensure static_features are sorted for consistent tensor creation
                sorted_static_features = sorted(set(self.hparams.static_features))

                temp_cache: dict[str, np.ndarray] = {}
                for row in static_df.select(
                    [self.hparams.group_identifier] + [sf for sf in sorted_static_features if sf in static_df.columns]
                ).iter_rows(named=True):
                    basin_id = row[self.hparams.group_identifier]
                    if basin_id:
                        # Create array with NaNs where features are missing, then fill with 0.0
                        feature_values = np.full(len(sorted_static_features), np.nan, dtype=np.float32)
                        for i, feature_name in enumerate(sorted_static_features):
                            if feature_name in row:
                                feature_values[i] = row.get(feature_name, np.nan)

                        # Convert NaNs to 0.0 after collecting all values for the row
                        feature_values = np.nan_to_num(feature_values, nan=0.0)
                        temp_cache[basin_id] = feature_values

                # Convert numpy arrays to tensors
                self.static_data_cache = {bid: torch.from_numpy(arr) for bid, arr in temp_cache.items()}
                logger.info(f"Loaded and tensorized static data for {len(self.static_data_cache)} basins.")
            except Exception as e:
                logger.error(f"Failed to load/tensorize static data from {self.processed_static_attributes_path}: {e}")
                self.static_data_cache = {}
        else:
            logger.warning("Processed static attributes file not found. Static cache empty.")
            self.static_data_cache = {}

    def _check_and_reuse_existing_processed_data(self, run_uuid: str) -> tuple[bool, LoadedData | None, str | None]:
        """
        Check if processed data exists for the given run_uuid and load it if valid.

        Returns:
            Tuple of (success, loaded_data, error_message) where success indicates
            if data can be reused, loaded_data contains the artifacts if successful,
            and error_message explains any failure.
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
            try:
                loaded_config_dict = load_config(config_path)
            except FileOperationError as e:
                return (
                    False,
                    None,
                    f"Failed to load stored configuration: {e}",
                )

            current_config_dict = extract_relevant_config(self)
            # Critical: Ensure current_config_dict also serializes preprocessing_configs by name
            if self.preprocessing_configs:
                serialized_configs = {}
                for k, v in self.preprocessing_configs.items():
                    serialized_config = {}
                    for kk, vv in v.items():
                        if kk == "pipeline" and hasattr(vv, "get_params"):
                            # Serialize pipeline using get_params for accurate comparison
                            serialized_config[kk] = {"class": vv.__class__.__name__, "params": vv.get_params(deep=True)}
                        elif hasattr(vv, "__class__"):
                            serialized_config[kk] = vv.__class__.__name__
                        else:
                            serialized_config[kk] = str(vv)
                    serialized_configs[k] = serialized_config
                current_config_dict["preprocessing_configs"] = serialized_configs
            else:
                current_config_dict["preprocessing_configs"] = None

            mismatched_keys = []
            for key in current_config_dict:
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

                if key == "preprocessing_configs" and isinstance(current_val, dict) and isinstance(loaded_val, dict):
                    # Deep comparison of preprocessing configs including pipeline params
                    configs_match = True

                    # Check if both have same pipeline names
                    if set(current_val.keys()) != set(loaded_val.keys()):
                        configs_match = False
                    else:
                        # Compare each pipeline config
                        for pipeline_name in current_val.keys():
                            current_pipeline_config = current_val[pipeline_name]
                            loaded_pipeline_config = loaded_val[pipeline_name]

                            # Compare all non-pipeline keys
                            for config_key in set(current_pipeline_config.keys()) | set(loaded_pipeline_config.keys()):
                                if config_key == "pipeline":
                                    # Special handling for pipeline comparison
                                    current_pipeline = current_pipeline_config.get(config_key)
                                    loaded_pipeline = loaded_pipeline_config.get(config_key)

                                    # Both should be dicts with 'class' and 'params' if serialized correctly
                                    if isinstance(current_pipeline, dict) and isinstance(loaded_pipeline, dict):
                                        if current_pipeline.get("class") != loaded_pipeline.get("class"):
                                            configs_match = False
                                            break
                                        # Compare params (deep comparison)
                                        if current_pipeline.get("params") != loaded_pipeline.get("params"):
                                            configs_match = False
                                            break
                                    elif current_pipeline != loaded_pipeline:
                                        configs_match = False
                                        break
                                else:
                                    # Compare other config values normally
                                    if current_pipeline_config.get(config_key) != loaded_pipeline_config.get(
                                        config_key
                                    ):
                                        configs_match = False
                                        break

                            if not configs_match:
                                break

                    if not configs_match:
                        mismatched_keys.append(f"{key} (preprocessing_configs structure or params mismatch)")

                elif loaded_val != current_val:
                    mismatched_keys.append(f"{key} (loaded: {loaded_val}, current: {current_val})")

            if mismatched_keys and not (
                len(mismatched_keys) == 1 and mismatched_keys[0].startswith("preprocessing_configs")
            ):
                return (
                    False,
                    None,
                    f"Configuration mismatch: {'; '.join(mismatched_keys)}",
                )

            summary_path = run_dir / "quality_summary.json"
            with open(summary_path) as f:
                import json

                summary_quality_report_dict = json.load(f)
                summary_quality_report_obj = SummaryQualityReport(**summary_quality_report_dict)

            fitted_pipelines_dict = {}
            ts_pipelines_path = run_dir / "fitted_time_series_pipelines.joblib"
            try:
                ts_pipelines = load_time_series_pipelines(ts_pipelines_path)
                fitted_pipelines_dict.update(ts_pipelines)
            except (FileOperationError, PipelineCompatibilityError) as e:
                return (
                    False,
                    None,
                    f"Failed to load time series pipelines: {e}",
                )

            static_pipeline_path = run_dir / "fitted_static_pipeline.joblib"
            processed_static_path = run_dir / "processed_static_features.parquet"
            if static_pipeline_path.exists() and processed_static_path.exists():
                try:
                    static_pipeline = load_static_pipeline(static_pipeline_path)
                    fitted_pipelines_dict["static"] = static_pipeline
                except (FileOperationError, PipelineCompatibilityError) as e:
                    logger.warning(f"Found static pipeline file but failed to load: {e}")
                    processed_static_path = None
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
        """
        Loads fitted pipelines from the paths specified in ProcessingOutput.

        Raises:
            RuntimeError: If loading time series pipelines fails.
        """
        self.fitted_pipelines = {}
        ts_pipelines_path = processing_output.fitted_time_series_pipelines_path
        if ts_pipelines_path and ts_pipelines_path.exists():
            try:
                ts_pipelines = load_time_series_pipelines(ts_pipelines_path)
                self.fitted_pipelines.update(ts_pipelines)
            except (FileOperationError, PipelineCompatibilityError) as e:
                raise RuntimeError(f"Failed to load time series pipelines: {e}")
        else:
            logger.warning("Fitted time series pipelines file not found.")

        if processing_output.fitted_static_pipeline_path and processing_output.fitted_static_pipeline_path.exists():
            static_pipeline_path = processing_output.fitted_static_pipeline_path
            try:
                static_pipeline = load_static_pipeline(static_pipeline_path)
                self.fitted_pipelines["static"] = static_pipeline
            except (FileOperationError, PipelineCompatibilityError) as e:
                logger.warning(f"Failed to load static pipeline: {e}")
        else:
            logger.info("No fitted static pipeline found or specified in processing output.")
        logger.info(f"Successfully loaded {len(self.fitted_pipelines)} categories of fitted pipelines.")

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        basin_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Inverse transform predictions using the fitted target pipeline.

        Args:
            predictions: Array of predictions with shape (n_samples, forecast_horizon)
            basin_ids: Array of basin identifiers corresponding to each prediction

        Returns:
            Inverse transformed predictions with same shape as input

        Raises:
            RuntimeError: If target pipeline is missing or not fitted
            ValueError: If input validation fails or basin pipelines missing
        """
        # Validate target pipeline exists
        if "target" not in self.fitted_pipelines:
            raise RuntimeError("No 'target' pipeline found. Was prepare_data() called successfully?")

        target_pipeline = self.fitted_pipelines["target"]
        if not isinstance(target_pipeline, GroupedPipeline):
            raise RuntimeError("Expected 'target' pipeline to be a GroupedPipeline")

        if not target_pipeline.fitted_pipelines:
            raise RuntimeError("The 'target' GroupedPipeline has not been fitted")

        # Validate input shapes
        if predictions.shape[0] != basin_ids.shape[0]:
            raise ValueError(
                f"Shape mismatch: predictions ({predictions.shape[0]}) != basin_ids ({basin_ids.shape[0]})"
            )

        # Check all required basin pipelines exist
        unique_basin_ids = {str(bid) for bid in basin_ids}
        available_basin_ids = {str(bid) for bid in target_pipeline.fitted_pipelines}
        missing_basin_ids = unique_basin_ids - available_basin_ids

        if missing_basin_ids:
            raise ValueError(f"No fitted pipeline for basins: {sorted(missing_basin_ids)}")

        # Store original shape for reconstruction
        original_shape = predictions.shape

        # Prepare data for inverse transformation
        basin_id_list: list[str] = []
        prediction_list: list[float] = []

        # Flatten predictions while tracking basin associations
        for i, basin_id in enumerate(basin_ids):
            basin_str = str(basin_id)
            pred_sequence = predictions[i].flatten()

            basin_id_list.extend([basin_str] * len(pred_sequence))
            prediction_list.extend(pred_sequence.tolist())

        # Create DataFrame for inverse transformation
        # GroupedPipeline expects pandas, so create it directly
        import pandas as pd

        transform_df = pd.DataFrame(
            {
                self.group_identifier: basin_id_list,
                self.target: prediction_list,
            }
        )

        # Apply inverse transformation
        try:
            inverse_df = target_pipeline.inverse_transform(transform_df)
            inverse_values = inverse_df[self.target].values
        except Exception as e:
            raise RuntimeError(f"Inverse transformation failed: {str(e)}") from e

        # Reshape back to original dimensions
        try:
            reshaped_values = inverse_values.reshape(original_shape)
        except ValueError as e:
            raise RuntimeError(
                f"Failed to reshape inverse transformed values. "
                f"Expected shape: {original_shape}, got {inverse_values.shape}"
            ) from e

        return reshaped_values

    def get_train_dataloader(self) -> DataLoader:
        return self.train_dataloader()

    def get_val_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def get_test_dataloader(self) -> DataLoader:
        return self.test_dataloader()
