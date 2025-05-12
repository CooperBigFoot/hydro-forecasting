import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from returns.result import Failure, Result, Success

from ..data.in_memory_datamodule import HydroInMemoryDataModule
from ..experiment_utils import checkpoint_manager
from ..model_evaluation.hp_from_yaml import hp_from_yaml
from ..models import model_factory

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    pl.seed_everything(seed, workers=True)


def train_model_from_scratch(
    gauge_ids: list[str],
    datamodule_config: dict[str, Any],
    training_config: dict[str, Any],
    output_dir: str,
    model_types: list[str],
    yaml_paths: list[str] | str,
    experiment_name: str,
    num_runs: int = 1,
    base_seed: int = 42,
    override_previous_attempts: bool = False,
) -> Result[dict[str, tuple[str, dict[str, float]]], str]:
    """
    Train one or more hydrological forecasting models from scratch.

    This function serves as a high-level orchestrator for training deep learning models
    within the hydrological forecasting framework. It manages the entire process from
    data preparation to model training and checkpoint management.

    For each model type specified, the function:
    1. Creates and sets up a HydroInMemoryDataModule
    2. Performs multiple independent training runs with different random seeds
    3. Monitors validation loss and saves the best checkpoint for each run
    4. Identifies the overall best checkpoint across all runs
    5. Records this best checkpoint in an overall_best_model_info.txt file

    The function creates a versioned directory structure for outputs

    Args:
        gauge_ids: List of basin/gauge IDs to use for training
        datamodule_config: Configuration for the HydroInMemoryDataModule, should include:
            - region_time_series_base_dirs: Dict mapping region prefixes to time series directories
            - region_static_attributes_base_dirs: Dict mapping region prefixes to static attributes directories
            - path_to_preprocessing_output_directory: Path for processed data cache
            - group_identifier: Column name for basin grouping (default: "gauge_id")
            - batch_size: Training batch size
            - forcing_features: List of forcing feature column names
            - static_features: List of static feature column names
            - target: Target variable column name
            - num_workers: Number of workers for DataLoader
            - min_train_years: Minimum required years of data for training
            - train_prop, val_prop, test_prop: Data split proportions
            - max_imputation_gap_size: Maximum gap size for imputation
            - chunk_size: Number of basins per chunk
            - is_autoregressive: Whether to use autoregressive modeling
            - preprocessing_configs: Dictionary with pipeline objects for preprocessing
              (must include instantiated Pipeline or GroupedPipeline objects)
            Note: input_length and output_length will be extracted from model hyperparameters
        training_config: Configuration for PyTorch Lightning Trainer, can include:
            - max_epochs: Maximum number of training epochs
            - accelerator: "cpu", "gpu", "tpu", etc.
            - devices: Number of devices to use
            - precision: Numerical precision for training
            - early_stopping_patience: Patience for early stopping
            - reload_dataloaders_every_n_epochs: Frequency for reloading dataloaders
            - torch_float32_matmul_precision: Precision for float32 matmul operations
        output_dir: Base directory for storing outputs (checkpoints, logs)
        model_types: List of model types to train ('tide', 'tft', 'ealstm', etc.)
        yaml_paths: List of paths to YAML files with model hyperparameters,
                   or a single path to a directory containing YAML files named after model types
                   (e.g., '/path/to/yamls/tide.yaml' for model_type 'tide'),
                   or a single path to a YAML file to be used for all model types
        experiment_name: Name of the experiment for directory organization
        num_runs: Number of independent training runs for each model type
        base_seed: Base random seed (will be incremented for each run)
        override_previous_attempts: Whether to override existing outputs

    Returns:
        Result containing a dictionary mapping model types to tuples of
        (best_checkpoint_path, metrics_dict) if successful, or an error message if failed.

        The metrics_dict includes:
        - val_loss: Best validation loss achieved
        - run_index: Index of the run that achieved the best result
        - seed: Random seed used for the best run

        Example:
        {
            'tide': ('/path/to/checkpoint.ckpt', {'val_loss': 0.123, 'run_index': 2, 'seed': 44}),
            'tft': ('/path/to/checkpoint.ckpt', {'val_loss': 0.456, 'run_index': 0, 'seed': 42})
        }
    """
    if not gauge_ids:
        return Failure("No gauge IDs provided for training")

    if isinstance(yaml_paths, str):
        yaml_dir = Path(yaml_paths)
        if yaml_dir.is_dir():
            resolved_yaml_paths = []
            for model_type in model_types:
                yaml_path = yaml_dir / f"{model_type.lower()}.yaml"
                if not yaml_path.exists():
                    return Failure(f"YAML file for model type '{model_type}' not found at {yaml_path}")
                resolved_yaml_paths.append(str(yaml_path))
            yaml_paths = resolved_yaml_paths
        else:
            logger.warning(
                "Provided yaml_paths is a string but not a directory. Using it as a single YAML path for all models."
            )
            yaml_paths = [yaml_paths] * len(model_types)

    if len(model_types) != len(yaml_paths):
        return Failure("Length of model_types must match length of yaml_paths")

    main_experiment_dir = Path(output_dir) / experiment_name

    try:
        main_experiment_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return Failure(f"Failed to create experiment directory {main_experiment_dir}: {e}")

    logger.info(f"Starting training for experiment '{experiment_name}'")
    logger.info(f"Output directory: {main_experiment_dir}")
    logger.info(f"Models to train: {', '.join(model_types)}")
    logger.info(f"Number of runs per model: {num_runs}")

    all_models_overall_best_results = {}

    for i, (current_model_type, current_yaml_path) in enumerate(zip(model_types, yaml_paths, strict=False)):
        logger.info(f"Processing model ({i + 1}/{len(model_types)}): {current_model_type}")

        model_checkpoints_base_dir = main_experiment_dir / checkpoint_manager.CHECKPOINTS_DIR_NAME / current_model_type
        model_logs_base_dir = main_experiment_dir / checkpoint_manager.LOGS_DIR_NAME / current_model_type

        try:
            model_checkpoints_base_dir.mkdir(parents=True, exist_ok=True)
            model_logs_base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directories for {current_model_type}: {e}")
            all_models_overall_best_results[current_model_type] = (None, {"error": str(e)})
            continue

        current_model_run_results = []

        try:
            model_hp = hp_from_yaml(current_model_type, current_yaml_path)
            input_length = model_hp.get("input_len")
            output_length = model_hp.get("output_len")

            if input_length is None or output_length is None:
                logger.error(f"Missing input_len or output_len in hyperparameters for {current_model_type}")
                all_models_overall_best_results[current_model_type] = (
                    None,
                    {"error": "Missing required hyperparameters"},
                )
                continue

            current_datamodule_config = datamodule_config.copy()
            current_datamodule_config["input_length"] = input_length
            current_datamodule_config["output_length"] = output_length

            datamodule = HydroInMemoryDataModule(list_of_gauge_ids_to_process=gauge_ids, **current_datamodule_config)

            # This is for sanity check. I want this to fail before the training starts
            datamodule.prepare_data()
            datamodule.setup()

        except Exception as e:
            logger.error(f"Failed to create or setup datamodule for {current_model_type}: {e}")
            all_models_overall_best_results[current_model_type] = (
                None,
                {"error": f"DataModule setup failed: {str(e)}"},
            )
            continue

        for run_idx in range(num_runs):
            logger.info(f"Starting run {run_idx + 1}/{num_runs} for {current_model_type}")

            current_run_seed = base_seed + run_idx
            set_seed(current_run_seed)
            logger.info(f"Using seed: {current_run_seed}")

            checkpoint_run_attempt_path_result = checkpoint_manager.determine_output_run_attempt_path(
                base_model_output_dir=model_checkpoints_base_dir,
                run_index=run_idx,
                override_previous_attempts=override_previous_attempts,
            )

            if not isinstance(checkpoint_run_attempt_path_result, Success):
                logger.error(
                    f"Failed to determine checkpoint path for run {run_idx}: {checkpoint_run_attempt_path_result.failure()}"
                )
                continue

            specific_checkpoint_dir = checkpoint_run_attempt_path_result.unwrap()

            log_run_attempt_path_result = checkpoint_manager.determine_log_run_attempt_path(
                base_model_log_dir=model_logs_base_dir,
                run_index=run_idx,
                override_previous_attempts=override_previous_attempts,
            )

            if not isinstance(log_run_attempt_path_result, Success):
                logger.error(f"Failed to determine log path for run {run_idx}: {log_run_attempt_path_result.failure()}")
                continue

            specific_log_dir = log_run_attempt_path_result.unwrap()
            attempt_name = specific_log_dir.name

            try:
                model, model_hp = model_factory.create_model(current_model_type, current_yaml_path)
            except Exception as e:
                logger.error(f"Failed to create model for {current_model_type}, run {run_idx}: {e}")
                continue

            checkpoint_filename = f"{current_model_type}-run{run_idx}-{attempt_name}-{{epoch:02d}}-{{val_loss:.4f}}"
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=str(specific_checkpoint_dir),
                filename=checkpoint_filename,
                save_top_k=1,
                mode="min",
                save_last=True,
            )

            early_stopping_patience = training_config.get("early_stopping_patience", 10)
            early_stopping_callback = EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                verbose=True,
                mode="min",
            )

            lr_monitor_callback = LearningRateMonitor(logging_interval="step")

            tb_logger = TensorBoardLogger(
                save_dir=str(model_logs_base_dir),
                name=f"run_{run_idx}",
                version=attempt_name,
                default_hp_metric=False,
            )

            callbacks_list = [early_stopping_callback, checkpoint_callback, lr_monitor_callback]

            trainer_kwargs = {
                "max_epochs": training_config.get("max_epochs", 100),
                "accelerator": training_config.get("accelerator", "auto"),
                "devices": training_config.get("devices", 1),
                "enable_progress_bar": training_config.get("enable_progress_bar", True),
                "logger": tb_logger,
                "callbacks": callbacks_list,
                "num_sanity_val_steps": training_config.get("num_sanity_val_steps", 2),
                "reload_dataloaders_every_n_epochs": training_config.get("reload_dataloaders_every_n_epochs", 1),
            }

            precision = training_config.get("precision")
            if precision is not None:
                valid_precision_values = ["32-true", "32", "16-mixed", "bf16-mixed"]
                if str(precision) in valid_precision_values:
                    trainer_kwargs["precision"] = precision
                else:
                    logger.warning(f"Invalid precision value: {precision}. Using default.")

            matmul_precision = training_config.get("torch_float32_matmul_precision")
            if matmul_precision is not None:
                valid_matmul_values = ["highest", "high", "medium"]
                if matmul_precision in valid_matmul_values:
                    torch.set_float32_matmul_precision(matmul_precision)
                else:
                    logger.warning(f"Invalid torch_float32_matmul_precision value: {matmul_precision}. Not setting.")

            status = "failed"
            try:
                trainer = pl.Trainer(**trainer_kwargs)
                trainer.fit(model, datamodule=datamodule)

                # v. Result Collection
                best_model_path = checkpoint_callback.best_model_path
                best_model_score = checkpoint_callback.best_model_score

                if best_model_path and best_model_score is not None:
                    best_model_score_value = best_model_score.item()
                    logger.info(f"Run {run_idx} completed with best val_loss: {best_model_score_value}")

                    # Store the result for this run
                    current_model_run_results.append(
                        (
                            best_model_path,
                            {"val_loss": best_model_score_value, "run_index": run_idx, "seed": current_run_seed},
                        )
                    )
                    status = "success"
                else:
                    logger.warning(f"No best model path or score found for run {run_idx}")

            except Exception as e:
                logger.error(f"Training failed for {current_model_type}, run {run_idx}: {e}")

            finally:
                if tb_logger:
                    tb_logger.finalize(status)

        if current_model_run_results:
            # Sort by validation loss (ascending)
            sorted_results = sorted(current_model_run_results, key=lambda x: x[1]["val_loss"])
            overall_best_model_path, overall_best_metrics = sorted_results[0]

            # Get relative path for overall best model
            try:
                relative_best_path = Path(overall_best_model_path).relative_to(model_checkpoints_base_dir)

                update_result = checkpoint_manager.update_overall_best_model_info_file(
                    model_checkpoints_output_dir=model_checkpoints_base_dir,
                    best_checkpoint_relative_path=str(relative_best_path),
                )

                if not isinstance(update_result, Success):
                    logger.error(
                        f"Failed to update best model info for {current_model_type}: {update_result.failure()}"
                    )

                logger.info(f"Overall best model for {current_model_type}: {relative_best_path}")
                logger.info(f"Best metrics: {overall_best_metrics}")

                all_models_overall_best_results[current_model_type] = (overall_best_model_path, overall_best_metrics)

            except Exception as e:
                logger.error(f"Error recording overall best model for {current_model_type}: {e}")
                all_models_overall_best_results[current_model_type] = (
                    None,
                    {"error": f"Error recording best model: {str(e)}"},
                )
        else:
            logger.warning(f"No successful runs for {current_model_type}")
            all_models_overall_best_results[current_model_type] = (None, {"error": "No successful runs"})

    if not all_models_overall_best_results:
        return Failure("No models were successfully trained")

    return Success(all_models_overall_best_results)
