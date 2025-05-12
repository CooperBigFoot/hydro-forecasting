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


def finetune_pretrained_models(
    # --- Pretrained Model Identification ---
    gauge_ids: list[str],
    pretrained_experiment_output_dir: str | Path,
    model_types: list[str],
    pretrained_yaml_paths: list[str | Path] | str | Path,
    datamodule_config: dict[str, Any],
    training_config: dict[str, Any],
    output_dir: str | Path,
    experiment_name: str,
    select_best_from_pretrained: bool = True,
    pretrained_run_index: int | None = None,
    pretrained_attempt_index: int | None = None,
    lr_reduction_factor: float = 10.0,
    num_runs: int = 1,
    base_seed: int = 42,
    override_previous_attempts: bool = False,
) -> Result[dict[str, tuple[str, dict[str, Any]]], str]:
    """
    Fine-tune pre-trained hydrological forecasting models.

    This function orchestrates the fine-tuning process for one or more pre-trained models:
    1. Locates pre-trained checkpoints from a previous experiment
    2. Loads models with reduced learning rates
    3. Sets up data modules with the fine-tuning dataset
    4. Executes training loops for fine-tuning
    5. Manages versioned outputs and identifies best checkpoints
    6. Returns a summary of fine-tuning results

    Args:
        pretrained_experiment_output_dir: Directory containing pre-trained model checkpoints
        model_types: List of model types to fine-tune ('tide', 'tft', 'ealstm', etc.)
        pretrained_yaml_paths: List of paths to YAML files with model hyperparameters,
                              or a single path to a directory containing YAML files,
                              or a single path to a YAML file to be used for all model types
        select_best_from_pretrained: Whether to use the best checkpoint from pre-training
        pretrained_run_index: Specific run index to use when select_best_from_pretrained is False
        pretrained_attempt_index: Specific attempt index to use when select_best_from_pretrained is False
        lr_reduction_factor: Factor by which to reduce the learning rate for fine-tuning
        gauge_ids: List of basin/gauge IDs to use for fine-tuning
        datamodule_config: Configuration for the HydroInMemoryDataModule
        training_config: Configuration for PyTorch Lightning Trainer
        output_dir: Base directory for storing fine-tuning outputs
        experiment_name: Name of the fine-tuning experiment for directory organization
        num_runs: Number of independent fine-tuning runs for each model type
        base_seed: Base random seed (will be incremented for each run)
        override_previous_attempts: Whether to override existing fine-tuning outputs

    Returns:
        Result containing a dictionary mapping model types to tuples of
        (best_checkpoint_path, metrics_dict) if successful, or an error message if failed.
    """
    if not gauge_ids:
        return Failure("No gauge IDs provided for fine-tuning")

    # Resolve pretrained_yaml_paths
    if isinstance(pretrained_yaml_paths, str | Path):
        yaml_dir = Path(pretrained_yaml_paths)
        if yaml_dir.is_dir():
            resolved_yaml_paths = []
            for model_type in model_types:
                yaml_path = yaml_dir / f"{model_type.lower()}.yaml"
                if not yaml_path.exists():
                    return Failure(f"YAML file for model type '{model_type}' not found at {yaml_path}")
                resolved_yaml_paths.append(str(yaml_path))
            pretrained_yaml_paths = resolved_yaml_paths
        else:
            logger.warning(
                "Provided pretrained_yaml_paths is a string but not a directory. Using it as a single YAML path for all models."
            )
            pretrained_yaml_paths = [pretrained_yaml_paths] * len(model_types)

    if len(model_types) != len(pretrained_yaml_paths):
        return Failure("Length of model_types must match length of pretrained_yaml_paths")

    # Create the main experiment directory for fine-tuning outputs
    main_experiment_dir = Path(output_dir) / experiment_name

    try:
        main_experiment_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return Failure(f"Failed to create experiment directory {main_experiment_dir}: {e}")

    logger.info(f"Starting fine-tuning for experiment '{experiment_name}'")
    logger.info(f"Output directory: {main_experiment_dir}")
    logger.info(f"Models to fine-tune: {', '.join(model_types)}")
    logger.info(f"Number of runs per model: {num_runs}")
    logger.info(f"Learning rate reduction factor: {lr_reduction_factor}")

    # Dictionary to store the best fine-tuned checkpoint and metrics for each model type
    all_models_overall_best_results = {}

    # Iterate through each model type to fine-tune
    for i, (current_model_type, current_yaml_path) in enumerate(zip(model_types, pretrained_yaml_paths, strict=False)):
        logger.info(f"Processing model ({i + 1}/{len(model_types)}): {current_model_type}")

        # Set up paths for fine-tuning outputs for this model type
        model_checkpoints_base_dir = main_experiment_dir / checkpoint_manager.CHECKPOINTS_DIR_NAME / current_model_type
        model_logs_base_dir = main_experiment_dir / checkpoint_manager.LOGS_DIR_NAME / current_model_type

        try:
            model_checkpoints_base_dir.mkdir(parents=True, exist_ok=True)
            model_logs_base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directories for {current_model_type}: {e}")
            all_models_overall_best_results[current_model_type] = (None, {"error": str(e)})
            continue

        # Locate the pre-trained model checkpoint
        pretrained_checkpoint_path_result = checkpoint_manager.get_checkpoint_path_to_load(
            base_checkpoint_load_dir=Path(pretrained_experiment_output_dir),
            model_type=current_model_type,
            select_overall_best=select_best_from_pretrained,
            specific_run_index=pretrained_run_index,
            specific_attempt_index=pretrained_attempt_index,
        )

        if not isinstance(pretrained_checkpoint_path_result, Success):
            error_msg = pretrained_checkpoint_path_result.failure()
            logger.warning(f"Could not find pre-trained checkpoint for {current_model_type}: {error_msg}")
            all_models_overall_best_results[current_model_type] = (
                None,
                {"error": f"Checkpoint not found: {error_msg}"},
            )
            continue

        pretrained_checkpoint_path = pretrained_checkpoint_path_result.unwrap()
        logger.info(f"Found pre-trained checkpoint for {current_model_type}: {pretrained_checkpoint_path}")

        # List to store results of all fine-tuning runs for this model type
        current_model_run_results = []

        # Load model hyperparameters to configure the data module correctly
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

            # Create a copy of datamodule_config with the correct input/output lengths
            current_datamodule_config = datamodule_config.copy()
            current_datamodule_config["input_length"] = input_length
            current_datamodule_config["output_length"] = output_length

            # Set up the data module for fine-tuning
            datamodule = HydroInMemoryDataModule(list_of_gauge_ids_to_process=gauge_ids, **current_datamodule_config)

            # Prepare data and verify setup
            datamodule.prepare_data()
            datamodule.setup()

        except Exception as e:
            logger.error(f"Failed to create or setup datamodule for {current_model_type}: {e}")
            all_models_overall_best_results[current_model_type] = (
                None,
                {"error": f"DataModule setup failed: {str(e)}"},
            )
            continue

        # Perform multiple fine-tuning runs with different seeds
        for run_idx in range(num_runs):
            logger.info(f"Starting fine-tuning run {run_idx + 1}/{num_runs} for {current_model_type}")

            # Set seed for reproducibility
            current_run_seed = base_seed + run_idx
            pl.seed_everything(current_run_seed, workers=True)
            logger.info(f"Using seed: {current_run_seed}")

            # Determine versioned output paths for this fine-tuning run
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

            # Load the pre-trained model with reduced learning rate
            try:
                model, model_hp = model_factory.load_pretrained_model(
                    model_type=current_model_type,
                    yaml_path=current_yaml_path,
                    checkpoint_path=str(pretrained_checkpoint_path),
                    lr_factor=lr_reduction_factor,
                )
                logger.info(
                    f"Loaded pre-trained model with original LR: {model_hp.get('original_lr')} and fine-tuning LR: {model_hp.get('learning_rate')}"
                )

            except Exception as e:
                logger.error(f"Failed to load pre-trained model for {current_model_type}, run {run_idx}: {e}")
                continue

            # Set up callbacks for training
            checkpoint_filename = (
                f"{current_model_type}-finetune-run{run_idx}-{attempt_name}-{{epoch:02d}}-{{val_loss:.4f}}"
            )
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

            # Configure the trainer
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

            # Configure precision if specified
            precision = training_config.get("precision")
            if precision is not None:
                valid_precision_values = ["32-true", "32", "16-mixed", "bf16-mixed"]
                if str(precision) in valid_precision_values:
                    trainer_kwargs["precision"] = precision
                else:
                    logger.warning(f"Invalid precision value: {precision}. Using default.")

            # Configure matmul precision if specified
            matmul_precision = training_config.get("torch_float32_matmul_precision")
            if matmul_precision is not None:
                valid_matmul_values = ["highest", "high", "medium"]
                if matmul_precision in valid_matmul_values:
                    torch.set_float32_matmul_precision(matmul_precision)
                else:
                    logger.warning(f"Invalid torch_float32_matmul_precision value: {matmul_precision}. Not setting.")

            # Execute the fine-tuning
            status = "failed"
            try:
                trainer = pl.Trainer(**trainer_kwargs)
                trainer.fit(model, datamodule=datamodule)

                # Collect results
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
                logger.error(f"Fine-tuning failed for {current_model_type}, run {run_idx}: {e}")

            finally:
                if tb_logger:
                    tb_logger.finalize(status)

        # Process results for all runs of this model type
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

                logger.info(f"Overall best fine-tuned model for {current_model_type}: {relative_best_path}")
                logger.info(f"Best metrics: {overall_best_metrics}")

                all_models_overall_best_results[current_model_type] = (overall_best_model_path, overall_best_metrics)

            except Exception as e:
                logger.error(f"Error recording overall best model for {current_model_type}: {e}")
                all_models_overall_best_results[current_model_type] = (
                    None,
                    {"error": f"Error recording best model: {str(e)}"},
                )
        else:
            logger.warning(f"No successful fine-tuning runs for {current_model_type}")
            all_models_overall_best_results[current_model_type] = (None, {"error": "No successful runs"})

    # Return the overall results
    if not all_models_overall_best_results:
        return Failure("No models were successfully fine-tuned")

    return Success(all_models_overall_best_results)
